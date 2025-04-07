import cv2
import glob
import math
import numpy as np
import os
import time
import torch

from pathlib import Path
from PIL import Image
from threading import Thread

from Data.utils import IMG_FORMATS, VID_FORMATS
from Utils.ops import clean_str


class IncrementPath:
    def __init__(self, path, exist_ok=False, sep="", mkdir=False):
        self.path = Path(path)
        self.exist_ok = exist_ok
        self.sep = sep
        self.mkdir = mkdir

    def __call__(self):
        return self.get()

    def get(self):
        if self.path.exists() and not self.exist_ok:
            self.path, self.suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

            for n in range(2, 9999):
                p = f"{path}{self.sep}{n}{self.suffix}"
                if not os.path.exists(p):
                    break
            path = Path(p)

        if self.mkdir:
            self.path.mkdir(parents=True, exist_ok=True)

        return self.path

    def __str__(self):
        return str(self.get())


class LoadStreams:
    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        torch.backends.cudnn.benchmark = True
        self.buffer = buffer
        self.running = True
        self.mode = "stream"
        self.vid_stride = vid_stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [[] for _ in range(n)]
        self.shape = [[] for _ in range(n)]
        self.sources = [clean_str(x) for x in sources]
        for i, s in enumerate(sources):
            st = f"{i + 1}/{n}: {s}... "
            s = eval(s) if s.isnumeric() else s
            self.caps[i] = cv2.VideoCapture(s)
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            success, im = self.caps[i].read()
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            print(f"{st}Success ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print("")

    def update(self, i, cap, stream):
        n, f = 0, self.frames[i]
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:
                n += 1
                cap.grab()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        print("WARNING Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)

    def close(self):
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        for cap in self.caps:
            try:
                cap.release()
            except Exception as e:
                print(f"WARNING Could not release VideoCapture object: {e}")
        cv2.destroyAllWindows()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    print(f"WARNING Waiting for stream {i}")

            if self.buffer:
                images.append(x.pop(0))

            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self):
        return self.bs


class LoadImagesAndVideos:
    def __init__(self, path, batch=1, vid_stride=1):
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))
            elif os.path.isfile(a):
                files.append(a)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv
        self.ni = ni
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}.")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:
                            self.count += 1
                            self.cap.release()
                else:
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = cv2.imread(path)
                if im0 is None:
                    print(f"WARNING Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1
                if self.count >= self.ni:
                    break

        return paths, imgs, info

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        return math.ceil(self.nf / self.bs)


class LoadPilAndNumpy:
    def __init__(self, im0):
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        self.count = 0
        return self


class LoadTensor:
    def __init__(self, im0) -> None:
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            print(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:
            print(
                f"WARNING torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. "
                f"Dividing input by 255."
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self):
        return self.bs


LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos)
