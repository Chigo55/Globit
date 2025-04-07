import glob
import numpy as np
import torch
import requests

from pathlib import Path
from PIL import Image

from Utils import ROOT, downloads
from Utils.ops import clean_url, url2file
from Data.utils import IMG_FORMATS, VID_FORMATS, SourceTypes
from Data.loaders import LoadTensor, LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LOADERS


class LoadSource:
    def __init__(self, source, batch=1, vid_stride=1, buffer=False):
        self.source = source
        self.batch = batch
        self.vid_stride = vid_stride
        self.buffer = buffer

    def __call__(self):
        return self.load_inference_source(self.source, self.batch, self.vid_stride, self.buffer)

    def load_inference_source(self, source, batch, vid_stride, buffer):
        source, stream, from_img, in_memory, tensor = self.check_source(source)
        source_type = source.source_type if in_memory else SourceTypes(stream, from_img, tensor)

        if tensor:
            dataset = LoadTensor(source)
        elif in_memory:
            dataset = source
        elif stream:
            dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
        elif from_img:
            dataset = LoadPilAndNumpy(source)
        else:
            dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

        setattr(dataset, "source_type", source_type)

        return dataset

    def check_source(self, source):
        webcam, from_img, in_memory, tensor = False, False, False, False

        if isinstance(source, (str, int, Path)):
            source = str(source)
            is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
            is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
            webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
            if is_url and is_file:
                source = self.check_file(source)

        elif isinstance(source, LOADERS):
            in_memory = True
        elif isinstance(source, (list, tuple)):
            source = self.autocast_list(source)
            from_img = True
        elif isinstance(source, (Image.Image, np.ndarray)):
            from_img = True
        elif isinstance(source, torch.Tensor):
            tensor = True
        else:
            raise TypeError("Unsupported image type.")

        return source, webcam, from_img, in_memory, tensor

    def check_file(self, file, suffix="", download=True, download_dir=".", hard=True):
        self.check_suffix(file, suffix)
        file = str(file).strip()
        if (
            not file
            or ("://" not in file and Path(file).exists())
            or file.lower().startswith("grpc://")
        ):
            return file
        elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            url = file
            file = Path(download_dir) / url2file(file)
            if file.exists():
                print(f"Found {clean_url(url)} locally at {file}")
            else:
                downloads.safe_download(url=url, file=file, unzip=False)
            return str(file)
        else:
            files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))
            if not files and hard:
                raise FileNotFoundError(f"'{file}' does not exist")
            elif len(files) > 1 and hard:
                raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
            return files[0] if len(files) else []

    def autocast_list(self, source):
        files = []
        for im in source:
            if isinstance(im, (str, Path)):
                files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
            elif isinstance(im, (Image.Image, np.ndarray)):
                files.append(im)
            else:
                raise TypeError(f"type {type(im).__name__} is not a supported prediction source type. \n")

        return files

    def check_suffix(self, file="yolov8n.pt", suffix=".pt", msg=""):
        if file and suffix:
            if isinstance(suffix, str):
                suffix = (suffix,)
            for f in file if isinstance(file, (list, tuple)) else [file]:
                s = Path(f).suffix.lower().strip()
                if len(s):
                    assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"
