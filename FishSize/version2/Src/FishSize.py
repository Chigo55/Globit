import cv2
import json
import math
import mimetypes
import torch

from pathlib import Path
from ultralytics import YOLO

from Src.Utils import IncrementPath, IMAGE_FORMATS, VIDEO_FORMATS
from Src.Utils.loaders import LoadImages, LoadVideos, LoadWebcam
from Src.Utils.interface import (
    IModelLoader,
    IDataLoader,
    IDataPredictor,
    IResultSaver,
    ISizeEstimator,
    IVisualizer,
)


class ModelLoader(IModelLoader):
    """모델 로더 구현 클래스."""

    def __init__(self, weight, device):
        """
        ModelLoader를 초기화합니다.

        Args:
            weight (Any): 모델 가중치 파일 경로.
            device (Any): 사용할 디바이스 (예: "cpu", "cuda").

        Raises:
            FileNotFoundError: 지정된 가중치 파일이 존재하지 않을 경우.
            ValueError: CUDA 디바이스를 선택했으나 사용 불가능한 경우.
        """
        self.weight = weight
        self.device = device if device != "" else "cpu"

        p = Path(self.weight)
        if not p.is_file():
            raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {self.weight}")

        if self.device.lower() == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA가 사용 가능하지 않습니다. 'cpu'를 사용하거나 CUDA 환경을 확인하세요.")

    def load_model(self):
        """
        모델을 로드하여 지정된 디바이스로 이동시킵니다.

        Returns:
            모델 객체.

        Raises:
            RuntimeError: 모델 로딩 중 오류가 발생할 경우.
        """
        try:
            model = YOLO(self.weight)
            model = model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"모델 로딩 중 오류 발생: {e}")
        return model


class DataLoader(IDataLoader):
    """데이터 로더 구현 클래스."""

    def __init__(self, source, device, use_webcam=None):
        """
        DataLoader를 초기화합니다.

        Args:
            source (str): 데이터 경로 또는 웹캠 파이프.
            device (str): 사용할 디바이스.
            use_webcam (bool): 웹캠 사용 여부.

        Raises:
            FileNotFoundError: 지정된 소스가 존재하지 않을 경우.
            ValueError: 소스에서 이미지나 비디오 파일을 찾을 수 없을 경우.
        """
        self.source = Path(source)
        self.device = device
        if use_webcam is None:
            self.use_webcam = str(source) == "0"
        else:
            self.use_webcam = bool(use_webcam)

        if self.use_webcam:
            self.files = []
            return

        if "**" in source:
            base_str = source.split("**")[0]
            base_path = Path(base_str)
            files = sorted(base_path.rglob("*"))
        elif "*" in str(self.source):
            files = sorted(Path().glob(str(self.source)))
        elif self.source.is_dir():
            files = sorted(self.source.glob("*.*"))
        elif self.source.is_file():
            files = [self.source]
        else:
            raise FileNotFoundError(f"오류: {self.source}가 존재하지 않습니다.")

        self.images = []
        self.videos = []
        for x in files:
            mime = mimetypes.guess_type(str(x))[0]
            if mime is not None:
                file_ext = mime.split("/")[-1]
                if file_ext.lower() in IMAGE_FORMATS:
                    self.images.append((x, mime))
                elif file_ext.lower() in VIDEO_FORMATS:
                    self.videos.append((x, mime))
        self.files = self.images + self.videos

        if len(self.files) == 0:
            raise ValueError(f"{self.source}에서 이미지나 비디오를 찾을 수 없습니다.\n지원되는 형식은:\n 이미지: {IMAGE_FORMATS}\n 비디오: {VIDEO_FORMATS}")

    def load_data(self):
        """
        데이터를 로드하여 제너레이터 형태로 반환합니다.

        Yields:
            LoadImages, LoadVideos, 또는 LoadWebcam 객체.
        """
        if self.use_webcam:
            yield LoadWebcam(self.source, self.device)
        else:
            for file_path, mtype in self.files:
                main_type = mtype.split("/")[0]
                if main_type == "image":
                    yield LoadImages(file_path, self.device)
                elif main_type == "video":
                    yield LoadVideos(file_path, self.device)
                else:
                    raise ValueError(f"지원되지 않는 미디어 유형입니다: {mtype}")


class DataPredictor(IDataPredictor):
    """데이터 예측기 구현 클래스."""

    def __init__(self, model, device, conf_thres, iou_thres, verbose):
        """
        DataPredictor를 초기화합니다.

        Args:
            model (Any): 모델 객체.
            device (str): 사용할 디바이스.
            conf_thres (float): 객체 신뢰도 임계값.
            iou_thres (float): NMS에 사용할 IOU 임계값.
            verbose (bool): 상세 로그 출력 여부.
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def predict_data(self, data):
        """
        입력 데이터를 모델로 예측합니다.

        Args:
            data (torch.Tensor): 전처리된 입력 데이터 텐서.

        Returns:
            tuple: (예측 결과, 모델 입력 텐서)

        Raises:
            RuntimeError: 모델 예측 중 오류가 발생할 경우.
        """
        data = data.to(self.device)

        if data.ndimension() == 3:
            data = data.unsqueeze(0)

        try:
            results = self.model(data, conf=self.conf_thres, iou=self.iou_thres, verbose=self.verbose)
        except Exception as e:
            raise RuntimeError(f"모델 예측 중 에러 발생: {e}")

        return results, data


class SizeEstimator(ISizeEstimator):
    """크기 추정기 구현 클래스."""

    @staticmethod
    def calc_dist(kp1, kp2):
        """
        두 점 사이의 유클리드 거리를 계산합니다.

        Args:
            kp1 (tuple): 첫 번째 점의 좌표 (x, y).
            kp2 (tuple): 두 번째 점의 좌표 (x, y).

        Returns:
            float: 두 점 사이의 거리.
        """
        return math.sqrt((kp2[0] - kp1[0]) ** 2 + (kp2[1] - kp1[1]) ** 2)

    @staticmethod
    def ensure_list(x):
        """
        입력값이 리스트인지 확인하고, 아니라면 리스트로 변환합니다.

        Args:
            x: 임의의 값.

        Returns:
            list: 입력값이 리스트이면 그대로, 아니면 리스트로 변환한 결과.
        """
        return x if isinstance(x, list) else [x]

    def estimate_size(self, data, result):
        """
        입력 이미지와 예측 결과를 바탕으로 객체의 크기를 추정합니다.

        Args:
            data (numpy.ndarray): 원본 이미지 데이터.
            result: 모델 예측 결과 객체.

        Returns:
            list: 각 객체에 대한 [높이, 너비] 목록.
        """
        sizes = []
        height, width, _ = data.shape

        keypoints = self.ensure_list(result.keypoints.xyn.tolist())

        for kp in keypoints:
            if len(kp) != 4:
                sizes.append([0.0, 0.0])
            else:
                kp1, kp2, kp3, kp4 = kp
                w = self.calc_dist(kp1, kp2) * width
                h = self.calc_dist(kp3, kp4) * height
                sizes.append([h, w])
        return sizes


class Visualizer(IVisualizer):
    """시각화 구현 클래스."""

    @staticmethod
    def ensure_list(x):
        """
        입력값이 리스트인지 확인하고, 아니라면 리스트로 변환합니다.

        Args:
            x: 임의의 값.

        Returns:
            list: 입력값을 리스트 형태로 반환.
        """
        return x if isinstance(x, list) else [x]

    @staticmethod
    def draw_bbox(data, bbox):
        """
        이미지에 경계 상자를 그립니다.

        Args:
            data (numpy.ndarray): 이미지 데이터.
            bbox (list): 경계 상자 좌표 [x1, y1, x2, y2].

        Returns:
            numpy.ndarray: 경계 상자가 그려진 이미지.
        """
        x1, y1, x2, y2 = bbox
        data = cv2.rectangle(data, (x1, y1), (x2, y2), (255, 0, 255), 5, cv2.LINE_AA)
        return data

    @staticmethod
    def draw_keypoint(data, keypoint):
        """
        이미지에 키포인트를 그립니다.

        Args:
            data (numpy.ndarray): 이미지 데이터.
            keypoint (list): 키포인트 좌표 리스트.

        Returns:
            numpy.ndarray: 키포인트가 그려진 이미지.
        """
        for i, (kp_x, kp_y) in enumerate(keypoint):
            data = cv2.circle(data, (kp_x, kp_y), 5, (0, 0, 255), -1)
        return data

    @staticmethod
    def put_text(data, cls, conf, obj_size, pos):
        """
        이미지에 텍스트를 출력합니다.

        Args:
            data (numpy.ndarray): 이미지 데이터.
            cls (int 또는 str): 객체 클래스.
            conf (float): 객체 신뢰도.
            obj_size (list): 객체의 크기 [높이, 너비].
            pos (list): 텍스트 출력 위치를 위한 좌표 [x, y, _, _].

        Returns:
            numpy.ndarray: 텍스트가 출력된 이미지.
        """
        x, y, _, _ = pos
        text = f"{cls}:{conf * 100:.2f}% | H:{obj_size[0]:.2f} W:{obj_size[1]:.2f}"
        data = cv2.putText(data, text, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        return data

    def draw(self, data, result, size):
        """
        이미지에 예측 결과(경계 상자, 키포인트, 텍스트)를 그립니다.

        Args:
            data (numpy.ndarray): 원본 이미지 데이터.
            result: 모델 예측 결과 객체.
            size (list): 추정된 객체 크기 목록.

        Returns:
            tuple: (원본 이미지, 주석이 추가된 이미지)
        """
        bboxes = self.ensure_list(result.boxes.xyxyn.tolist())
        classes = self.ensure_list(result.boxes.cls.tolist())
        confs = self.ensure_list(result.boxes.conf.tolist())
        keypoints = self.ensure_list(result.keypoints.xyn.tolist())
        sizes = self.ensure_list(size)

        height, width, _ = data.shape
        annot = data.copy()

        for i, (bbox, cls, conf, kps, obj_size) in enumerate(
            zip(bboxes, classes, confs, keypoints, sizes)
        ):
            bbox = [int(x * width) if j in [0, 2] else int(x * height) for j, x in enumerate(bbox)]
            pos = bbox.copy()
            kps = [[int(x * width), int(y * height)] for j, (x, y) in enumerate(kps)]

            annot = self.draw_bbox(annot, bbox)
            annot = self.draw_keypoint(annot, kps)
            annot = self.put_text(annot, cls, conf, obj_size, pos)
        return data, annot

    def visualize(self, data):
        """
        이미지나 비디오 프레임을 화면에 표시합니다.

        Args:
            data (numpy.ndarray): 시각화할 이미지 데이터.
        """
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", data.shape[1] // 2, data.shape[0] // 2)
        cv2.imshow("Video", data)
        cv2.waitKey(0)


class ResultSaver(IResultSaver):
    """결과 저장기 구현 클래스."""

    def __init__(self, save_path, exist_ok=False):
        """
        ResultSaver를 초기화합니다.

        Args:
            save_path (str): 결과를 저장할 경로.
            exist_ok (bool): 기존 디렉토리가 있어도 덮어쓸지 여부.
        """
        self.save_path = Path(save_path)
        self.exist_ok = exist_ok

        # self.save_path.mkdir(parents=True, exist_ok=self.exist_ok)

        self.file_name = None
        self.writer_on = False
        self.video_writer = None

    def set_video_writer(self, data, filename):
        """
        비디오 작성을 위한 VideoWriter 객체를 초기화합니다.

        Args:
            data (numpy.ndarray): 초기 프레임 데이터.
            filename (str): 저장할 파일 이름의 기본 부분.

        Raises:
            ValueError: 유효하지 않은 프레임인 경우.
            RuntimeError: VideoWriter 객체를 열 수 없는 경우.
        """
        if data is None or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("유효하지 않은 프레임입니다. VideoWriter 초기화를 할 수 없습니다.")

        height, width, _ = data.shape
        writer_path = self.save_path / f"{filename}_pred.avi"
        self.video_writer = cv2.VideoWriter(
            str(writer_path),
            cv2.VideoWriter_fourcc(*'XVID'),
            30.0,
            (width, height)
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"VideoWriter를 열 수 없습니다: {writer_path}")

        self.writer_on = True

    def off_video_writer(self):
        """
        VideoWriter 객체를 해제하고, 작성 모드를 종료합니다.
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.writer_on = False

    def save_img(self, data, filename):
        """
        이미지를 파일로 저장합니다.

        Args:
            data (numpy.ndarray): 저장할 이미지 데이터.
            filename (str): 저장할 파일 이름의 기본 부분.
        """
        image_path = self.save_path / f"{filename}_pred.jpg"
        cv2.imwrite(str(image_path), data)

    def save_vid(self, data):
        """
        비디오 프레임을 파일에 저장합니다.

        Args:
            data (numpy.ndarray): 저장할 프레임 데이터.

        Raises:
            RuntimeError: VideoWriter가 초기화되지 않은 경우.
        """
        if self.video_writer is not None:
            self.video_writer.write(data)
        else:
            raise RuntimeError("VideoWriter가 초기화되지 않았습니다.")

    def save(self, mtype, data, filename):
        """
        미디어 유형에 따라 결과를 저장합니다.

        Args:
            mtype (str): 미디어 유형 ("image", "video", "webcam").
            data (numpy.ndarray): 저장할 이미지 또는 프레임 데이터.
            filename (str): 저장할 파일 이름의 기본 부분.

        Raises:
            ValueError: 지원되지 않는 미디어 유형일 경우.
        """
        if mtype == "image":
            self.save_img(data, filename)
        elif mtype in ("video", "webcam"):
            if (self.file_name != filename) or (not self.writer_on):
                if self.writer_on:
                    self.off_video_writer()
                self.file_name = filename
                self.set_video_writer(data, filename)
            self.save_vid(data)
        else:
            raise ValueError(f"지원되지 않는 미디어 유형입니다: {mtype}")

    def save_txt(self, result, size, filename):
        """
        예측 결과를 JSON 형식의 텍스트 파일로 저장합니다.

        Args:
            result: 모델 예측 결과 객체.
            size (list): 추정된 객체 크기 목록.
            filename (str): 저장할 파일 이름의 기본 부분.
        """
        txt_path = self.save_path / "label" / f"{filename}_pred.json"
        txt_path.mkdir(parents=True, exist_ok=self.exist_ok)
        data = {
            "bbox": result.boxes.xyxyn.tolist(),
            "conf": result.boxes.conf.tolist(),
            "cls": result.boxes.cls.tolist(),
            "keypoints": result.keypoints.xyn.tolist(),
            "size": size,
        }

        with txt_path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)


class VideoProcessor:
    """비디오 프레임을 처리하고 객체 탐지 및 크기 추정을 수행하는 클래스."""

    def __init__(self, opt):
        """
        VideoProcessor를 초기화합니다.

        Args:
            opt (Any): 명령줄 옵션 객체.
        """
        self.opt = opt
        self.weights = opt.weights
        self.source = opt.source
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.device = opt.device

        self.use_webcam = opt.use_webcam
        self.verbose = opt.verbose
        self.visualize = opt.visualize
        self.save = opt.save
        self.save_txt = opt.save_txt
        self.project = opt.project
        self.name = opt.name
        self.exist_ok = opt.exist_ok

        self.save_path = IncrementPath(Path(self.project, self.name), self.exist_ok)

        self.model_loader = ModelLoader(self.weights, self.device)
        self.data_loader = DataLoader(self.source, self.device, self.use_webcam)
        self.model = self.model_loader.load_model()
        self.dataset = self.data_loader.load_data()

        self.data_predictor = DataPredictor(self.model, self.device, self.conf_thres, self.iou_thres, self.verbose)
        self.result_saver = ResultSaver(self.save_path, self.exist_ok)
        self.size_estimator = SizeEstimator()
        self.visualizer = Visualizer()

    def run(self):
        """
        로드된 데이터셋에 대해 순차적으로 객체 탐지, 크기 추정, 시각화 및 결과 저장을 수행합니다.
        각 미디어 유형(이미지, 비디오, 웹캠)에 맞게 처리하며, 필요 시 VideoWriter를 관리합니다.
        """
        for file_gen in self.dataset:
            current_mtype = None
            current_filename = None

            for data, input_tensor, mtype, fname in file_gen:
                if current_mtype is None and current_filename is None:
                    current_mtype = mtype
                    current_filename = fname

                result, input_tensor = self.data_predictor.predict_data(input_tensor)
                size = self.size_estimator.estimate_size(data, result[0])
                data, annot = self.visualizer.draw(data, result[0], size)

                if self.visualize:
                    self.visualizer.visualize(annot)

                if self.save:
                    self.result_saver.save(mtype, annot, fname)

                if self.save_txt:
                    self.result_saver.save_txt(result[0], size, fname)

            if self.visualize and current_mtype in ("video", "webcam"):
                self.result_saver.off_video_writer()

        if self.visualize:
            cv2.destroyAllWindows()
