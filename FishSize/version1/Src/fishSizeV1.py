import cv2
import csv
import time
import mimetypes
import numpy as np

# Import YOLO for object detection / YOLO 객체 검출을 위한 임포트
from ultralytics import YOLO

# Import Depth Anything for depth estimation / Depth Anything 깊이 추정을 위한 임포트
import torch
import torch.nn.functional as F

from Lib.depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
from Lib.depth_anything.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)

# Import libraries for class and function definitions / 클래스 및 함수 정의용 라이브러리 임포트
from abc import ABCMeta, abstractmethod
from enum import Enum
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Tuple, Union, Optional

# Enable performance improvements for PyTorch / PyTorch 성능 향상 설정
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# Abstract base class for fish size estimation / 물고기 크기 추정을 위한 추상 기본 클래스
class FishSizeBase(metaclass=ABCMeta):

    def __init__(self):
        self.DEVICE = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        self.YOLOV8 = YOLO("./Model/Batch/all_classes_nano.pt")  # YOLO model path
        self.CORR = 70  # Correction factor for converting pixel to cm

        # Initialize fish species enumeration / 물고기 종에 대한 열거형 선언
        self.Fish = Enum(
            "Fish",
            [
                "Olive flounder",
                "Korea rockfish",
                "Red seabream",
                "Black porgy",
                "Rock bream",
            ],
            start=0,
        )

        # Load DepthAnything model / DepthAnything 모델 로드
        self.DEPTHANYTHING = (
            DepthAnything.from_pretrained(
                "LiheYoung/depth_anything_{}14".format("vits")
            )
            .to(self.DEVICE)
            .eval()
        )
        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    # Determine path type: file, directory, or stream / 경로 타입 판단: 파일, 디렉토리 또는 스트림
    def checkInputPath(self, path: str) -> List[Tuple[str, Union[str, Path]]]:
        path_obj = Path(path)

        # Result storage /결과 저장
        result = []

        # Check if it's a file / 파일일 경우
        if path_obj.is_file():
            mime_type, _ = mimetypes.guess_type(str(path_obj))
            result.append((str(mime_type), str(path_obj)))

        # Check if it's a directory / 디렉토리일 경우
        elif path_obj.is_dir():
            result = [
                (mimetypes.guess_type(str(file))[0], str(file))
                for file in path_obj.iterdir()
                if file.is_file()
            ]

        # Assume video stream if path is 0 or 'stream' / 비디오 스트림 경로일 경우
        else:
            if path == "0" or path == "stream":
                result.append(("video/mp4", "stream"))
            else:
                raise FileNotFoundError(f"No such file or directory: '{path}'")

        return result

    # Create a unique directory in the specified path / 지정된 경로에 고유한 디렉토리 생성
    def createUniqueDirectory(sel, project_path: str, name_path: str) -> Path:
        project_path = Path(project_path)
        name_path = Path(name_path)

        # Initialize the base paths / 기본 경로 초기화
        base_project_path = project_path
        project_counter = 2

        while True:
            # Construct the full path including name_path / name_path를 포함한 전체 경로 구성
            full_path = project_path / name_path

            if not full_path.exists():
                # Create the unique directory if it does not exist / 고유 디렉토리 생성
                full_path.mkdir(parents=True, exist_ok=True)
                return full_path

            # If the full_path exists, update project_path to make it unique / full_path가 이미 존재하는 경우 project_path를 고유하게 업데이트
            project_path = (
                base_project_path.parent / f"{base_project_path.name}{project_counter}"
            )
            project_counter += 1

    # Check the file type and determine processing parameters / 파일 유형을 확인하고 처리 매개변수 결정
    def checkFileType(
        self, file_path: Union[Path, str], file_type: str, output_path: Union[Path, str]
    ) -> Optional[Tuple[str, str, Path, bool]]:
        file_path = Path(file_path)

        # Get the file name and extension / 파일의 이름과 확장자 추출
        basename = file_path.stem
        extension = file_path.suffix

        if file_type is not None and file_type.split("/")[0] == "image":
            # If input is an image, set flag to process images / 입력이 이미지일 경우, 이미지 처리 플래그 설정
            csv_path = Path(output_path) / "output.csv"
            flag = True

        elif file_type is not None and (
            file_type.split("/")[0] == "video" or file_type.split("/")[0] == "stream"
        ):
            # If input is a video or stream, set flag to process video / 입력이 비디오 또는 스트림일 경우, 비디오 처리 플래그 설정
            csv_path = Path(output_path) / "output.csv"
            flag = False
        else:
            return None  # If none of the conditions match, exit the function

        return basename, extension, csv_path, flag

    # Check if any keypoint has coordinates (0, 0) / 키포인트 중 하나라도 좌표가 (0, 0)인지 확인
    def checkZeroKeypoints(self, keypoint: List[int]) -> bool:
        if keypoint == [0, 0]:
            return False
        return True

    # Draw bounding box on the image / 이미지에 바운딩 박스 그리기
    def drawBbox(
        self,
        image: np.ndarray,
        class_name: str,
        conf: float,
        bbox: Tuple[float, float, float, float],
        width: Union[float, int],
        height: Union[float, int],
    ) -> None:
        x1, y1, x2, y2 = list(map(int, bbox))

        # Draw rectangle and label text / 사각형과 라벨 텍스트 그리기
        if width == None and height == None:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{class_name}:{conf:.2f}% Size estimation not possible",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 0),
                2,
            )
        elif width == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{class_name}:{conf:.2f}% | H:{height:.2f}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 0),
                2,
            )
        elif height == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{class_name}:{conf:.2f}% | W:{width:.2f}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 0),
                2,
            )
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{class_name}:{conf:.2f}% | W:{width:.2f} H:{height:.2f}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 0),
                2,
            )

    # Draw keypoints on the image / 이미지에 키포인트 그리기
    def drawPoint(
        self, image: np.ndarray, keypoints: List[Tuple[float, float]]
    ) -> None:
        for kp in keypoints:
            kp = list(map(int, kp))

            # Only Draw if points are valid (not (0, 0))/ 키포인트가 유효한 경우에만 그리기 (0, 0 아님)
            if self.checkZeroKeypoints(kp):
                cv2.line(image, (kp), (kp), (0, 0, 255), 15)

    # Draw lines between keypoints / 키포인트 사이에 선 그리기
    def drawLine(self, image: np.ndarray, keypoints: List[Tuple[float, float]]) -> None:
        kp1, kp2, kp3, kp4 = keypoints

        # Only Draw if points are valid (not (0, 0))/ 키포인트가 유효한 경우에만 그리기 (0, 0 아님)
        if self.checkZeroKeypoints(kp1) and self.checkZeroKeypoints(kp2):
            # Draw lines between specific keypoints / 특정 키포인트 사이에 선 그리기
            cv2.line(
                image, (list(map(int, kp1))), (list(map(int, kp2))), (0, 255, 0), 5
            )

        # Only Draw if points are valid (not (0, 0))/ 키포인트가 유효한 경우에만 그리기 (0, 0 아님)
        if self.checkZeroKeypoints(kp3) and self.checkZeroKeypoints(kp4):
            # Draw lines between specific keypoints / 특정 키포인트 사이에 선 그리기
            cv2.line(
                image, (list(map(int, kp3))), (list(map(int, kp4))), (0, 255, 0), 5
            )

    # Convert pixel-based keypoints to real-world size in cm / 픽셀 단위의 키포인트를 실제 세계 크기(cm)로 변환
    def ConvertCM(
        self,
        keypoints: List[Tuple[float, float]],
        depth: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> Tuple[float, float]:
        kp1, kp2, kp3, kp4 = keypoints
        kpx, kpy = list(zip(kp1, kp2, kp3, kp4))

        # Extract depth for keypoints / 키포인트에 대한 깊이 정보 추출
        kpz = list(
            map(
                lambda x, y: depth[y - 1, x - 1],
                (list(map(int, kpx))),
                (list(map(int, kpy))),
            )
        )

        # Normalize keypoints to image dimensions / 키포인트를 이미지 크기로 정규화
        kpx, kpy = list(map(lambda x: x / image_width, kpx)), list(
            map(lambda x: x / image_height, kpy)
        )

        # Calculate width and height in cm using correction factor / 보정 값을 사용해 cm 단위로 폭과 높이 계산
        width = (
            ((kpx[0] - kpx[1]) ** 2 + (kpy[0] - kpy[1]) ** 2 + (kpz[0] - kpz[1]) ** 2)
            ** 0.5
        ) * self.CORR
        height = (
            ((kpx[2] - kpx[3]) ** 2 + (kpy[2] - kpy[3]) ** 2 + (kpz[2] - kpz[3]) ** 2)
            ** 0.5
        ) * self.CORR

        # Only width and height output if points are valid (not (0, 0))/ 키포인트가 유효한 경우에만 가로 세로 길이 출력 (0, 0 아님)
        if not self.checkZeroKeypoints(kp1) or not self.checkZeroKeypoints(kp2):
            width = 0.0
        if not self.checkZeroKeypoints(kp3) or not self.checkZeroKeypoints(kp4):
            height = 0.0
        return width, height

    # Write result data to CSV file / 결과 데이터를 CSV 파일로 작성
    def writeToCSV(
        self,
        csv_path: Union[str, Path],
        class_number: int,
        width: float,
        height: float,
        date: str,
        time: str,
    ) -> None:
        data = {
            "Class Number": class_number,
            "Width": width,
            "Height": height,
            "Date": date,
            "Time": time,
        }

        # Convert to Path object / Path 객체로 변환
        csv_path = Path(csv_path)

        # Check if file exists / 파일이 존재하는지 확인
        file_exists = csv_path.exists()

        with csv_path.open(mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not file_exists:
                # Write header if file does not exist / 파일이 존재하지 않으면 헤더 작성
                writer.writeheader()

            # Write the data row / 데이터 행 작성
            writer.writerow(data)

    # Abstract method to be implemented for prediction / 예측을 위해 구현되어야 할 추상 메서드
    @abstractmethod
    def predict(self):
        pass


# Fish size estimation class / 물고기 크기 추정 클래스
class FishSize(FishSizeBase):

    def __init__(
        self,
        input_path: str,
        project: str = "predict",
        name: str = "exp",
        save: bool = True,
        save_txt: bool = True,
        draw_bbox: bool = True,
        draw_keypoint: bool = True,
        draw_line: bool = True,
    ):
        super().__init__()
        self.input_path = input_path
        self.project = Path("./runs/" + project)
        self.name = name
        self.output_path = None

        # Options for saving and drawing / 저장 및 그리기 옵션
        self.save = save
        self.save_txt = save_txt
        self.draw_bbox = draw_bbox
        self.draw_keypoint = draw_keypoint
        self.draw_line = draw_line

        # File information and processing flags / 파일 정보 및 처리 플래그
        self.basename = None
        self.extension = None
        self.csv_path = None
        self.flag = None

    def predict(self):
        # Determine input path type (image, video, or stream) / 입력 경로 유형 판단 (이미지, 비디오, 스트림)
        path = self.checkInputPath(self.input_path)

        # Create output directory if not exists / 출력 경로가 없으면 생성
        self.output_path = self.createUniqueDirectory(self.project, self.name)

        for type, file in tqdm(path):
            self.basename, self.extension, self.csv_path, self.flag = (
                self.checkFileType(file, type, self.output_path)
            )

            # Process image files / 이미지 파일 처리
            if self.flag:
                img = cv2.imread(f"{file}")
                img_height, img_width, _ = img.shape

                # Depth estimation using Depth Anything / Depth Anything를 사용한 깊이 추정
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                rgb_img = self.transform({"image": rgb_img})["image"]
                rgb_img = torch.from_numpy(rgb_img).unsqueeze(0).to(self.DEVICE)

                with torch.inference_mode():
                    depth_img = self.DEPTHANYTHING(rgb_img)

                depth_img = F.interpolate(
                    depth_img[None],
                    (img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                depth_img = (depth_img - depth_img.min()) / (
                    depth_img.max() - depth_img.min()
                )
                depth_img = depth_img.cpu().numpy()

                # YOLOv8 for object detection / YOLOv8를 사용한 객체 검출
                results = self.YOLOV8.predict(
                    source=img,
                    save=False,
                    stream=True,
                    conf=0.75,
                    iou=0.75,
                    save_txt=False,
                    verbose=False,
                )

                # Annotate results / 결과 표시
                for result in results:
                    for r in result:
                        keypoints = r.keypoints.xy.squeeze().tolist()
                        bboxes = r.boxes.xyxy.squeeze().tolist()
                        conf = r.boxes.conf.squeeze().tolist() * 100
                        class_number = int(r.boxes.cls)

                        # Draw keypoints / 키포인트 그리기
                        if self.draw_keypoint:
                            self.drawPoint(img, keypoints)

                        # Draw lines between keypoints and calculate dimensions / 키포인트 사이에 선을 그리고 크기 계산
                        if self.draw_line:
                            if len(keypoints) > 3:
                                width, height = self.ConvertCM(
                                    keypoints, depth_img, img_height, img_width
                                )
                                self.drawLine(img, keypoints)

                        # Draw bounding boxes / 바운딩 박스 그리기
                        if self.draw_bbox:
                            class_name = r.names[class_number]
                            self.drawBbox(img, class_name, conf, bboxes, width, height)

                        # Write data to CSV / CSV 파일에 데이터 작성
                        if self.save_txt:
                            cur_date, cur_time = (
                                f"{time.strftime('%x', time.localtime(time.time()))}",
                                f"{time.strftime('%X', time.localtime(time.time()))}",
                            )
                            self.writeToCSV(
                                self.csv_path,
                                class_number,
                                width,
                                height,
                                cur_date,
                                cur_time,
                            )

                    # Save annotated image / 주석이 달린 이미지 저장
                    if self.save:
                        cv2.imwrite(
                            f"{self.output_path}/{self.basename}_pred.{self.extension}",
                            img,
                        )

            # Process video files / 비디오 파일 처리
            else:
                cap = cv2.VideoCapture(f"{file}")
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define video output writer / 비디오 출력 형식 지정
                out = cv2.VideoWriter(
                    f"{self.output_path}/{self.basename}_pred.{self.extension}",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    int(cap.get(cv2.CAP_PROP_FPS)),
                    (frame_width, frame_height),
                )

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Depth estimation for each frame / 각 프레임에 대한 깊이 추정
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                    rgb_frame = self.transform({"image": rgb_frame})["image"]
                    rgb_frame = torch.from_numpy(rgb_frame).unsqueeze(0).to(self.DEVICE)

                    with torch.inference_mode():
                        depth_frame = self.DEPTHANYTHING(rgb_frame)

                    depth_frame = F.interpolate(
                        depth_frame[None],
                        (frame_height, frame_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    depth_frame = (depth_frame - depth_frame.min()) / (
                        depth_frame.max() - depth_frame.min()
                    )
                    depth_frame = depth_frame.cpu().numpy()

                    # YOLOv8 for object detection / YOLOv8를 사용한 객체 검출
                    results = self.YOLOV8.predict(
                        source=frame,
                        save=False,
                        stream=True,
                        conf=0.75,
                        iou=0.75,
                        save_txt=False,
                    )

                    for result in results:
                        for r in result:
                            keypoints = r.keypoints.xy.squeeze().tolist()
                            bboxes = r.boxes.xyxy.squeeze().tolist()
                            conf = r.boxes.conf.squeeze().tolist() * 100
                            class_number = int(r.boxes.cls)

                            # Only draw if keypoints are valid (not (0, 0)) / 키포인트가 유효한 경우에만 그리기 (0, 0 아님)
                            if self.draw_keypoint:
                                # Draw keypoints / 키포인트 그리기
                                self.drawPoint(frame, keypoints)

                            if self.draw_line:
                                # Draw lines between keypoints and calculate dimensions / 키포인트 사이에 선을 그리고 크기 계산
                                if len(keypoints) > 3:
                                    width, height = self.ConvertCM(
                                        keypoints,
                                        depth_frame,
                                        frame_width,
                                        frame_height,
                                    )
                                    self.drawLine(frame, keypoints)

                            if self.draw_bbox:
                                # Draw bounding boxes / 바운딩 박스 그리기
                                class_name = r.names[class_number]
                                self.drawBbox(
                                    frame, class_name, conf, bboxes, width, height
                                )

                            if self.save_txt:
                                # Write data to CSV / CSV 파일에 데이터 작성
                                cur_date, cur_time = (
                                    f"{time.strftime('%x', time.localtime(time.time()))}",
                                    f"{time.strftime('%X', time.localtime(time.time()))}",
                                )
                                self.writeToCSV(
                                    self.csv_path,
                                    class_number,
                                    width,
                                    height,
                                    cur_date,
                                    cur_time,
                                )

                        if self.save:
                            # Write the processed frame to video output / 처리된 프레임을 비디오 출력에 작성
                            out.write(frame)
