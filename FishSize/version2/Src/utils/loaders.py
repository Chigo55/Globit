import cv2

from pathlib import Path
from torchvision import transforms


class LoadImages:
    """
    단일 이미지 파일을 로드하고, 전처리된 텐서를 반환하는 클래스.

    Args:
        path (str or Path): 이미지 파일의 경로.
        device (str): 사용할 디바이스 (예: "cpu", "cuda").
    """

    def __init__(self, path, device):
        self.path = Path(path)
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def __iter__(self):
        """
        이미지 파일을 읽어와 전처리 후, 배치 형태의 텐서로 변환합니다.

        Yields:
            tuple: (원본 이미지, 전처리된 텐서, "image" 문자열, 파일명(확장자 제외)).

        Raises:
            FileNotFoundError: 이미지 파일을 찾을 수 없을 경우 발생합니다.
        """
        image = cv2.imread(str(self.path))
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {self.path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)
        yield (image, tensor.unsqueeze(dim=0).to(self.device), "image", self.path.stem)


class LoadVideos:
    """
    단일 비디오 파일에서 프레임을 순차적으로 로드하고, 전처리된 텐서를 반환하는 클래스.

    Args:
        path (str or Path): 비디오 파일의 경로.
        device (str): 사용할 디바이스 (예: "cpu", "cuda").
    """

    def __init__(self, path, device):
        self.path = Path(path)
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def __iter__(self):
        """
        비디오 파일의 각 프레임을 읽어와 전처리한 후, 배치 형태의 텐서로 변환합니다.

        Yields:
            tuple: (원본 프레임, 전처리된 텐서, "video" 문자열, 파일명(확장자 제외)).

        Raises:
            ValueError: 비디오 파일을 열 수 없을 경우 발생합니다.
        """
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {self.path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self.transform(frame)
                yield (frame, tensor.unsqueeze(dim=0).to(self.device), "video", self.path.stem)
        finally:
            cap.release()


class LoadWebcam:
    """
    웹캠에서 실시간으로 프레임을 읽어와 전처리된 텐서를 반환하는 클래스.

    Args:
        pipe (int or str): 웹캠 인덱스 또는 파이프 스트림.
        device (str): 사용할 디바이스 (예: "cpu", "cuda").
    """

    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def __iter__(self):
        """
        웹캠에서 프레임을 실시간으로 읽어와 전처리 후, 배치 형태의 텐서로 변환합니다.

        Yields:
            tuple: (원본 프레임, 전처리된 텐서, "webcam" 문자열, 웹캠 식별자).

        Raises:
            ValueError: 웹캠을 열 수 없을 경우 발생합니다.
        """
        cap = cv2.VideoCapture(self.pipe)
        if not cap.isOpened():
            raise ValueError(f"웹캠을 열 수 없습니다: {self.pipe}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self.transform(frame)
                yield (frame, tensor.unsqueeze(dim=0).to(self.device), "webcam", str(self.pipe))
        finally:
            cap.release()
