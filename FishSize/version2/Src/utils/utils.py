import os

from pathlib import Path


IMAGE_FORMATS = [
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
    "HEIC",
]
VIDEO_FORMATS = [
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
]


class IncrementPath:
    """
    경로에 숫자를 증가시켜 새로운 디렉토리를 생성하는 클래스.

    Args:
        path (str): 기본 경로.
        exist_ok (bool): 기본 경로가 존재해도 그대로 사용할지 여부.
        sep (str): 기본 경로와 증가 번호 사이의 구분자.
    """

    def __init__(self, path, exist_ok=False, sep="", mkdir=False):
        self.path = Path(path)
        self.exist_ok = exist_ok
        self.sep = sep
        self.mkdir = mkdir

    def __call__(self):
        return self.get()

    def get(self):
        """
        경로가 존재하거나 사용 가능한 경우, 해당 경로를 생성 및 반환합니다.
        존재하지 않는 경우에는 기본 경로와 숫자 구분자를 결합해 새로운 경로를 생성합니다.

        Returns:
            새로 생성된 경로 (str 또는 Path 객체).
        """
        if self.path.exists() and not self.exist_ok:
            self.path, self.suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

            for n in range(2, 9999):
                p = f"{path}{self.sep}{n}{self.suffix}"  # increment path
                if not os.path.exists(p):
                    break
            path = Path(p)

        if self.mkdir:
            self.path.mkdir(parents=True, exist_ok=True)

        return self.path

    def __str__(self):
        """
        객체를 문자열로 표현할 때 호출됩니다.

        Returns:
            경로를 문자열 형태로 반환합니다.
        """
        return str(self.get())
