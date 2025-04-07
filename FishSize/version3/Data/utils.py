from dataclasses import dataclass

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes


@dataclass
class SourceTypes:
    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False
