import contextlib
import os
import torch

from pathlib import Path
from tqdm import tqdm as tqdm_original


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def is_online() -> bool:
    with contextlib.suppress(Exception):
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    return False


def smart_inference_mode():
    def decorate(fn):
        if torch.is_inference_mode_enabled():
            return torch.inference_mode()(fn)
        else:
            return torch.no_grad()(fn)

    return decorate


class TQDM(tqdm_original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
