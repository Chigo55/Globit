import torch

from pathlib import Path

from Engine.predictor import Predictor


class Model:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = self._load(path)

    def _load(self, weight):
        return torch.load(Path(weight), map_location="cpu")

    def __call__(self, source, ):
        return self.predict(source=source)

    def predict(self, source, ):
        self.predictor = self._load_predictor()(self.model)
        return self.predictor(source=source)

    def _load_predictor(self):
        return Predictor
