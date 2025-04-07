import torch
from torchvision import transforms

from Data.build import LoadSource
from Engine.results import Results
from Utils import smart_inference_mode
from Utils.ops import non_max_suppression, scale_boxes, scale_coords


class Predictor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def __call__(self, source, model):
        return self.stream_inference(source, model)

    @smart_inference_mode()
    def stream_inference(self, img0):

        self.setup_model()
        self.setup_source()

        img, img0 = self.preprocess(img0=img0)
        preds = self.inference(img=img)
        results = self.postprocess(preds=preds, img=img, img0=img0)
        yield from results

    def setup_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def setup_source(self, source):
        self.dataset = LoadSource(source=source)

    def preprocess(self, img0):
        not_tensor = not isinstance(img0, torch.Tensor)
        if not_tensor:
            img = self.pre_transform(img0)
        img = img.unsqueeze(dim=0)
        img = img.to(self.device)

        return img, img0

    def pre_transform(self, im):
        return self.transform(im)

    def inference(self, img):
        self.model = self.model.eval()
        return self.model(img)

    def postprocess(self, preds, img, img0):
        preds = non_max_suppression()
        results = []
        for pred, orig_img, img_path in zip(preds, img0, self.batch[0]):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results
