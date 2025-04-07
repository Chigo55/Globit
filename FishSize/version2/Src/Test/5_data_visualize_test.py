# %%
import torch

from Src.FishSize import ModelLoader
from Src.FishSize import DataLoader
from Src.FishSize import DataPredictor
from Src.FishSize import SizeEstimator
from Src.FishSize import Visualizer

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
pose_model_loader = ModelLoader(
    weight='./Models/Custom/halibut_nano.pt',
    device=device
)

# %%
pose_model = pose_model_loader.load_model()

# %%
data_loader = DataLoader(
    source='./Data/Images/infrared/train/003500.jpg',
    device=device
)

# %%
data_iter = data_loader.load_data()
load_images_instance = next(data_iter)
raw_data, input_tensor, mtype, fname = next(iter(load_images_instance))

# %%
pose_predictor = DataPredictor(
    model=pose_model,
    device=device,
    conf_thres=0.25,
    iou_thres=0.45,
    verbose=False
)

# %%
pose_result, input_tensor = pose_predictor.predict_data(data=input_tensor)

# %%
size_estimator = SizeEstimator()
size = size_estimator.estimate_size(raw_data, pose_result[0])

# %%
visualizer = Visualizer()
raw_data, annot_data = visualizer.draw(raw_data, pose_result[0], size)

visualizer.visualize(annot_data)
