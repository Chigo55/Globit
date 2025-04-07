# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Src.FishSize import ModelLoader
from Src.FishSize import DataLoader
from Src.FishSize import DataPredictor

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
plt.imshow(raw_data)
plt.axis('off')
plt.show()
