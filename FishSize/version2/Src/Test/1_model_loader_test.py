# %%
from Src.FishSize import ModelLoader

# %%
pose_model_loader = ModelLoader(
    weight='./Models/Custom/halibut_nano.pt',
    device='cuda'
)
pose_model = pose_model_loader.load_model()

# %%
print(pose_model)
