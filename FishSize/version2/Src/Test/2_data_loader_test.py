# %%
import matplotlib. pyplot as plt
from Src.FishSize import DataLoader

# %%
data_loader = DataLoader('./Data/Images/infrared/train/003500.jpg', 'cuda')
data_iter = data_loader.load_data()

# %%
load_images_instance = next(data_iter)
image_data_tuple = next(iter(load_images_instance))
original_image = image_data_tuple[0]

# %%
plt.imshow(original_image)
plt.axis('off')
plt.show()
