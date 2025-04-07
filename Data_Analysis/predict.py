from tqdm.auto import tqdm
from ultralytics import YOLO


# for i in range(1, 8):
#     path = f"./runs/pose/train{i}/weights/best.pt"
#     model = YOLO(path)
#     results = model.predict(
#         './Fish_Size/Data/Videos/infrared/feed_summery.mp4',
#         stream = True,
#         save = True,
#         save_txt = True,
#         verbose = False,
#         device = 0
#         )

# for result in tqdm(results):
#     pass

# path = f"./runs/pose/train/train8/weights/best.pt"
# model = YOLO(path)
# results = model.predict(
#     './Fish_Size/Data/Videos/infrared/abnormal_rotate.mp4',
#     stream = True,
#     save = True,
#     save_txt = True,
#     verbose = False,
#     project="runs/pose/predict",

#     device = 0
#     )

# for result in tqdm(results):
#     pass

# conf = [0.10, 0.15, 0.20, 0.25]
# iou = [0.2, 0.4, 0.6, 0.8]

# for i in iou:
#     path = f"./runs/pose/train/train8/weights/best.pt"
#     model = YOLO(path)
#     results = model.predict(
#         './Fish_Size/Data/Videos/infrared/abnormal_rotate.mp4',
#         stream = True,
#         save = True,
#         save_txt = True,
#         verbose = False,
#         project="runs/pose/predict",
#         name=f"iou_change{i}",
#         iou=i,
#         device = 0
#         )

#     for result in tqdm(results):
#         pass

# for i in conf:
#     path = f"./runs/pose/train/train8/weights/best.pt"
#     model = YOLO(path)
#     results = model.predict(
#         './Fish_Size/Data/Videos/infrared/abnormal_rotate.mp4',
#         stream = True,
#         save = True,
#         save_txt = True,
#         verbose = False,
#         project="runs/pose/predict",
#         name=f"conf_change{i}",
#         conf=i,
#         device = 0
#         )

#     for result in tqdm(results):
#         pass

name = 'augment_model_no_tsp'
optimizers = ["auto", "SGD", "Adam", 'AdamW', "NAdam", "RAdam", "RMSProp"]
for idx, opt in enumerate(optimizers):
    path = f"./runs/pose/train/{idx+1}_{opt}/{name}/weights/best.pt"
    model = YOLO(path)
    results = model.predict(
        './Fish_Size/Data/Videos/infrared/abnormal_rotate.mp4',
        stream=True,
        save=True,
        save_txt=True,
        verbose=False,
        project=f"runs/pose/predict/{idx+1}_{opt}",
        name=f"{name}",
        conf=0.1,
        device=0
    )

    for result in tqdm(results):
        pass
