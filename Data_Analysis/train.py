from ultralytics import YOLO


name = 'flatfish_size_label_augment'
optimizers = ["auto", "SGD", "Adam", 'AdamW', "NAdam", "RAdam", "RMSProp"]
for idx, opt in enumerate(optimizers):
    model = YOLO('./runs/pose/train/1_auto/train/weights/best.pt')
    results = model.train(
        data='Fish_Size/Data/Images/infrared/augment/data.yaml',
        epochs=1000,
        patience=250,
        batch=-1,
        save_period=50,
        device=0,
        project=f"runs/pose/train/{idx+1}_{opt}",
        name=name,
        optimizer=opt,
    )
