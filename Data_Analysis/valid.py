from ultralytics import YOLO


name = 'augment_model_no_tsp'
optimizers = ["auto", "SGD", "Adam", 'AdamW', "NAdam", "RAdam", "RMSProp"]
for idx, opt in enumerate(optimizers):
    model = YOLO(f'./runs/pose/train/{idx+1}_{opt}/{name}/weights/best.pt')
    metrics = model.val(
        data='Fish_Size/Data/Images/infrared/globit_nas_07_flatfish_size_label/data.yaml',
        project=f"runs/pose/valid/{idx+1}_{opt}",
        name=f"{name}",
    )

    f1 = metrics.pose.f1
    p = metrics.pose.p
    r = metrics.pose.r
    mAP = metrics.pose.map
    print(f'{sum(f1)/len(f1):.4f}, {sum(p)/len(p):.4f}, {sum(r)/len(r):.4f}, {mAP:.4f}, sum([sum(f1)/len(f1), sum(p)/len(p), sum(r)/len(r), mAP])')
