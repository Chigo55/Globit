import cv2
import os
from pathlib import Path


def draw_yolo_pose(image_path, label_path, save_path=None):
    image_path = str(image_path)

    label_path = str(label_path)
    img = cv2.imread(image_path)

    if img is None:

        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    img_h, img_w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        class_id = int(parts[0])

        x_center, y_center, width, height = map(float, parts[1:5])
        keypoints = list(map(float, parts[5:]))

        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        print(f"YOLO BBox → 이미지 복원: ({x1}, {y1}) → ({x2}, {y2})")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(img, f'ID: {class_id}', (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

        for i in range(0, len(keypoints), 3):
            x = int(keypoints[i] * img_w)

            y = int(keypoints[i + 1] * img_h)
            v = int(keypoints[i + 2])

            if v > 0:
                color = (0, 0, 255) if v == 1 else (255, 0, 0)
                cv2.circle(img, (x, y), 5, color, -1)

    image_path = Path(image_path)

    # cv2.imshow("YOLO Pose Visualization", img)
    cv2.imwrite(f'./{image_path.name}', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / image_path.name), img)


image_dir = "./Fish_Size/Data/Images/infrared/augment2/images"
label_dir = "./Fish_Size/Data/Images/infrared/augment2/labels"

image_dir, label_dir = Path(image_dir), Path(label_dir)

image_paths = [path for path in image_dir.glob('*.jpg')]
label_paths = [path for path in label_dir.glob('*.txt')]

n = 0
for image_path, label_path in zip(image_paths, label_paths):
    if image_path.stem == label_path.stem:
        draw_yolo_pose(image_path, label_path)
    else:
        n += 1

print(n)
