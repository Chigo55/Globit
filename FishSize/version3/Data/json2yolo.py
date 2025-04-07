import json
from pathlib import Path


def normalize_bbox(bbox, img_width, img_height):
    """
    바운딩 박스 좌표를 YOLO 형식에 맞춰 (x_center, y_center, w, h)로 변환.
    bbox: [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h


def normalize_keypoints(kpts, img_width, img_height):
    """
    keypoints: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 형태라 가정.
    각 (x, y)를 정규화하여 튜플로 변환.
    """
    kp_normalized = []
    for (x, y) in kpts:
        kp_normalized.append((x / img_width, y / img_height))
    return kp_normalized


def json2yolo(path):
    path = Path(path)
    output_dir = path.parent / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 이미지 정보
        img_width = data["images"][0]["width"]
        img_height = data["images"][0]["height"]

        yolo_lines = []
        annotations = data.get("annotations", [])

        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            keypoints = ann["keypoints"]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

            x_center, y_center, w, h = normalize_bbox(bbox, img_width, img_height)
            kp1, kp2, kp3, kp4 = normalize_keypoints(keypoints, img_width, img_height)

            # 최종 텍스트 라인 생성
            line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {kp1} {kp2} {kp3} {kp4}"
            yolo_lines.append(line)

        # 결과 저장
        output_path = output_dir / (json_file.stem + ".txt")
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(yolo_lines) + "\n")


if __name__ == "__main__":
    json2yolo("D:/생육거점 데이터/07. 넙치사이즈 측정 라벨링/json")
