import cv2
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm


class ImageProcessor:
    def vertical(self, image):
        flipped_image = cv2.flip(image, 0)
        return flipped_image

    def horizontal(self, image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image

    def gaussian_noise(self, image, noise_percent=1):
        sigma = 255 * (noise_percent / 100)
        gaussian = np.random.normal(0, sigma, image.shape).astype(np.float32)
        gaussian_image = cv2.add(image.astype(np.float32), gaussian)
        gaussian_image = np.clip(gaussian_image, 0, 255).astype(np.uint8)
        return gaussian_image

    def salt_pepper_noise(self, image, noise_percent=1):
        salt_pepper_image = np.copy(image)
        noise_percent = noise_percent / 100
        salt_pepper_image = self._salt_pepper(image=image, noise_percent=noise_percent)
        return salt_pepper_image

    def adjust_contrast(self, image, adjust_percent=10):
        adjust_percent = (100+adjust_percent)/100
        adjusted_image = cv2.convertScaleAbs(image, alpha=adjust_percent)
        return adjusted_image

    def adjust_bright(self, image, adjust_percent=10):
        adjust_percent = int(255*(adjust_percent/100))
        adjusted_image = cv2.convertScaleAbs(image, beta=adjust_percent)
        return adjusted_image

    def _salt_pepper(self, image, noise_percent):
        num_pixels = image.shape[0] * image.shape[1]
        point = np.ceil(noise_percent * num_pixels * 0.5)
        salt = self._apply_noise(image=image, point=point, value=255)
        salt_pepper = self._apply_noise(image=salt, point=point, value=0)
        return salt_pepper

    def _apply_noise(self, image, point, value):
        coords = [np.random.randint(0, i, int(point)) for i in image.shape[:2]]
        image[coords[0], coords[1], :] = value
        return image


class LabelProcessor:
    def vertical(self, labels):
        new_labels = self._apply_vertical_flip(labels=labels)
        return new_labels

    def horizontal(self, labels):
        new_labels = self._apply_horizontal_flip(labels=labels)
        return new_labels

    def _apply_vertical_flip(self, labels):
        new_labels = []
        for label in labels:
            cls, bbox, keypoints = self._parse_label(label=label)
            new_bbox = self._vertical_flip_bbox(bbox=bbox)
            new_keypoints = self._vertical_flip_keypoints(keypoints=keypoints)
            new_label = cls + new_bbox + new_keypoints
            new_labels.append(new_label)
        return new_labels

    def _apply_horizontal_flip(self, labels):
        new_labels = []
        for label in labels:
            cls, bbox, keypoints = self._parse_label(label=label)
            new_bbox = self._horizontal_flip_bbox(bbox=bbox)
            new_keypoints = self._horizontal_flip_keypoints(keypoints=keypoints)
            new_label = cls + new_bbox + new_keypoints
            new_labels.append(new_label)
        return new_labels

    def _parse_label(self, label):
        cls = [int(label[0])]
        bbox = list(label[1:5])
        keypoints = list(label[5:])
        return cls, bbox, keypoints

    def _vertical_flip_bbox(self, bbox):
        bbox[1] = 1 - bbox[1]
        return bbox

    def _vertical_flip_keypoints(self, keypoints):
        for i in range(0, len(keypoints), 3):
            keypoints[i] = 1 - keypoints[i]
        return keypoints

    def _horizontal_flip_bbox(self, bbox):
        bbox[0] = 1 - bbox[0]
        return bbox

    def _horizontal_flip_keypoints(self, keypoints):
        for i in range(1, len(keypoints), 3):
            keypoints[i] = 1 - keypoints[i]
        return keypoints


class DataAugment:
    def __init__(self, image_dir, label_dir, output_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_paths = [path for path in self.image_dir.glob('*.jpg')]
        self.label_paths = [path for path in self.label_dir.glob('*.txt')]

        self.output_dir = Path(output_dir)
        self.output_image_path = self.output_dir / 'images'
        self.output_label_path = self.output_dir / 'labels'
        self.output_image_path.mkdir(parents=True, exist_ok=True)
        self.output_label_path.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor()
        self.label_processor = LabelProcessor()

    def load_data(self):
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            self.image_stem = image_path.stem
            self.label_stem = label_path.stem
            if self.image_stem == self.label_stem:
                self.image_suffix = image_path.suffix
                self.label_suffix = label_path.suffix

                self.current_image = cv2.imread(str(image_path))
                with open(label_path, 'r') as f:
                    self.current_labels = [list(map(float, line.strip().split())) for line in f.readlines()]

                self.filename_suffix = ""
                yield self

    def vertical(self):
        self.current_image = self.image_processor.vertical(image=self.current_image)
        self.current_labels = self.label_processor.vertical(labels=self.current_labels)
        self.filename_suffix += f"_vert"
        return self

    def horizontal(self):
        self.current_image = self.image_processor.horizontal(image=self.current_image)
        self.current_labels = self.label_processor.horizontal(labels=self.current_labels)
        self.filename_suffix += f"_horiz"
        return self

    def gaussian_noise(self, noise_percent=1):
        self.current_image = self.image_processor.gaussian_noise(image=self.current_image, noise_percent=noise_percent)
        self.filename_suffix += f"_gauss{noise_percent}"
        return self

    def salt_pepper_noise(self, noise_percent=1):
        self.current_image = self.image_processor.salt_pepper_noise(image=self.current_image, noise_percent=noise_percent)
        self.filename_suffix += f"_salt{noise_percent}"
        return self

    def adjust_contrast(self, adjust_percent=10):
        self.current_image = self.image_processor.adjust_contrast(image=self.current_image, adjust_percent=adjust_percent)
        self.filename_suffix += f"_contrast{adjust_percent}"
        return self

    def adjust_bright(self, adjust_percent=10):
        self.current_image = self.image_processor.adjust_bright(image=self.current_image, adjust_percent=adjust_percent)
        self.filename_suffix += f"_bright{adjust_percent}"
        return self

    def save(self):
        image_name = self.image_stem + self.filename_suffix + self.image_suffix
        label_name = self.label_stem + self.filename_suffix + self.label_suffix

        output_image = self.output_image_path / image_name
        output_label = self.output_label_path / label_name

        cv2.imwrite(str(output_image), self.current_image)
        with open(output_label, "w") as f:
            for label in self.current_labels:
                label_str = " ".join(str(x) for x in label)
                f.write(label_str + "\n")

        self.filename_suffix = ""
        return self


if __name__ == '__main__':
    image_path = "Fish_Size/Data/Images/infrared/globit_nas_07_flatfish_size_label2/train/images/"
    label_path = "Fish_Size/Data/Images/infrared/globit_nas_07_flatfish_size_label2/train/labels/"
    output_dir = "Fish_Size/Data/Images/infrared/augment2"

    dataset = DataAugment(image_path, label_path, output_dir).load_data()
    for data in tqdm(dataset):
        data.vertical().save()
        data.vertical().gaussian_noise().save()
        data.vertical().salt_pepper_noise().save()
        data.vertical().gaussian_noise().salt_pepper_noise().save()
        data.vertical().adjust_contrast().save()
        data.vertical().adjust_bright().save()
        data.vertical().adjust_contrast().adjust_bright().save()
        data.vertical().adjust_contrast(-10).save()
        data.vertical().adjust_bright(-10).save()
        data.vertical().adjust_contrast(-10).adjust_bright(-10).save()

        data.horizontal().save()
        data.horizontal().gaussian_noise().save()
        data.horizontal().salt_pepper_noise().save()
        data.horizontal().gaussian_noise().salt_pepper_noise().save()
        data.horizontal().adjust_contrast().save()
        data.horizontal().adjust_bright().save()
        data.horizontal().adjust_contrast().adjust_bright().save()
        data.horizontal().adjust_contrast(-10).save()
        data.horizontal().adjust_bright(-10).save()
        data.horizontal().adjust_contrast(-10).adjust_bright(-10).save()

        data.vertical().horizontal().save()
        data.vertical().horizontal().gaussian_noise().save()
        data.vertical().horizontal().salt_pepper_noise().save()
        data.vertical().horizontal().gaussian_noise().salt_pepper_noise().save()
        data.vertical().horizontal().adjust_contrast().save()
        data.vertical().horizontal().adjust_bright().save()
        data.vertical().horizontal().adjust_contrast().adjust_bright().save()
        data.vertical().horizontal().adjust_contrast(-10).save()
        data.vertical().horizontal().adjust_bright(-10).save()
        data.vertical().horizontal().adjust_contrast(-10).adjust_bright(-10).save()

        break
