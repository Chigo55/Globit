import argparse
import cv2.ximgproc
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch.nn.functional as F
import os
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
import random
from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

def DEPTH_ANYTHING(img):
   
    parser = argparse.ArgumentParser()    

    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
    h, w = image.shape[:2]
        
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
    with torch.no_grad():
        depth = depth_anything(image)
        
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
    depth = depth.cpu().numpy().astype(np.uint8)
        
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
    return depth

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

ap = argparse.ArgumentParser()
In_path = r'C:\Users\User\Desktop\Backup\Code\Depth-Anything-main\Depth-Anything-main\assets\fish_examples/'

PATH = r'C:\Users\User\Desktop\ICICIC2024/'

DIR = os.listdir(In_path)

ind = 19
Ind = DIR[ind]

args = vars(ap.parse_args())

image = cv2.imread(In_path + Ind)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

[col, row, ch] = image.shape
image = cv2.resize(image, (640, 480))
image_depth = DEPTH_ANYTHING(image)
image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2RGB)

cv2.imwrite(PATH + str(ind) + '_img.png', image)
cv2.imwrite(PATH + str(ind) + '_img_depth.png', image_depth)

results = model.predict(image, conf=0.25, fuse_model=False)
output = results
bboxes = output.prediction.bboxes_xyxy
confs = output.prediction.confidence
labels = output.prediction.labels
class_names = output.class_names

random.seed(0)
labels = [int(l) for l in list(labels)]
label_colors = [tuple(random.choices(np.arange(0, 256), k=3)) for i in range(len(class_names))]
names = [class_names[int(label)] for label in labels]

IMG = np.zeros(image.shape, dtype='uint8')
IMG3 = image.copy()
IMG4 = np.zeros(image.shape, np.uint8)
IMG5 = IMG.copy()

for idx, bbox in enumerate(bboxes):
        bbox_left = int(bbox[0])
        bbox_top = int(bbox[1])
        bbox_right = int(bbox[2])
        bbox_bot = int(bbox[3])
              
        colors = tuple(int(i) for i in label_colors[labels[idx]])

        frame = image

        cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bot), color=colors, thickness=2)

        pts2 = np.array([[bbox_left,bbox_top],[bbox_left,bbox_bot],[bbox_right,bbox_bot],[bbox_right,bbox_top]])

        COL = np.abs(bbox_top - bbox_bot)        
        ROW = np.abs(bbox_right - bbox_left)

        cv2.fillPoly(IMG5,[pts2],color=(255,255,255))

IMG2 = (255 - IMG3) * (IMG5)

Img_depth = ((255 - image_depth.copy())) * IMG5
Img_depth = (Img_depth * 255).astype(np.uint8)

cv2.imwrite(PATH + str(ind) + '_IMG2.png', IMG2)
cv2.imwrite(PATH + str(ind) + '_img_depth.png', Img_depth)
cv2.imwrite(PATH + str(ind) + '_Image_depth.png', image_depth)

gray = IMG2

R = gray[:,:,2]
G = gray[:,:,1]
B = gray[:,:,0]

RG = R-G
RG = (RG - RG.min()) / (RG.max() - RG.min())
RG = np.clip(RG, 0, 1)
RG = (RG * 255).astype(np.uint8)

GB = G-B
GB = (GB - GB.min()) / (GB.max() - GB.min())
GB = np.clip(GB, 0, 1)
GB = (GB * 255).astype(np.uint8)

RB = R-B
RB = (RB - RB.min()) / (RB.max() - RB.min())
RB = np.clip(RB, 0, 1)
RB = (RB * 255).astype(np.uint8)

RGB = np.maximum(RG, RB, GB) 
RGB = RGB.astype(np.uint8)

RGB7 = Img_depth.copy()
RGB7 = cv2.cvtColor(RGB7, cv2.COLOR_BGR2GRAY)

RGB7 = RGB7.astype(np.uint8)

RGB7 = (RGB7 * 255)

cv2.imwrite(PATH + str(ind) + '_th1.png', RGB7)
cv2.imwrite(PATH + str(ind) + '_.th2.png', RGB)

(_, thresh2) = cv2.threshold(RGB, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
(_, thresh0) = cv2.threshold(RGB7, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#

THRESH = np.maximum(thresh0, thresh2)

cv2.imwrite(PATH + str(ind) + '_tho1.png', thresh0)
cv2.imwrite(PATH + str(ind) + '_tho2.png', thresh2)
cv2.imwrite(PATH + str(ind) + '_tho3.png', THRESH)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(3,3))
kernel2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))
kernel3 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(7,7))

edged = THRESH

ITER0 = int(np.round((COL/100 + ROW/100) * np.minimum(ROW, COL) / np.maximum(ROW, COL)))
edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel2, iterations=ITER0)

ITER = int(np.round(np.minimum(COL, ROW)/10) * np.maximum(COL, ROW) / np.minimum(ROW, COL))
ITER = ITER0 + ITER
edged = cv2.dilate(edged, kernel, iterations=ITER)

cv2.imwrite(PATH + str(ind) + '_dilate.png', edged)

edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=ITER0)

ITER2 = COL/10 + ROW/10
ITER2 = ITER2 * ((np.minimum(COL, ROW) / np.maximum(ROW, COL)))
ITER2 = int(np.round((ITER2 / (1 + ITER0)) * row / col))
edged = cv2.erode(edged, kernel, iterations=ITER2)

cv2.imwrite(PATH + str(ind) + '_erode.png', edged)

edged = cv2.Canny(edged, 50, 150)

cv2.imwrite(PATH + str(ind) + '_canny.png', edged)

edged = cv2.dilate(edged, kernel2, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
(cnts, _) = contours.sort_contours(cnts)

orig = IMG3.copy()

Ratio_row = col / 640
Ratio_col = row / 480

# loop over the contours individually
for index, c in enumerate(cnts):

    # if the contour is not sufficiently large, ignore it

    if cv2.contourArea(c) < 100:
       continue
    
	# compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    dimA = dA  * 0.026458333 * Ratio_col
    dimB = dB  * 0.026458333 * Ratio_row
    
    cv2.putText(orig, "{:.1f}cm".format(dimA),    
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
        
    cv2.putText(orig, "{:.1f}cm".format(dimB),    
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

cv2.imwrite(PATH + str(ind) + '_orig.png', orig)

