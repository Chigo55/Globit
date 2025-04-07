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

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def OS(IMG, image):

    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
    
    # construct the argument parse and parse the arguments
    
    [col, row, ch] = image.shape    
    [col, row, ch] = IMG.shape

    image = cv2.resize(image, (640, 480))
    image = (image * 255).astype(np.uint8)
    IMG = cv2.resize(IMG, (640, 480))

    image_depth = IMG.copy()

    results = model.predict(image, conf=0.25, fuse_model=False)
    
    output = results

    bboxes = output.prediction.bboxes_xyxy

    labels = output.prediction.labels    
    class_names = output.class_names

    random.seed(0)
    labels = [int(l) for l in list(labels)]
    label_colors = [tuple(random.choices(np.arange(0, 256), k=3)) for i in range(len(class_names))]
    
    IMG = np.zeros(image.shape, dtype='uint8')

    IMG3 = image.copy()

    IMG5 = IMG.copy()
    IMG6 = IMG.copy()

    COL = 0
    ROW = 0

    IMG6 = image.copy()

    for idx, bbox in enumerate(bboxes):
           bbox_left = int(bbox[0])
           bbox_top = int(bbox[1])
           bbox_right = int(bbox[2])
           bbox_bot = int(bbox[3])
                            
           colors = tuple(int(i) for i in label_colors[labels[idx]])

           cv2.rectangle(IMG6, (bbox_left, bbox_top), (bbox_right, bbox_bot), color=colors, thickness=2)

           pts2 = np.array([[bbox_left,bbox_top],[bbox_left,bbox_bot],[bbox_right,bbox_bot],[bbox_right,bbox_top]])

           COL = np.abs(bbox_top - bbox_bot)                
           ROW = np.abs(bbox_right - bbox_left)

           cv2.fillPoly(IMG5,[pts2],color=(255,255,255))

    IMG2 = (255 - IMG3) * (IMG5)
    Img_depth = (image_depth) * IMG5

    Img_depth = (Img_depth * 255).astype(np.uint8)
    
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
    (_, thresh2) = cv2.threshold(RGB, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)        
    (_, thresh0) = cv2.threshold(RGB7, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#

    THRESH = np.maximum(thresh0, thresh2)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(3,3))
    kernel2 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(5,5))

    edged = np.zeros((480, 640))

    edged = edged.astype(np.uint8)    

    edged = THRESH.copy()    

    orig = IMG3.copy()

    if COL & ROW != 0:
    
        ITER0 = int(np.round((COL/100 + ROW/100) * np.minimum(ROW, COL) / np.maximum(ROW, COL)))
        edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel2, iterations=2)

        ITER = int(np.round(np.minimum(COL, ROW)/10) * np.maximum(COL, ROW) / np.minimum(ROW, COL))
        ITER = ITER0 + ITER
        
        edged = cv2.dilate(edged, kernel, iterations=ITER)

        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

        ITER2 = COL/10 + ROW/10
        ITER2 = ITER2 * ((np.minimum(COL, ROW) / np.maximum(ROW, COL)))
        ITER2 = int(np.round((ITER2 / (1 + ITER0)) * row / col))

        edged = cv2.erode(edged, kernel, iterations=ITER2)

        edged = cv2.Canny(edged, 10, 50)

        edged = cv2.dilate(edged, kernel2, iterations=1)

                # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0]

        if len(cnts) != 0:

            (cnts, _) = contours.sort_contours(cnts)
        
            Ratio_row = col / 640
            Ratio_col = row / 480

                    # loop over the contours individually
            for index, c in enumerate(cnts):

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
        
    return orig

parser = argparse.ArgumentParser()

parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

args = parser.parse_args()

margin_width = 50

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()
    
transform = Compose([
        Resize(
            width=640,
            height=480,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

In_path = r'C:\Users\User\Desktop\Backup\Code\Depth-Anything-main\Depth-Anything-main\assets\fish_examples_video/'

DIR = os.listdir(In_path)

ind = 1

Ind = DIR[ind]

filename = In_path + Ind

raw_video = cv2.VideoCapture(filename)

frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

output_width = frame_width * 2 + margin_width
        
filename = os.path.basename(filename)

output_path = r'C:\Users\User\Desktop\Backup\Code\Depth-Anything-main\Depth-Anything-main\assets\Output/'
output_path = os.path.join(output_path, filename[:filename.rfind('.')] + '_video_depth.mp4')

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (640, 480))
    
while raw_video.isOpened():
    ret, raw_frame = raw_video.read()
    if not ret:
        break
            
    raw_frame = cv2.resize(raw_frame, (640, 480))

    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

    IMG = frame
            
    frame = transform({'image': frame})['image']

    frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(frame)

    depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
    depth = depth.cpu().numpy().astype(np.uint8)

    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
 
    OUt = OS(depth_color, IMG)
    out.write(OUt)
        
raw_video.release()
out.release()
    