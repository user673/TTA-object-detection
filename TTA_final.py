# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import numpy as np
import time
from NMS  import NMS
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def change_brightness(img, value):
    """func to change brightness 
    input: image to augment, value<0 - decreases, value>0 - increases
    output: augmented image
    """
    num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = int(-value)
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def closer(imag, scale):
    """func for  making image smaller
    input: image, scale as float ( scale<0 - smaller, scale>0 - cropping and making bigger - bad idea for od)
    output: augmented image
    """
    seq = iaa.Sequential([
        iaa.Affine(scale=scale )
    ])

    # Augment  image
    image_aug= seq(image=imag)

    
    image_aug_res = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
    return(image_aug)
def create_bbs(df,image):

    """func that create bb for scaling image
    input: df with xmin,xmax,ymin,ymax for each object ang original image
    output: BoundingBoxesOnImage object
    """
    b = []
    for i in range(len(df)):
        b.append(BoundingBox(x1 =df.iloc[i][0], y1 = df.iloc[i][1], x2 = df.iloc[i][2], y2 = df.iloc[i][3] ))
    bbs = BoundingBoxesOnImage(b,  shape=image.shape)
    return bbs

def transform_back(num, scale, res, image):
    
    """func  to transform augmented bb to original form
    input: image, scale as float, boxes,scores,labels for augmented images as res by num
    output: boxes,scores,labels in original form
    """
    boxes, scores, labels = res[num]
    df = pd.DataFrame(boxes[0])
    bbs = create_bbs(df, image)
    
    seq = iaa.Sequential([
        iaa.Affine(scale=1/scale)
    ])
    boxes = []
    bbs_aug = seq( bounding_boxes = bbs)

    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        boxes.append([int(after.x1), int(after.y1),int(after.x2), int(after.y2)])
    boxes = np.array([boxes])
    return boxes, scores, labels


def TTA(model, pred_image,thresh = 0.55, iou_thrash = 0.3):
    """func  to provide Test Time Augmentation
    input: image, model, thresh for scores, iou_thrash for NMS
    output: final boxes,scores,labels 
    """
    image_list = []
    image = read_image_bgr(pred_image)

    image_list.append(image)
    image_list.append(change_brightness(image, 70))
    image_list.append(change_brightness(image, -70))
    image_list.append(closer(image, 0.5))
    image_list.append(closer(change_brightness(image, -30), 0.5))
    image_list.append(closer(change_brightness(image, 40), 0.7))
    res = {}

    for i in range(len(image_list)):

        img = preprocess_image(image_list[i])
        img, scale = resize_image(img)
        
        # process every augmented image 
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
        print("processing time: ", time.time() - start)
        #boxes, scores, labels  = NMS(boxes, scores, labels, THRESH=.5, IoU_trash=.2)
        # save results 
        boxes /= scale
        res[i] = [boxes, scores, labels]

    # transform resulted bbs to original form
    many = []
    many.append(transform_back(0,1,res,image_list[0]))
    many.append(transform_back(1,1,res,image_list[1]))
    many.append(transform_back(2,1,res,image_list[2]))
    many.append(transform_back(3,0.5,res,image_list[3]))
    many.append(transform_back(4,0.5,res,image_list[4]))
    many.append(transform_back(5,0.7,res,image_list[5]))
    resss_l = []
    resss_s = []
    resss_b = []

    for i in range(len(many)):
        for j in range(len(many[i][0][0])):
            resss_b.append((many[i][0][0][j]))
            resss_l.append((many[i][2][0][j]))
            resss_s.append((many[i][1][0][j]))

    boxes = np.array([resss_b])
    scores = np.array([resss_s])
    labels = np.array([resss_l])
    
    boxes_NMS, scores_NMS, labels_NMS  = NMS(boxes, scores, labels, THRESH=thresh, IoU_trash=iou_thrash)
    return boxes_NMS,scores_NMS,labels_NMS 

