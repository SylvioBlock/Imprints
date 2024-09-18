#!/usr/bin/env python

#import xml.dom.minidom as minidom
#import xml.etree.cElementTree as cet
#import xml.etree.ElementTree as et
#import glob
import os
import math
import keras
import numpy as np
import cv2 as cv
#from rectangle import Rectangle
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import sys
import time
from settings import *

#--------------------------------------------------------------------------------
def draw_regions (image,xmin,ymin,xmax,ymax,label,categories):
   if True:
      cv.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)
      font = cv.FONT_HERSHEY_SIMPLEX
      cv.putText(image, str(categories[label]), (xmin,ymin-2), font, 0.5, (255,255,255), 1, cv.LINE_AA)

def split_image_with_overlap (image, image_name="none", chip_size=(800,800), overlap=200):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    iw, ih, _ = image.shape
    wn, hn = chip_size
    wn_overlap = wn - overlap
    hn_overlap = hn - overlap
    slices_w = int(math.ceil(float(iw)/wn_overlap))
    slices_h = int(math.ceil(float(ih)/hn_overlap))
    prefix = image_name.split(".")
    shifts = []
    image_chunks = []
    for i in range(slices_w):
       for j in range(slices_h):
          chunk_name = prefix[0] + '_' + str(i) + '_' + str(j) + '.png'
          if slices_w == 1 and slices_h == 1:
             chip = image[0 : iw, 0 : ih, : 3]
             shifts.append ((0,0,chunk_name))
          elif (i < (slices_w-1)) and (j < (slices_h-1)) and ((wn_overlap*(i+1))+overlap < iw) and ((hn_overlap*(j+1))+overlap < ih):
             chip = image[wn_overlap*i : (wn_overlap*(i+1))+overlap, hn_overlap*j: (hn_overlap*(j+1))+overlap, : 3]
             shifts.append ((wn_overlap*i,hn_overlap*j,chunk_name))
          elif (i < (slices_w-1)) and ((wn_overlap*(i+1))+overlap < iw):
             hsidea = max(0, ih - hn)
             hsideb = ih
             chip = image[wn_overlap*i : (wn_overlap*(i+1))+overlap, hsidea : hsideb, : 3]
             shifts.append ((wn_overlap*i,hsidea,chunk_name))
          elif j < (slices_h-1) and ((hn_overlap*(j+1))+overlap < ih):
             wsidea = max(0, iw - wn)
             wsideb = iw
             chip = image[wsidea : wsideb, hn_overlap*j: (hn_overlap*(j+1))+overlap, : 3]
             shifts.append ((wsidea,hn_overlap*j,chunk_name))
          else:
             hsidea = max(0, ih - hn)
             hsideb = ih
             wsidea = max(0, iw - wn)
             wsideb = iw
             chip = image[wsidea : wsideb, hsidea : hsideb, : 3]
             shifts.append ((wsidea,hsidea,chunk_name))
          image_chunks.append (chip)
          #cv2.imwrite(prefix[0] + "_" + str(i) + "_" + str(j) +  ".png", chip)

    return image_chunks, shifts

# Malisiewicz et al.
def non_maxima_suppression (boxes, overlapThresh):

   boxes = np.array (boxes) 
    
   # if there are no boxes, return an empty list
   if len(boxes) == 0:
      return []

   # if the bounding boxes integers, convert them to floats --
   # this is important since we'll be doing a bunch of divisions
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")

   pick = []

   # grab the coordinates of the bounding boxes
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]
   confidence = boxes[:,4]

   # compute the area of the bounding boxes and sort the bounding
   # boxes by the bottom-right y-coordinate of the bounding box
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(confidence)

   while len(idxs) > 0:
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])

      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      overlap1 = (w * h) / area[idxs[:last]]
      if (area[i] != 0):
         overlap2 = (w * h) / area[i]
      else:
         overlap2 = 0.0
      overlap = np.minimum (overlap1, overlap2)

      idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

   remaining = boxes[pick]

   D = []

   for line in remaining:
       xmin = int(line[0])
       ymin = int(line[1])
       xmax = int(line[2])
       ymax = int(line[3])
       confidence = float(line[4])
       clas = int(line[5])
       D.append((xmin, ymin, xmax, ymax, confidence, categories[clas]))

   return D

'''
'''
def image_cropping (raw_image):

   height = np.size (raw_image, 0)
   width = np.size (raw_image, 1) 
   dimage = cv.resize (raw_image, (int(scale * width), int(scale * height)), interpolation = cv.INTER_CUBIC)
   image_chunks, shifts = split_image_with_overlap (dimage)
   return image_chunks, shifts

def detection_classification_mapping (model, raw_image, image_chunks, shifts, sframe):

   clone = raw_image.copy();
   
   det_reg = [] 

   height = np.size (raw_image, 0)

   width = np.size (raw_image, 1) 
 
   index = 0
   
   for raw in image_chunks:

      image = preprocess_image (raw.copy())

      image, scale = resize_image (image)

      # Defect detection and classification:
      boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

      # correct for image scale
      boxes /= scale

      # Shifting coordinates to original image domain: 
      sx = shifts[index][1]
      sy = shifts[index][0]

      for box, score, label in zip(boxes[0], scores[0], labels[0]):
         if score > score_threshold:
            add = False  
            xmin = int((box[0] + sx) * inverse_scale)
            ymin = int((box[1] + sy) * inverse_scale)
            xmax = int((box[2] + sx) * inverse_scale)
            ymax = int((box[3] + sy) * inverse_scale)
            cx = (xmin + xmax)/2.0
            cy = (ymin + ymax)/2.0
            if (cx >= half_window) and (cx <= (width - half_window)) and (cy >= half_window) and (cy <= (height - half_window)):
               add = True 
               draw_regions (clone, xmin, ymin, xmax, ymax, label, categories)

            #Enlarging the detected regions to facilitate the tracking step! 
            xmin = xmin - enlarge
            ymin = ymin - enlarge
            xmax = xmax + enlarge
            ymax = ymax + enlarge
 
            if add:
               det_reg.append((xmin,ymin,xmax,ymax,score,label))

      index += 1
   
   cv.imwrite(sframe, clone)

   return det_reg

