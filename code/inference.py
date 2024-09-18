#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import heapq
import sys
import cv2
import math
import keras
import tensorflow as tf
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
#from ..preprocessing.pascal_voc import PascalVocGenerator
#from ..preprocessing.csv_generator import CSVGenerator
from ..utils.keras_version import check_keras_version
from ..utils.eval import evaluate
from ..utils.image import read_image_bgr, preprocess_image, resize_image
#from ..models.resnet import custom_objects
from ..utils.visualization import draw_detections, draw_annotations
from ..utils.colors import label_color
from keras_retinanet import models

categories = ["P1", "P2"]

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def split_image_with_overlap (image, image_name="nada", chip_size=(800,800), overlap=400):
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

def split_image (image, image_name="nada", chip_size=(800,800), equalsize=True):
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

    slices_w = int(math.ceil(float(iw)/wn))

    slices_h = int(math.ceil(float(ih)/hn))

    prefix = image_name.split(".")

    shifts = []

    image_chunks = []

    for i in range(slices_w):
       for j in range(slices_h):
          chunk_name = prefix[0] + '_' + str(i) + '_' + str(j) + '.png'
          if slices_w == 1 and slices_h == 1:
             chip = image[0 : iw, 0 : ih, : 3]
             shifts.append ((0,0,chunk_name))
          elif (i < (slices_w-1)) and (j < (slices_h-1)):
             chip = image[wn*i : wn*(i+1), hn*j : hn*(j+1), : 3]
             shifts.append ((wn*i,hn*j,chunk_name))
          elif i < (slices_w-1):
             if equalsize:
                hsidea = max(0, ih - hn)
             else:
                hsidea = hn * j
             hsideb = ih
             chip = image[wn*i : wn*(i+1), hsidea : hsideb, : 3]
             shifts.append ((wn*i,hsidea,chunk_name))
          elif j < (slices_h-1):
             if equalsize:
                wsidea = max(0, iw - wn)
             else:
                wsidea = wn * i
             wsideb = iw
             chip = image[wsidea : wsideb, hn*j : hn*(j+1), : 3]
             shifts.append ((wsidea,hn*j,chunk_name))
          else:
             if equalsize:
                hsidea = max(0, ih - hn)
             else:
                hsidea = hn * j
             hsideb = ih
             if equalsize:
                wsidea = max(0, iw - wn)
             else:
                wsidea = wn * i
             wsideb = iw
             chip = image[wsidea : wsideb, hsidea : hsideb, : 3]
             shifts.append ((wsidea,hsidea,chunk_name))
          image_chunks.append (chip)
          #cv2.imwrite(prefix[0] + "_" + str(i) + "_" + str(j) +  ".png", chip)

    return image_chunks, shifts

#--------------------------------------------------------------------------------
def draw_regions (image,xmin,ymin,xmax,ymax,label):
   #if categories[label] == 'helicopter':  
   if True:  
      cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,0,255),1)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(image, str(categories[label]), (xmin,ymin-2), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

#--------------------------------------------------------------------------------
if __name__ == '__main__':
   
   #scale = 1.5
   scale = 1.0
   inverse_scale = 1.0/scale
   score_threshold = 0.5
   max_detections = 10000
 
   # main function
   image_name = sys.argv[1]
   model_name = sys.argv[2]
   output_path = sys.argv[3]

   base = os.path.splitext(os.path.basename(image_name))[0]

   # make sure keras is the minimum required version
   check_keras_version()

   print('Loading model...')
   #model = models.load_model(model_name, backbone_name='resnet50', convert=True)
   model = models.load_model(model_name, backbone_name='resnet50')
   # print(model.summary())

   fdetections = open(output_path + base + '.txt', 'w')
   idetections = output_path + base + '.jpg'

   # reading image
   raw_image = read_image_bgr (image_name)
   height = np.size(raw_image, 0)
   width = np.size(raw_image, 1)
   plot_image = cv2.imread(image_name, 1)
   dimage = cv2.resize (raw_image, (int(scale * width), int(scale * height)), interpolation = cv2.INTER_CUBIC)
   image_chunks, shifts = split_image (dimage)
   #image_chunks, shifts = split_image_with_overlap (dimage)

   index = 0

   for raw in image_chunks:
   
      image = preprocess_image (raw.copy())

      image, scale = resize_image (image)

      boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

      # correct for image scale
      boxes /= scale

      # Shifting coordinates to original image domain: 
      sx = shifts[index][1]
      sy = shifts[index][0]

      for box, score, label in zip(boxes[0], scores[0], labels[0]):
         if score > score_threshold:
           xmin = int((box[0] + sx) * inverse_scale)
           ymin = int((box[1] + sy) * inverse_scale)
           xmax = int((box[2] + sx) * inverse_scale) 
           ymax = int((box[3] + sy) * inverse_scale)
           #fdetections.write (("%s %f %d %d %d %d %s\n") % (base, score, xmin, ymin, xmax, ymax, categories[label]))
           fdetections.write (("%s %f %d %d %d %d\n") % (categories[label], score, xmin, ymin, xmax, ymax))
           draw_regions (plot_image, xmin, ymin, xmax, ymax, label)
      index += 1

   fdetections.close()
   cv2.imwrite(idetections, plot_image)
