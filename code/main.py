#!/usr/bin/env python

# Python 2/3 compatibility
#from __future__ import print_function
#import sys
#PY3 = sys.version_info[0] == 3
#
#if PY3:
#    xrange = range

from common import draw_str, RectSelector
import glob
import os
import math
import keras
import numpy as np
import cv2 as cv
from rectangle import Rectangle
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import sys
import time
from tracking import MOSSE
from tracking import analyse
from detection import image_cropping, detection_classification_mapping, non_maxima_suppression
from settings import *

class App:

    def __init__(self):
        self.trackers = {}

    def run(self, cnn_model_path, label, out_path, show, image_list):
    
        previous_frame = None
        ntrackers = 0
        nframes = 0
        mean_time = 0.0
        olabel = label.split("/")
        label = olabel[-1]

        print('Loading convolutionl neural network model (retinanet) ...')
        model = models.load_model (cnn_model_path, backbone_name='resnet50')

        log = open ('imprint_defect_log.txt', 'w')

        for image_name in image_list:

            start = time.time()
            sframe = ((((image_name.split('_'))[-1]).split('.'))[0])
            frame = int(sframe) 
            
            self.frame = cv.imread (image_name, int(rgb))
            current_frame = cv.cvtColor (self.frame, cv.COLOR_BGR2GRAY)
            vis = self.frame.copy()

            dimgout = ((image_name.split("/"))[-1]).replace(".png", ".detec.jpg")
            timgout = ((image_name.split("/"))[-1]).replace(".png", ".track.jpg")
            txtout = ((image_name.split("/"))[-1]).replace(".png", ".txt")

            fout = open (out_path + '/' + txtout, 'w')

            R_slices, shifts = image_cropping (self.frame)

            D_hat = detection_classification_mapping (model, self.frame, R_slices, shifts, out_path + '/' + dimgout)

            D = non_maxima_suppression (D_hat, overlap_to_join)

            for id in list(self.trackers):
                self.trackers[id].update(current_frame)
                self.trackers[id].draw_state(vis, label)
                (x,y),(w,h),psr,border = self.trackers[id].pos, self.trackers[id].size, self.trackers[id].psr, self.trackers[id].border
                x1,y1,x2,y2 = x-w/2,y-w/2,x+w/2,y+w/2 

                #Removing a tracking region:
                if ((y <= half_window) or (psr < 5)):
                    analyse (self.trackers[id], log)
                    del self.trackers[id]
           
            if (show == 'True'):
                cv.imwrite(out_path + '/' + timgout, vis)
          
            # Testing if a new detection is already being tracked:
            index = 0
            remove = [False] * len(D) 
            for d in D:
               _d = Rectangle(d[0], d[1], d[2], d[3])
               for t in list(self.trackers):
                 (x,y),(w,h) = self.trackers[t].pos, self.trackers[t].size
                 _t = Rectangle(x-w/2,y-w/2,x+w/2,y+w/2)
                 # Computing the region overlap:
                 if _d.IoU(_t) > overlap_to_join:
                    remove[index] = True
                    self.trackers[t].ndetections += 1
                    if d[5] == 'P1':
                        #Severe imprint defect!
                        self.trackers[t].classification[1] += 1
                    elif d[5] == 'P2':    
                        #Mild imprint defect!
                        self.trackers[t].classification[0] += 1
                    self.trackers[t].det_conf = d[4]
                    self.trackers[t].det_xmin = d[0]
                    self.trackers[t].det_ymin = d[1]
                    self.trackers[t].det_xmax = d[2]
                    self.trackers[t].det_ymax = d[3]
                    self.trackers[t].det_clas = d[5]
                    self.trackers[t].det_update = frame
               index += 1

            # Adding new defects without any associated track region:
            index = 0
            for d in D:
                if remove[index] is False:
                   det_clas = d[5] 
                   tracker = MOSSE (current_frame, d, ntrackers, frame)
                   self.trackers[d] = tracker
                   ntrackers += 1
                index += 1

            # Saving results for each frame:
            for t in list(self.trackers):
                (x,y),(w,h),rt,dconf,dupd = self.trackers[t].pos, self.trackers[t].size,self.trackers[t].label,self.trackers[t].det_conf,self.trackers[t].det_update
                self.trackers[t].ntracking += 1
                dxmin, dymin, dxmax, dymax, dclas = self.trackers[t].det_xmin, self.trackers[t].det_ymin, self.trackers[t].det_xmax, self.trackers[t].det_ymax, self.trackers[t].det_clas
                ex = enlarge 
                if (self.trackers[t].det_update == frame):
                   fout.write(("%s %f %d %d %d %d %d %f %d %s\n") % (dclas, dconf, max(0,dxmin+ex), max(0,dymin+ex), dxmax-ex, dymax-ex, 1, self.trackers[t].psr, frame, label + '_' + str(rt)))
                else:
                   fout.write(("%s %f %d %d %d %d %d %f %d %s\n") % (dclas, dconf, max(0,(x-w/2)+ex), max(0,(y-w/2)+ex), (x+w/2)-ex, (y+w/2)-ex, 0, self.trackers[t].psr, frame, label + '_' + str(rt)))
            
            end = time.time()

            mean_time += (end - start)

            nframes += 1 

            print ('Processing image: ', image_name, ', elapsed time: ', (end - start))

            previous_frame = current_frame

            if (show == 'True'):
               cv.imshow("vis",vis)
               cv.waitKey(1)
            
            fout.close()

        print ('Average execution time: ', mean_time/float(nframes)) 

if __name__ == '__main__':
    cnn_model_path = sys.argv[4]
    show=sys.argv[5]
    image_list = glob.glob(sys.argv[1] + '/' + '*[0-9][0-9][0-9].png');
    image_list.sort()
    App().run(sys.argv[2], sys.argv[3], cnn_model_path, show, image_list)

