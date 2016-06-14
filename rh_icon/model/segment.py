#---------------------------------------------------------------------------
# predict.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a module that manages
#           the module for segmenting images.  It runs the latest activated
#           project's classifier to perform segmentation on all images in
#           in the project. 
#---------------------------------------------------------------------------

import os
import sys
import signal
import threading
import time
import numpy as np
import StringIO
import base64
import math
import zlib

import mahotas
import glob

from rh_icon.model.manager import Manager
from rh_icon.common.utility import Utility
from rh_icon.common.utility import enum
from rh_icon.common.settings import Settings
from rh_icon.common.paths import Paths
from rh_icon.database.db import DB
from rh_icon.common.performance import Performance

#---------------------------------------------------------------------------
class Prediction(Manager):

    #-------------------------------------------------------------------
    # Reads the input image and normalize it.  Also creates
    # a version of the input image with corner pixels
    # mirrored based on the sample size
    # arguments: - path : directory path where image is located
    #            - id   : the unique identifier of the image
    #            - pad  : the amount to pad the image by.
    # return   : returns original image and a padded version
    #-------------------------------------------------------------------
    def __init__(self):
        Manager.__init__( self, 'prediction')
        self.high     = []
        self.low      = []
        self.priority = 0
        self.modTime  = None
        self.revision = 0

    def can_load_model(self, path):
        return os.path.exists( path )

    #-------------------------------------------------------------------
    # Retrieve segmentation tasks from database and call classifier
    # to perform actual work.
    #-------------------------------------------------------------------
    def work(self, project):

        if not self.online:
            self.work_offline(project)
            self.done = True
            return

        start_time = time.clock()

        if project is None:
            return

        print 'prediction.... running', len(self.high)

        if len(self.high) == 0:
            self.high = DB.getPredictionImages( project.id, 1)


        #FG - march 4th 2016
        #if len(self.low) == 0:
        #    self.low = DB.getPredictionImages( project.id, 0 )

        '''
        for img in self.high:
            print 'hid:', img.id, img.modelModifiedTime, img.segmentationTime

        print '----'
        for img in self.low:
            print 'lid:', img.id, img.modelModifiedTime, img.segmentationTime

        exit(1)
        '''

        task = None
        if (self.priority == 0 or len(self.low) == 0) and len(self.high) > 0:
            self.priority = 1
            task = self.high[0]
            del self.high[0]
        elif len(self.low) > 0:
            self.priority = 0
            task = self.low[0]
            del self.low[0]

        if task == None:
            return

        has_new_model = (self.modTime != project.modelTime)
        revision = DB.getRevision( project.id )
        print 'revision:', revision
        #has_new_model = (revision != self.revision or has_new_model)

        # reload the model if it changed
        if has_new_model:
            #self.revision = revision
            print 'initializing...'
            self.model.initialize()
            self.modTime = project.modelTime

        # read image to segment
        basepath = Paths.TrainGrayscale if task.purpose == 0 else Paths.ValidGrayscale
        path = '%s/%s.tif'%(basepath, task.id)
        #success, image = Utility.get_image_padded(path, project.patchSize ) #model.get_patch_size())

        print 'segment - path:', path
        print 'priority - ', task.segmentationPriority
        # perform segmentation

        Utility.report_status('segmenting %s'%(task.id),'')
        #probs = self.model.predict( path )
        #probs = self.model.classify( image )

        # serialize to file
        segPath = '%s/%s.%s.seg'%(Paths.Segmentation, task.id, project.id)
        #self.save_probs( probs, project.id, task.id )
        self.classify_n_save( path, segPath, project )         

        end_time = time.clock()
        duration = (end_time - start_time)
        DB.finishPrediction( self.projectId, task.id, duration, self.modTime )

        # measure performance if new model
        #if has_new_model:
        #	Performance.measureOnline( self.classifier.model, self.projectId )

    #-------------------------------------------------------------------
    # perform offlne segmentation of images in a specific directory
    #-------------------------------------------------------------------
    def work_offline(self, project):
        
        imagePaths = sorted( glob.glob( '%s/*.tif'%(Paths.TrainGrayscale)  ) )

        for path in imagePaths:
            if self.done:
                break

            name = Utility.get_filename_noext( path )
          
            print 'path:', path 
            Utility.report_status('segmenting', '%s'%(name))

            #segPath = '%s/%s.offline.seg'%(Paths.TrainGrayscale, name)
            segPath = '%s/%s.%s.offline.seg'%(Paths.Segmentation, name, project.id)

            self.classify_n_save( path, segPath, project )

    def classify_n_save(self, imagePath, segPath, project):

        image = mahotas.imread( imagePath )
        #image = Utility.normalizeImage( image ) - project.mean
        image = Utility.normalizeImage( image )

        # classify the image
        #prob = self.model.classify( image=image, mean=project.mean, std=project.std )
        prob = self.model.predict( image=image, mean=project.mean, std=project.std, threshold=project.threshold)


        #TODO: how to deal with multiple labels
        # extract the predicted labels
        '''
        prob[ prob >= project.threshold ] = 9
        prob[ prob <  project.threshold ] = 1
        prob[ prob == 9                 ] = 0
        '''
        prob = prob.astype(dtype=int)
        prob = prob.flatten()
        print 'results:', np.bincount( prob ), self.revision
        self.save_probs( prob, segPath)

    #-------------------------------------------------------------------
    # save probability map
    #-------------------------------------------------------------------
    def save_probs(self, data, path):
        output = StringIO.StringIO()
        output.write(data.tolist())
        content = output.getvalue()
        encoded = base64.b64encode(content)
        compressed = zlib.compress(encoded)
        with open(path, 'w') as outfile:
            outfile.write(compressed)

			
manager = None
def signal_handler(signal, frame):
        if manager is not None:
                manager.shutdown()

def main():
    print sys.argv
    Utility.report_status('running prediction module', '')
    signal.signal(signal.SIGINT, signal_handler)

    manager = Prediction()
    Manager.start( sys.argv, manager )
    
#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
    main()