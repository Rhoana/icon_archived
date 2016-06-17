import os
import sqlite3 as lite
import sys
import time
from datetime import datetime, date



from rh_icon.common.utility import Utility
from rh_icon.common.paths import Paths

#---------------------------------------------------------------------------
# Image datum
#---------------------------------------------------------------------------
class Image (object):
    PURPOSE_TRAINING = 0
    PURPOSE_VALIDATION = 1
    PURPOSE_TEST = 2
    PURPOSE_ALL = 3
    PURPOSE_ANNOTATED = 4
    
    def __init__(	self, id, purpose=0 ):
        self.id                       = id
        self.purpose                  = purpose
        self.segmentationReqTime      = None
        self.segmentationTime         = None
        self.segmentationPriority     = 0
        self.trainingTime             = None
        self.trainingPriority         = 0
        self.trainingStatus           = 0
        self.trainingScore            = 0
        self.annotationTime           = 0
        self.annotationLockTime       = 0
        self.annotationLockId         = 0
        self.annotationStatus         = 0
        self.modelModifiedTime        = 0
        self.creationTime             = None
        self.startTime                = None
        self.locked                   = 0
        self.annotationFile           = None
        self.segmentationFile         = None
        self.hasNewModel              = False

    def toJson(self):
        data = {}
        data['image_id']              = self.id
        data['training_time']         = self.trainingTime
        data['training_status']       = self.trainingStatus
        data['training_score']        = self.trainingScore
        data['segmentation_reqtime']  = self.segmentationReqTime
        data['segmentation_time']     = self.segmentationTime
        data['segmentation_priority'] = self.segmentationPriority
        data['segmentation_file']     = self.segmentationFile
        data['annotation_time']       = self.annotationTime
        data['annotaiton_locktime']   = self.annotationLockTime
        data['annotation_lockid']     = self.annotationLockId
        data['annotation_status']     = self.annotationStatus
        data['annotation_file']       = self.annotationFile
        data['model_mod_time']        = self.modelModifiedTime
        data['creation_time']         = self.creationTime
        data['start_time']            = self.startTime
        data['has_new_model']         = self.hasNewModel
        return data
