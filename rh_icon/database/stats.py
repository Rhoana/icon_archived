import os
import sqlite3 as lite
import sys

#---------------------------------------------------------------------------
# Label datum
#---------------------------------------------------------------------------
class TrainingStats (object):

    def __init__(   self, validationError=0.0, trainingCost=0.0, trainingTime=None):
        self.validationError = validationError
        self.trainingCost    = trainingCost
        self.trainingTime    = trainingTime

    def toJson(self):
        data = {}
        data['validation_error']  = self.validationError
        data['training_cost']     = self.trainingCost
        data['training_time']     = self.trainingTime
        return data
