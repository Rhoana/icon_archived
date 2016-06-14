import os
import sys
import theano
import theano.tensor as T
import numpy
import numpy as np
import mahotas
import partition_comparison
import StringIO
import glob

from rh_icon.database.db import DB
from rh_icon.database.project import Project
from rh_icon.common.performance import Performance

from rh_icon.model.cnn.cnn import CNN

if __name__ == '__main__':

    # load the model to use for performance evaluation

    
    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    # retrieve the project settings from the database
    project = DB.getProject('evalcnn')


    print 'measuring offline performance...'

    # create the model based on the project
    model = CNN(
            rng=rng,
            input=x,
            offline=True,
            batch_size=project.batchSize,
            patch_size=project.patchSize,
            nkerns=project.nKernels,
            kernel_sizes=project.kernelSizes,
            hidden_sizes=project.hiddenUnits,
            train_time=project.trainTime,
            momentum=project.momentum,
            learning_rate=project.learningRate,
            path=project.path_offline,
            id=project.id)

    # Generate offline performance results
    # NOTE: this assumes an offline model has been fully trained.  This code
    # will generate variation of information and pixel errors for test images
    # specified in Paths.TestGrayscale
    nTests = 10
    Performance.measureOffline(model, project.id, mean=project.mean, std=project.std,maxNumTests=nTests)


    print 'measuring online performance...'

    model = CNN(
            rng=rng,
            input=x,
            batch_size=project.batchSize,
            patch_size=project.patchSize,
            nkerns=project.nKernels,
            kernel_sizes=project.kernelSizes,
            hidden_sizes=project.hiddenUnits,
            train_time=project.trainTime,
            momentum=project.momentum,
            learning_rate=project.learningRate,
            path=project.path,
            id=project.id)

    Performance.measureOnline(model, project.id, mean=project.mean, std=project.std,maxNumTests=nTests)
    #Performance.measureBaseline(model, project.id,maxNumTests=nTests)

