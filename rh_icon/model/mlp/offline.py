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
from rh_icon.common.paths import Paths

from rh_icon.model.mlp.mlp import MLP
from rh_icon.model.data import Data

if __name__ == '__main__':

    # load the model to use for performance evaluation
    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    project = DB.getProject('mlpnew') #evalmlp')

    model = MLP(
            id=project.id,
            rng=rng,
            input=x,
            momentum=0.0,
            offline=True,
            n_in=project.patchSize**2,
            n_hidden=project.hiddenUnits,
            n_out=len(project.labels),
            train_time=project.trainTime,
            #batch_size=project.batchSize,
            batch_size=50,
            patch_size=project.patchSize,
            path=project.path_offline)


    data = Data( project, offline=True, n_train_samples=700000, n_valid_samples=5000)
    #model.train(offline=True, data=data, mean=project.mean, std=project.std)
    #data.load(project )

    #print data.get_pixel_count(project)
    #exit(1)

    n_iterations = 5000
    for iteration in xrange(n_iterations):
        print 'iteration:', iteration
        model.train(data=data, offline=True, mean=project.mean, std=project.std)
