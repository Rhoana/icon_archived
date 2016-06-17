#---------------------------------------------------------------------------
# database.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains database access layer implementation footer
#           sqlite3
#---------------------------------------------------------------------------

import os;
import sqlite3 as lite
import sys
import json
import glob
import time
import urllib
import uuid

from datetime import datetime, date

from rh_icon.common.utility import *
from rh_icon.common.paths import Paths
from rh_icon.database.tables import Tables
from rh_icon.database.project import Project
from rh_icon.database.db import DB

TILESPEC_TEMPLATE =    {
    "layer": 0, 
    "minIntensity": 0.0, 
    "mipmapLevels": {
        "0": {
            "imageUrl": "file:///path_to/ac3_input_0000.tif"
        }
        }, 
    "height": 1024, 
    "width": 1024, 
    "transforms": [
        {
            "className": "mpicbg.trakem2.transform.TranslationModel2D", 
            "dataString": "0 0"
            }], 
    "mfov": 1, 
    "tile_index": 1, 
    "maxIntensity": 255.0, 
    "bbox": [0, 1023, 0, 1023]}

def install(project):
    # remove any existing model files associated with this project
    path = '%s/best_%s.%s.pkl'%(Paths.Models, project.id, project.type )
    path = path.lower()
    if os.path.exists( path ):
        os.remove( path )

    # remove any existing model files associated with this project
    path = '%s/best_%s.%s.offline.pkl'%(Paths.Models, project.id, project.type )
    path = path.lower()
    if os.path.exists( path ):
        os.remove( path )

    # install the default model
    project.addLabel( 0, 'background', 255,0,0)
    project.addLabel( 1, 'membrane', 0,255,0)

    # setup training set
    paths = glob.glob('%s/*.tif'%(Paths.TrainGrayscale))
    paths.sort()

    # ensure that the tilespecs directory is present
    if not os.path.isdir(Paths.Tilespecs):
        os.makedirs(Paths.Tilespecs)
    
    DB.addButterflyProject(project.id, 
                           "icon", "ac3", "ac3", "raw")    
    # setup the first 20 images as a training set
    i = 0
    purpose = 0
    for z, path in enumerate(paths):
        name = Utility.get_filename_noext( path )
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, name, project.id)
        annFile = '%s/%s.%s.json'%(Paths.Labels, name, project.id)
        segFile = segFile.lower()
        annFile = annFile.lower()
        if not os.path.exists( segFile ):
            segFile = None
        if not os.path.exists( annFile ):
            annFile = None

        #print 'adding image: %s ann: %s'%(name, annFile)
        project.addImage( imageId=name, annFile=annFile, segFile=segFile,
                          purpose=purpose)
        i += 1
        if i > 20:
            # All subsequent images are validation
            purpose = 1
        #
        # Write the tilespec for this image
        #
        url = "file://" + urllib.pathname2url(os.path.abspath(path))
        tilespec = TILESPEC_TEMPLATE.copy()
        tilespec["layer"] = z
        tilespec["mipmapLevels"] = {"0": { "imageUrl": url}}
        tilespec_loc = os.path.join(Paths.Tilespecs, "W01_Sec%03d.json" % z)
        json.dump([tilespec], open(tilespec_loc, "w"))
        #
        # Write the butterfly plane record for this image
        #
        DB.addButterflyPlane(project.id, name, z, 1024, 1024, 0, 0)

    # store the project
    DB.storeProject( project )


#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
def main():
    print 'Icon database (installation interface)'

    # start in a blank slate
    Tables.drop();

    # install the tables
    Tables.create()

    # install mlp project
    p             = Project( id='testmlp', type='MLP')
    p.batchSize   = 16
    p.patchSize   = 39
    p.hiddenUnits = [500,500,500]
    install( p )

    # install cnn project	
    cnn              = Project( id='testcnn', type='CNN')
    cnn.trainTime    = 30
    cnn.learningRate = 0.1
    cnn.batchSize    = 128
    cnn.patchSize    = 39
    cnn.nKernels     = [48,48]
    cnn.kernelSizes  = [5,5]
    cnn.hiddenUnits  = [200]
    install( cnn )

if __name__=="__main__":
    main()
