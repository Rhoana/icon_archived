import tornado.ioloop
import tornado.web
import socket
import time
import os
import pkg_resources
import sys
import tempfile
import zlib
import StringIO
import base64
import numpy as np;
import json
from datetime import datetime, date

from rh_icon.database.db import DB
from rh_icon.common.paths import Paths
from rh_icon.common.utility import Utility;

class AnnotationHandler(tornado.web.RequestHandler):

    DefaultProject = 'default'

    def get(self):
        print ('-->AnnotationHandler.get...' + self.request.uri)
        #self.__logic.handle( self );
        tokens  = self.request.uri.split(".")
        imageId = tokens[1]
        projectId = tokens[2]
        action  = None if (len(tokens) < 4) else tokens[3]

        if action == 'getlabels':
            self.set_header('Content-Type', 'text')
            self.write(self.getLabels( imageId, projectId ))
        elif action == 'getuuid':
            #uuid.uuid1()
            guid = tokens[4]
            self.set_header('Content-Type', 'application/octstream')
            self.write(self.getuuid(projectId, imageId, guid))
        elif action == 'getannotations':
            self.set_header('Content-Type', 'text')
            self.write(self.getAnnotations( imageId, projectId ))
        elif action == 'getsegmentation':
            self.set_header('Content-Type', 'application/octstream')
            segTime = None if (len(tokens) < 5) else tokens[4]
            self.write(self.getsegmentation( imageId, projectId, segTime ))
        elif action == 'getstatus':
            guid = tokens[4]
            segTime = tokens[5]
            self.set_header('Content-Type', 'application/octstream')
            self.write(self.getstatus( imageId, projectId, guid, segTime ))
        else:
            data = pkg_resources.resource_string(
                __name__, "resources/annotate.html")
            with tempfile.NamedTemporaryFile() as fd:
                fd.write(data)
                fd.flush()
                self.render(fd.name)

    def post(self):
        print ('-->AnnotationHandler.post...' + self.request.uri)
        tokens  = self.request.uri.split(".")
        imageId = tokens[1]
        projectId = tokens[2]
        action  = None if (len(tokens) < 4) else tokens[3]

        print 'action: ', action
        if action == 'saveannotations':
            data = self.get_argument("annotations", default=None, strip=False)
            imageId = self.get_argument("id", default=None, strip=False)
            self.saveannotations(imageId, projectId, data)

    def close(self, signal, frame):
        print ('Saving..')
        sys.exit(0)


    def getuuid(self, projectId, imageId, guid):
        data = {}
        project = DB.getProject( projectId )
        task = DB.getImage( projectId, imageId )

        expiration = project.syncTime*4

        if task.annotationLockId == guid:
            data['uuid'] = DB.lockImage( projectId, imageId )
            now = datetime.now()
            annotationTime = datetime.strptime(task.annotationTime, '%Y-%m-%d %H:%M:%S')
            diff = now - annotationTime
            print 'diff: ', diff.total_seconds()
        elif task.annotationStatus == 1:
            now = datetime.now()
            annotationTime = datetime.strptime(task.annotationTime, '%Y-%m-%d %H:%M:%S')
            diff = now - annotationTime
            diff = diff.total_seconds()
            print 'time diff:', diff
            if diff > expiration:
                data['uuid'] = DB.lockImage( projectId, imageId )
        else:
            data['uuid'] = DB.lockImage( projectId, imageId )

        return Utility.compress(json.dumps( data ))

    def getLabels(self, imageId, projectId):
        path = '%s/%s.%s.json'%(Paths.Labels, imageId, projectId)
        content = '[]'
        try:
            with open(path, 'r') as content_file:
                content = content_file.read()
        except:
            pass
        return Utility.compress(content)

    def getAnnotations(self, imageId, projectId):

        path = '%s/%s.%s.json' % (Paths.Labels, imageId, projectId)
        # check the incoming folder first before to ensure
        # the most latest data is being referenced.

        path_incoming = 'resources/incoming/%s.%s.json'%(imageId,projectId)
        path = path_incoming if os.path.exists(path_incoming) else path

        #default to the labels template
        content = '[]'
        try:
            with open(path, 'r') as content_file:
                content = content_file.read()
        except:
            pass

        return Utility.compress(content)

    def saveannotations(self, imageId, projectId, data):
        print 'saveannotations....%s'%(imageId)
        # Always save the annotations to the labels folder.
        path = '%s/%s.%s.json'%(Paths.Labels, imageId,projectId)
        with open(path, 'w') as outfile:
            outfile.write(data)

        # Add a training and prediction task to the database
        DB.saveAnnotations( projectId, imageId, path )

        print '---->>>>>training images for :',projectId
        images = DB.getTrainingImages( projectId )
        for img in images:
            print img.id, img.annotationFile, img.annotationTime, img.annotationStatus



    def has_new_segmentation(self, imageId, projectId, segTime):
        # if no new segmentation, just return nothing
        if segTime is None or segTime == 'undefined':
            return True

        task = DB.getImage(projectId, imageId)
        taskSegTime = time.strptime(task.segmentationTime, '%Y-%m-%d %H:%M:%S')
        segTime = segTime.replace("%20", " ")
        segTime = time.strptime(segTime, '%Y-%m-%d %H:%M:%S')

        print ''
        print 'dbtime:', taskSegTime, 
        print ' qtime:', segTime
        if segTime == taskSegTime:
            return False

        return True


    def getsegmentation(self, imageId, projectId, segTime):
        data = []
        # if no new segmentation, just return nothing
        if not self.has_new_segmentation(imageId, projectId, segTime):
            return Utility.compress(data)

        path = os.path.join(Paths.Segmentation,
                            '%s.%s.seg'%(imageId,projectId))
        data = []
        # Settings.addPredictionImage( projectId, imageId)
        if os.path.isfile( path ):
            with open(path, 'r') as content_file:
                compressed = content_file.read()
                decompressed = zlib.decompress(compressed)
                data = base64.b64decode(decompressed)
        return Utility.compress(data)

    def getstatus(self, imageId, projectId, guid, segTime):
        # make sure this image prioritize for segmentation
        DB.requestSegmentation( projectId, imageId )
        task = DB.getImage(projectId, imageId);
        data = {}
        data['image'] = task.toJson()
        data['project'] = DB.getProject(projectId).toJson()
        data['has_new_segmentation'] = self.has_new_segmentation(imageId, projectId, segTime)
        return Utility.compress(json.dumps( data ))
