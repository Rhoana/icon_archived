import cv2
import os
import tornado.web
import urllib

from rh_icon.database.db import DB
from rh_icon.common.imageaccess import read_image

class ImageHandler(tornado.web.RequestHandler):
    
    def get(self, project_id, image_id):
        project_id = urllib.unquote(project_id)
        image_id = urllib.unquote(image_id)
        img = read_image(project_id, image_id) 
        data = cv2.imencode(".tif", img)[1]
        self.set_header("Content-Type", "image/tif")
        self.write(data.tobytes())