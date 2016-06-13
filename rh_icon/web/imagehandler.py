import tornado.web
import urllib

from db import DB


class ImageHandler(tornado.web.RequestHandler):
    
    def get(self):
        path = self.request.uri
        filename = urllib.unquote(path.rsplit("/")[-1])
        DB
        
        