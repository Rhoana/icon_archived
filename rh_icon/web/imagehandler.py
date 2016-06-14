import tornado.web
import urllib

from rh_icon.database.db import DB


class ImageHandler(tornado.web.RequestHandler):
    
    def get(self):
        path = self.request.uri
        filename = urllib.unquote(path.rsplit("/")[-1])
        DB
        
        