import tornado.ioloop
import tornado.web
import socket
import mimetypes
import os
import pkg_resources
import posixpath
import sys
import time
import signal
# import datetime

from datetime import datetime, date

import tornado.httpserver
import rh_config
import rh_logger
from rh_icon.web.browserhandler import BrowseHandler
from rh_icon.web.annotationhandler import AnnotationHandler
from rh_icon.web.projecthandler import ProjectHandler
from rh_icon.web.helphandler import HelpHandler
from rh_icon.web.defaulthandler import DefaultHandler
from rh_icon.web.imagehandler import ImageHandler

from rh_icon.common.utility import Utility
from rh_icon.common.settings import ICON_PORT

MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 0.5

#
# Thank you Luigi:
# https://github.com/spotify/luigi/blob/
#    f7219c38121098d464011a094156d99b5b320362/luigi/server.py#L212
#
# Had vague idea that resources should be served via pkg_handler and
# Luigi did it, so I am cribbing from their implementation.
#
# TODO - share with Butterfly
#
class PkgResourcesHandler(tornado.web.RequestHandler):

    def initialize(self, path, default_filename=None):
	self.root = path

    def get(self, path):
	rh_logger.logger.report_event("GET " + self.request.uri)
	if path == "/":
	    self.redirect("index.html?"+self.request.query)
	path = posixpath.normpath(path)
	if os.path.isabs(path) or path.startswith(".."):
	    return self.send_error(404)

	extension = os.path.splitext(path)[1]
	if extension in mimetypes.types_map:
	    self.set_header("Content-Type", mimetypes.types_map[extension])
	elif extension == ".svg":
	    self.set_header("Content-Type", "image/svg+xml")
	data = pkg_resources.resource_string(
	    __name__, os.path.join(self.root, path))
	self.write(data)

class Application(tornado.web.Application):
    def __init__(self):
	handlers = [
	    (r"/", DefaultHandler),
	    (r"/browse.*", BrowseHandler),
	    (r"/project.*", ProjectHandler),
	    (r"/annotate.*", AnnotationHandler),
	    (r'/help*', HelpHandler),
	    (r'/settings/(.*)', PkgResourcesHandler,
	     {'path': 'resources/settings/'}),
	    (r'/js/(.*)', PkgResourcesHandler,
	     {'path': 'resources/js/'}),
	    (r'/js/vendors/(.*)', PkgResourcesHandler,
	     {'path': 'resources/js/vendors/'}),
	    (r'/css/(.*)', PkgResourcesHandler,
	     {'path': 'resources/css/'}),
	    (r'/uikit/(.*)', PkgResourcesHandler,
	     {'path': 'resources/uikit/'}),
	    (r'/images/(.*)', PkgResourcesHandler,
	     {'path': 'resources/images/'}),
	    (r'/open-iconic/(.*)', PkgResourcesHandler,
	     {'path': 'resources/open-iconic/'}),
	    (r'/train/(.*)', PkgResourcesHandler,
	     {'path': 'resources/train/'}),
            (r'/validate/(.*)', PkgResourcesHandler,
	     {'path': 'resources/validate/'}),
	    (r'/image/([^/]+)/(.*)', ImageHandler, {} )
	]

	settings = {
	    "template_path": 'resources',
	    "static_path": 'resources',
	}

	tornado.web.Application.__init__(self, handlers, **settings)


class Server():
    def __init__(self, name, port):
        self.name = name
        self.port = port
        application = Application()
        self.http_server = tornado.httpserver.HTTPServer( application )
        self.ip = socket.gethostbyname( socket.gethostname() )

    def print_status(self):
        Utility.print_msg ('.')
        Utility.print_msg ('\033[93m'+ self.name + ' running/' + '\033[0m', True)
        Utility.print_msg ('.')
        Utility.print_msg ('open ' + '\033[92m'+'http://' + self.ip + ':' + str(self.port) + '/' + '\033[0m', True)
        Utility.print_msg ('.')

    def start(self):
        self.print_status()
        self.http_server.listen( self.port )
        tornado.ioloop.IOLoop.instance().start()

    def stop(self):
        msg = 'shutting down %s in %s seconds'%(self.name, MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)
        Utility.print_msg ('\033[93m'+ msg + '\033[0m', True)
        io_loop = tornado.ioloop.IOLoop.instance()
        deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

        def stop_loop():
            now = time.time()
            if now < deadline and (io_loop._callbacks or io_loop._timeouts):
                io_loop.add_timeout(now + 1, stop_loop)
            else:
                io_loop.stop()
                Utility.print_msg ('\033[93m'+ 'shutdown' + '\033[0m', True, 'done')
        stop_loop()

def sig_handler(sig, frame):
    msg = 'caught interrupt signal: %s'%sig
    Utility.print_msg ('\033[93m'+ msg + '\033[0m', True)
    tornado.ioloop.IOLoop.instance().add_callback(shutdown)

def shutdown():
    server.stop()

def main():
    global server
    rh_logger.logger.start_process(
        "icon-webserver", "starting on port %d" % ICON_PORT)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    port = ICON_PORT
    name = 'icon webserver'
    server = Server(name, port)
    server.start()

if __name__ == "__main__":
    main()
