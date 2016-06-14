import rh_logger
import tornado.web
import pkg_resources
import tempfile

class HelpHandler(tornado.web.RequestHandler):

    def get(self):
        rh_logger.logger.report_event(
            'GET ' + self.request.uri, [__name__])
        data = pkg_resources.resource_string(__name__, "resources/help.html")
        with tempfile.NamedTemporaryFile() as fd:
            fd.write(data)
            fd.flush()
            self.render(fd.name)
