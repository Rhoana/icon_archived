'''imageaccess.py: unified image access

Images are accessed through Butterfly or from disk using the project ID and
the plane index to retrieve.
'''

from cv2 import imdecode
import json
import numpy as np
from tornado.httpclient import HTTPClient
from urllib2 import HTTPError

from rh_icon.database.db import DB
from rh_icon.common.settings import BUTTERFLY_HOST, BUTTERFLY_PORT

def read_image(project_id, image_id):
    '''Read an image
    
    :param project_id: the name of the project
    :param image_id: the name of the image within the project
    '''
    dataset = DB.getProjectButterflyDataset(project_id)
    plane = DB.getButterflyPlane(project_id, image_id)
    client = HTTPClient()
    url = "http://%s:%d/api/data" % (BUTTERFLY_HOST, BUTTERFLY_PORT) +\
        "?experiment=" + dataset.experiment +\
        "&sample=" + dataset.sample +\
        "&dataset=" + dataset.dataset +\
        "&channel=" + dataset.channel +\
        "&x=%d" % plane.xoff +\
        "&y=%d" % plane.yoff +\
        "&z=%d" % plane.z +\
        "&width=%d" % plane.width +\
        "&height=%d" % plane.height +\
        "&resolution=0" +\
        "&format=tif"
    response = client.fetch(url)
    if response.code >= 400:
        # TODO: response.headers -> HTTPError
        raise HTTPError(url, response.code, response.body, [], None)
    else:
        img = imdecode(np.frombuffer(response.body, np.uint8), 0)
        return img
        
def get_experiments():
    '''Get all experiments from the butterfly server'''
    url = "http://%s:%d/api/experiments"  % (BUTTERFLY_HOST, BUTTERFLY_PORT)
    return get_json_url(url)

def get_json_url(url):
    response = client.fetch(url) 
    if response.code >= 400:
        raise HTTPError(url, response.code, response.body, [], None)
    else:
        return json.loads(response.body)

def get_samples(experiment):
    '''Get all samples for an experiment
    '''
    url = "http://%s:%d/api/samples" % (BUTTERFLY_HOST, BUTTERFLY_PORT) +\
        "?experiment=" + experiment
    return get_json_url(url)

def get_datasets(experiment, sample):
    '''Get all datasets acquired on a sample'''
    
    url = "http://%s:%d/api/datasets" % (BUTTERFLY_HOST, BUTTERFLY_PORT) +\
            "?experiment=" + experiment +\
            "&sample=" + sample
    return get_json_url(url)

def get_channels(experiment, sample, dataset):
    '''Get all channels associated with a dataset'''
    url = "http://%s:%d/api/channels" % (BUTTERFLY_HOST, BUTTERFLY_PORT) +\
            "?experiment=" + experiment +\
            "&sample=" + sample +\
            "&dataset=" + dataset
    return get_json_url(url)


class ChannelData(object):
    def __init__(self, name, short_description, datatype, x, y, z):
        self.name = name
        self.short_description = short_description
        self.datatype = datatype
        self.x = x
        self.y = y
        self.z = z
        
def get_channel_data(experiment, sample, dataset, channel):
    url = "http://%s:%d/api/channels" % (BUTTERFLY_HOST, BUTTERFLY_PORT) +\
            "?experiment=" + experiment +\
            "&sample=" + sample +\
            "&dataset=" + dataset +\
            "&channel=" + channel
    d = get_json_url(url)
    return ChannelData(d["name"],
                       d["short-description"],
                       d["data-type"],
                       d["dimensions"]["x"],
                       d["dimensions"]["y"],
                       d["dimensions"]["z"])
    
    