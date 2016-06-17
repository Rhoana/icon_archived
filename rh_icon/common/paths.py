
import os
import rh_config

base_path = os.path.dirname(__file__)

root = rh_config.config\
    .get("icon", {})\
    .get("data", {})\
    .get("root", os.path.join(base_path, "..", "..", "data"))


def get_config_data_path(directory, key = None):
    '''Get a pathname from the configuration
    
    :param directory: the subdirectory under the data directory (the default
        path)
    :param key: The key in the data section of the icon config (defaults to
        the value of `directory`)
    '''
    default = os.path.join(root, directory)
    if key is None:
        key = directory
    return rh_config.config\
           .get("icon", {}) \
           .get("data", {}) \
           .get(key, default)


class Paths:
    Results        = get_config_data_path("results")
    Database       = get_config_data_path("database")
    Segmentation   = get_config_data_path("segmentation")
    Projects       = get_config_data_path("labels")
    Models         = get_config_data_path("models")
    Labels         = get_config_data_path("labels")
    Data           = root
    Baseline       = get_config_data_path("baseline")
    Reference      = get_config_data_path("reference")
    Membranes      = get_config_data_path(
                        '%s/labels/membranes/test'%(Reference), "membranes")
    TestLabels     = get_config_data_path(
                        '%s/labels/test'%(Reference), "test-labels")
    TestGrayscale  = get_config_data_path(
                        '%s/images/test'%(Reference), "test-images")
    TrainGrayscale = get_config_data_path(
                        '%s/images/train'%(Reference), "train-images")
    ValidGrayscale  = get_config_data_path(
                        '%s/images/validate'%(Reference), "validation-images")
    Tilespecs       = get_config_data_path("tilespecs")

all = [Paths]