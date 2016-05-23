
import os

base_path = os.path.dirname(__file__)

class Paths:
    Results        = os.path.join(base_path, "../../data/results")
    Database       = os.path.join(base_path, "../../data/database")
    Segmentation   = os.path.join(base_path, "../../data/segmentation")
    Projects       = os.path.join(base_path, "../../data/labels")
    Models         = os.path.join(base_path, "../../data/models")
    #Images         = os.path.join(base_path, "../../data/images")
    Labels         = os.path.join(base_path, "../../data/labels")
    Data           = os.path.join(base_path, "../../data")
    Baseline       = os.path.join(base_path, "../../data/baseline")
    Reference      = os.path.join(base_path, "../../data/reference")
    #Reference      = os.path.join(base_path, "../../data/eval")
    Membranes      = '%s/labels/membranes/test'%(Reference)
    TestLabels     = '%s/labels/test'%(Reference)
    TestGrayscale  = '%s/images/test'%(Reference)
    TrainGrayscale = '%s/images/train'%(Reference)
    ValidGrayscale  = '%s/images/validate'%(Reference)
