

class ButterflyDataset(object):
    '''A butterfly project data record'''
    
    def __init__(self, project_id, experiment, sample, dataset, channel):
        self.project_id = project_id
        self.experiment = experiment
        self.sample = sample
        self.dataset = dataset
        self.channel = channel

class ButterflyPlane(object):
    '''A butterfly image plane'''
    
    def __init__(self, project_id, image_id, z, width, height, xoff, yoff):
        self.project_id = project_id
        self.image_id = image_id
        self.z = z
        self.width = width
        self.height = height
        self.xoff = xoff
        self.yoff = yoff
    
    def __cmp__(self, other):
        return cmp(self.z, other.z)