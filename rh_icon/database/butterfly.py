

class ButterflyDataset(object):
    '''A butterfly project data record'''
    
    def __init__(self, project_id, experiment, dataset, channel):
        self.project_id = project_id
        self.experiment = experiment
        self.dataset = dataset
        self.channel = channel
        self.__planes = []
    
    def add_plane(self, plane):
        '''Add a plane to the list of planes'''
        self.__planes.append(plane)
    
    def planes(self):
        '''The butterfly planes in the dataset'''
        return sorted(self.__planes)

class ButterflyPlane(object):
    '''A butterfly image plane'''
    
    def __init__(self, z, width, height, xoff, yoff):
        self.z = z
        self.width = width
        self.height = height
        self.xoff = xoff
        self.yoff = yoff
    
    def __cmp__(self, other):
        return cmp(self.z, other.z)