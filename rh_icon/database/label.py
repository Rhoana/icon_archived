import os
import sqlite3 as lite
import sys


#---------------------------------------------------------------------------
# Label datum
#---------------------------------------------------------------------------
class Label (object):

    def __init__(self, 
                 index,     # unique index of the label
                 name,      # human readable name of the label
                 r,         # red component of color
                 g,         # green component of color
                 b          # blue component of color
                 ):
        self.index     = index
        self.name      = name
        self.r         = r
        self.g         = g
        self.b         = b

    def toJson(self):
        data = {}
        data['index']      = self.index
        data['name']       = self.name
        data['r']          = self.r
        data['g']          = self.g
        data['b']          = self.b
        return data
