#!/usr/bin/python

import os
import sys
import time
import numpy as np
import cPickle
import mahotas

import StringIO
import base64
import math
import zlib


import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from scipy.ndimage.interpolation import shift

#base_path = os.path.dirname(__file__)
#sys.path.insert(1,os.path.join(base_path, '../model'))
#sys.path.insert(2,os.path.join(base_path, '../external'))
#sys.path.insert(3,os.path.join(base_path, '../common'))

from utility import Utility
from activation_functions import rectified_linear
from logistic_sgd import LogisticRegression
from convlayer import LeNetConvPoolLayer

def get_max_pool_frag(convolved_image, offset1, offset2):
    image_width = convolved_image.shape[2]
    image_height = convolved_image.shape[3]
    
    convolved_image_shared = convolved_image[:,:,offset1:,offset2:]
    convolved_image_shared = convolved_image_shared.reshape(convolved_image_shared.shape.eval())

    pooled_out = downsample.max_pool_2d(
        input=convolved_image_shared,
        ds=(2,2),
        ignore_border=True
    )

    return pooled_out
    

def get_max_pool_fragments(convolved_image, clayer):
    start_time = time.clock()
    b = clayer.b.dimshuffle('x', 0, 'x', 'x')
    max_pooled_0_0 = get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=0)
    out_0_0 = rectified_linear(max_pooled_0_0 + b)

    max_pooled_0_1 = get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=1)
    out_0_1 = rectified_linear(max_pooled_0_1 + b)

    max_pooled_1_0 = get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=0)
    out_1_0 = rectified_linear(max_pooled_1_0 + b)

    max_pooled_1_1 = get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=1)
    out_1_1 = rectified_linear(max_pooled_1_1 + b)

    return (out_0_0, out_0_1, out_1_0, out_1_1)


def basic_convolution(image_shared, filterMap):
    conv_out = conv.conv2d(
        input=image_shared,
        filters=filterMap,
    )

    return conv_out


def get_convolution_output(image_shared, clayer):
    print 'get_convolution_output'
    print 'clayer.W:', clayer.W.eval().shape, ' image_shared:', image_shared.eval().shape

    output = basic_convolution(image_shared, clayer.W)
    output = theano.shared(np.float32(output.eval()), borrow=True)
    return output

def get_hidden_output(image_shared, hiddenLayer, nHidden, nfilt):
    W = hiddenLayer.W
    patchSize = np.int16(np.sqrt(W.shape.eval()[0] / np.double(nfilt)))
    W = np.rollaxis(W,1)
    W = W.reshape((nHidden,nfilt,patchSize,patchSize))

    b = hiddenLayer.b

    #flip kernel for convolution
    output = rectified_linear(basic_convolution(image_shared, W[:,:,::-1,::-1]) + b.dimshuffle('x', 0, 'x', 'x'))

    return output


def get_logistic_regression_output(image_shared, logregLayer, n_classes):
    output_shape = image_shared.shape

    W_lreg = logregLayer.W
    W_shape = W_lreg.shape
    W_lreg = np.rollaxis(W_lreg, 1)
    W_lreg = W_lreg.reshape((W_shape[1],W_shape[0],1,1))

    b_lreg = logregLayer.b

    # flip kernel for convolution
    output = basic_convolution(image_shared, W_lreg[:,:,::-1,::-1]) + b_lreg.dimshuffle('x', 0, 'x', 'x')

    output =  T.addbroadcast(output, 0)
    output = output.squeeze()
    output = output.flatten(2)
    output = T.nnet.softmax(output.T).T

    return output.reshape((n_classes,output_shape[2], output_shape[3]))


def extract_fragments( logreg_output, final_fragments_per_class ):
    for c in range( len(final_fragments_per_class) ):
        logreg_out = logreg_output[c,:,:]
        logreg_out = logreg_out.eval()

        final_fragments = final_fragments_per_class[c]
        final_fragments.append(logreg_out)


def gen_predicted_probs_n_classes(offsets, final_fragments_per_class, image):

    prob_img_per_class = []

    offset_jumps = np.int16(np.sqrt(len(offsets)))

    # compute final probability map for each class
    for final_fragments in final_fragments_per_class:

        print '-------:'
        prob_img = np.zeros(image.shape)

        offset_jumps = np.int16(np.sqrt(len(offsets)))
        for f, o in zip(final_fragments, offsets):
            prob_size = prob_img[o[0]::offset_jumps,o[1]::offset_jumps].shape
            f_s = np.zeros(prob_size)
            f_s[:f.shape[0], :f.shape[1]] = f.copy()
            prob_img[o[0]::offset_jumps,o[1]::offset_jumps] = f_s

        prob_img = prob_img.flatten()
        if len(prob_img_per_class) > 0:
            prob_img_per_class = np.column_stack((prob_img_per_class, prob_img))
        else:
            prob_img_per_class = prob_img

    predicted_classes = np.argmax( prob_img_per_class, axis=1 )
    predicted_probs   = np.max( prob_img_per_class, axis=1 )

    return predicted_probs, predicted_classes


def classify_image( image, classifier ):

    #image = mahotas.imread( path )
    #imageSize = 1024
    #image = image[0:imageSize,0:imageSize]
    #image = Utility.normalizeImage( image ) - 0.5

    imageSize = image.shape[0]

    start_time = time.clock()

    #GPU
    image_shared = theano.shared(np.float32(image), borrow=True)
    image_shared = image_shared.reshape((1,1,imageSize,imageSize))

    fragments = [image_shared]

    print "Convolutions"

    print '#convlayers:', len(classifier.convLayers)
    for clayer in classifier.convLayers:
        newFragments = []
        print '#fragments:', len(fragments)
        for img_sh in fragments:
            convolved_image = get_convolution_output(image_shared=img_sh, clayer=clayer)
            output = get_max_pool_fragments(convolved_image, clayer=clayer)
            newFragments.extend(output)

        fragments = newFragments

    #### now the hidden layer

    print "hidden layer"

    hidden_fragments = []

    for fragment in fragments:
        #hidden_out = get_hidden_output(image_shared=fragment, hiddenLayer=classifier.mlp.hiddenLayers[0], nHidden=200, nfilt=classifier.nkerns[-1])
        hidden_out = get_hidden_output(image_shared=fragment, hiddenLayer=classifier.mlp.hiddenLayers[0], nHidden=classifier.hiddenSizes[0], nfilt=classifier.nkerns[-1])
        hidden_fragments.append(hidden_out)

    ### VERIFIED CORRECT UNTIL HERE

    #### and the missing log reg layer

    print "logistic regression layer"

    final_fragments = []
    for fragment in hidden_fragments:
        logreg_out = get_logistic_regression_output(
                        image_shared=fragment, 
                        logregLayer=classifier.mlp.logRegressionLayer,
                        n_classes=classifier.n_classes)

        logreg_out = logreg_out.eval()
        final_fragments.append(logreg_out)

    print "assembling final image"

    prob_imgs = np.zeros( (classifier.n_classes, image.shape[0], image.shape[1]) )
    prob_img = np.zeros(image.shape)

    offsets_tmp = np.array([[0,0],[0,1],[1,0],[1,1]])

    if len(classifier.convLayers)>=1:
        offsets = offsets_tmp

    if len(classifier.convLayers)>=2:
        offset_init_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
        offset_init_2 = offset_init_1 * 2

        offsets = np.zeros((4,4,2))
        for o_1 in range(4):
            for o_2 in range(4):
                offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]

        offsets = offsets.reshape((16,2))

    if len(classifier.convLayers)>=3:
        offset_init_1 = offsets.copy()
        offset_init_2 =  np.array([[0,0],[0,1],[1,0],[1,1]]) * 4

        offsets = np.zeros((16,4,2))
        for o_1 in range(16):
            for o_2 in range(4):
                offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]

        offsets = offsets.reshape((64,2))

    # offsets = [(0,0),(0,2),(2,0),(2,2),
    #            (0,1),(0,3),(2,1),(2,3),
    #            (1,0),(1,2),(3,0),(3,2),
    #            (1,1),(1,3),(3,1),(3,3)]


    # offsets_1 = [(0,0),(0,4),(4,0),(4,4),
    #              (0,2),(0,6),(4,2),(4,6)]

    offset_jumps = np.int16(np.sqrt(len(offsets)))
    for f_fragments, o in zip(final_fragments, offsets):
        for c in range( classifier.n_classes ):
            f = f_fragments[c]
            prob_img = prob_imgs[c]

            prob_size = prob_img[o[0]::offset_jumps,o[1]::offset_jumps].shape
            f_s = np.zeros(prob_size)
            f_s[:f.shape[0], :f.shape[1]] = f.copy()
            prob_img[o[0]::offset_jumps,o[1]::offset_jumps] = f_s


    #floor of patchsize/2
    probs = []
    shift_amount = np.floor( classifier.patchSize/2 )
    for c in range( prob_imgs.shape[0] ):
        prob_img = shift(prob_imgs[c],(shift_amount, shift_amount)) 
        prob_img = prob_img.flatten()

        if len(probs) == 0:
            probs = prob_img
        else:
            probs = np.column_stack((probs, prob_img ))

    predicted_classes = np.argmax( probs, axis=1 )
    predicted_probs = np.max( probs, axis=1)

    total_time = time.clock() - start_time
    print "This took %f seconds." % (total_time)

    return predicted_probs, predicted_classes

