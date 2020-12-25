#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to label images,
# where each pixel has an ID that represents the ground truth label.
#
# Usage: json2labelImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdLabelImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt

# Image processing
from PIL import Image
from PIL import ImageDraw
from easydict import EasyDict as edict
import json
import numpy as np
# cityscapes imports
from .labels_huawei import label2id

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print('Reads labels as polygons in JSON format and converts them to label images,')
    print('where each pixel has an ID that represents the ground truth label.')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# Convert the given annotation to a label image
def createLabelImage(json_file):
    
    with open(json_file,'r') as f:
        ann=json.load(f)
    
    annotation=edict(ann)
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    labelImg = Image.new("P", size)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = [tuple(p) for p in obj.polygon]
        
        if len(polygon)<2:
            pass
        else:
            val=label2id(label)
            drawer.polygon( polygon, fill=val )
    return labelImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the label image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
#     - "color"    : classes are encoded using the corresponding colors
def json2labelImg(inJson,outImg):
    labelImg   = createLabelImage( inJson )
    cmap=color_map(256)
    palette=list(cmap.reshape(-1))
    # labelImg.convert('P',palette=palette)
    labelImg.putpalette(palette)
    labelImg.save( outImg )

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2labelImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if len(args) == 0:
        printError( "Missing input json file" )
    elif len(args) == 1:
        printError( "Missing output image filename" )
    elif len(args) > 2:
        printError( "Too many arguments" )

    inJson = args[0]
    outImg = args[1]

    json2labelImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
