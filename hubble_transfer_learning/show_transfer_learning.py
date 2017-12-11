import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage import data, exposure
import numpy as np
import scipy.misc
from scipy.misc import toimage
import glob
from PIL import Image
from skimage import exposure
import requests
from io import BytesIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

import shelve

figure(2)
clf()

while True:
    # Plot the tSNE 
    figure(2)
    plot(Y[:,0], Y[:,1], 'b.')

    # Find the points closest to the tsne_coord
    tsne_coord = ginput(1, timeout=0)[0]
    tsne_coord = [int(x) for x in tsne_coord]
    print('Coordinates: {}'.format(tsne_coord))
    inds = argsort(sum( (Y-[tsne_coord[0],tsne_coord[1]])**2, axis=1))
    print(inds[:9])

    # Display the cutouts corresponding to the closest in tSNE space.
    figure(3)
    clf()
    for ii in range(9):
        filename, cutout = list(allpredictions.keys())[inds[ii]]
        npzfile = np.load(filename)
        data = rgb2plot(npzfile['cutouts'][ cutout])
        subplot(3,3,ii+1)
        imshow(data)
        xticks([]), yticks([])
        gray()
        clow, chigh = prctile(data, (1, 99))
        clim((clow, chigh))
        title('{} {}'.format(filename.split('/')[-1].replace('hubble_cutouts_', ''), cutout ), fontsize=10)
    figure(3).canvas.draw()

    figure(2)
    clf()
    plot(Y[:,0], Y[:,1], 'b.')
    plot(Y[inds[:9],0], Y[inds[:9],1], 'go')
    figure(2).show()
    print('\tDone display...')
