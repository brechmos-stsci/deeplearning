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

model = ResNet50(weights='imagenet')

# ---------------------------------------------------------------------
# Load and pre-processing
# ---------------------------------------------------------------------

def gray2rgb(data):
    data_out = zeros((224,224,3,data.shape[0]))
    data_out[:,:,0] = data.transpose((1,2,0))
    data_out[:,:,1] = data.transpose((1,2,0))
    data_out[:,:,2] = data.transpose((1,2,0))

    return data_out

def rgb2plot(data):
    mindata, maxdata = prctile(data, (0.01, 99))
    return clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(uint8)

files = glob.glob('{}/Box Sync/DeepLearning/hubble/HST/cutouts/hubble_cutouts_*.npz'.format(os.environ['HOME']))

## Load in the data
allpredictions = OrderedDict()
for filename in files[:30]:
    print('processing {}'.format(filename.split('/')[-1]))
    npzfile = np.load(filename)
    data_orig = npzfile['cutouts']

    # Make gray scale location
    data = zeros((224,224,3,data_orig.shape[0]))
    data[:,:,0] = data_orig.transpose((1,2,0))
    data[:,:,1] = data_orig.transpose((1,2,0))
    data[:,:,2] = data_orig.transpose((1,2,0))

    # Display the data
    plt.figure(1)
    plt.clf()
    plt.imshow(rgb2plot(data[:,:,:,3]))
    plt.colorbar()
    plt.show()

    N = data.shape[3]

    # ---------------------------------------------------------------------
    # Now processing
    # ---------------------------------------------------------------------

    # Calculate the predicitons for all data
    N = data.shape[3]

    for ii in range(N):
        print('\r\t{} of {}'.format(ii, N), end='')
        x = data[0:224,0:224,:,ii]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        if sum(abs(x)) > 0.0001:
            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            predictions = decode_predictions(preds, top=100)[0]
        else:
            predictions = [('test', 'beaver', 0.0001),]
        
        allpredictions[(filename, ii)] = [(tt[1], tt[2]) for tt in predictions]
    print('\n')

# Get a list of all the unique labels used
labels = list(set([x[0] for k, y in allpredictions.items() for x in y ]))

# ----------------------------------------------------------
# Calculate tSNE
# ----------------------------------------------------------

print('Calculate the tSNE')
X = np.zeros((len(allpredictions), len(labels)))
for ii, (_, pred) in enumerate(allpredictions.items()):
    for jj in pred:
        X[ii,labels.index(jj[0])] = jj[1]
Y = TSNE(n_components=2).fit_transform(X)

