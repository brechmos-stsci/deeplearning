import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage import data, exposure
import numpy as np
import scipy.misc
from scipy.misc import toimage
from PIL import Image
from skimage import exposure
import requests
from io import BytesIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
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

## Load in the data
npzfile = np.load('{}/Box Sync/DeepLearning/hubble/HST/cutouts/hubble_cutouts_u25y0303t.npz'.format(os.environ['HOME']))
#npzfile = np.load('{}/Box Sync/DeepLearning/hubble/HST/cutouts/hubble_cutouts_u3j10106r.npz'.format(os.environ['HOME']))
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
labels = []
allpredictions = {}
N = data.shape[3]

print('Calculating predictions')
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
        print('Predictions {}'.format(predictions))
    else:
        predictions = [('test', 'beaver', 0.0001),]
    
    allpredictions[ii] = [(tt[1], tt[2]) for tt in predictions]
print('\n')

# Get a list of all the unique labels used
labels = list(set([x[0] for k, y in allpredictions.items() for x in y ]))

# ----------------------------------------------------------
# Calculate tSNE
# ----------------------------------------------------------

# Reformat the all information
print('Calculate the tSNE')
X = np.zeros((len(allpredictions), len(labels)))
for ii in range(len(allpredictions)):
    for jj in allpredictions[ii]:
        X[ii,labels.index(jj[0])] = jj[1]
Y = TSNE(n_components=2).fit_transform(X)

figure(2)
clf()
plot(Y[:,0], Y[:,1], '.')

## ----------------------------------------------------------
## Calculate L2 and Jaccard Similarities
## ----------------------------------------------------------
#
#p1 = np.zeros((len(labels),))
#p2 = np.zeros((len(labels),))
#p1[ [labels.index(p[0]) for p in allpredictions[0]] ] = [p[1] for p in allpredictions[0]]
#p2[ [labels.index(p[0]) for p in allpredictions[1]] ] = [p[1] for p in allpredictions[1]]
#
#A_jaccard = np.zeros((N,N))
#print('Calculate the similarity measures')
#for ii in np.arange(N):
#    print('\r\t{} of {}'.format(ii,N), end='')
#    p1 = np.zeros((len(labels),))
#    p1[ [labels.index(p[0]) for p in allpredictions[ii]] ] = [p[1] for p in allpredictions[ii]]
#
#    for jj in np.arange(N):
#        # Jaccard similarity
#        numerator = set([labels.index(p[0]) for p in allpredictions[ii]]) & set([labels.index(p[0]) for p in allpredictions[jj]])
#        denominator = set(set([labels.index(p[0]) for p in allpredictions[ii]]) | set([labels.index(p[0]) for p in allpredictions[jj]]))
#        A_jaccard[ii,jj] = len(numerator)/len(denominator)
#
#
