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

## Load in the data
carina_orig = np.load('{}/Box Sync/DeepLearning/carina/carina_224.npy'.format(os.environ['HOME']))

#carina_orig = carina_orig[:,:,:,:10]

# Fiddle with the data
#carina = zeros(carina_orig.shape)
#for ii in range(carina.shape[3]):
#    carina[:,:,:,ii] = exposure.equalize_hist(carina_orig[:,:,:,ii])

carina = carina_orig

# Display the data
plt.figure(1)
plt.clf()
plt.imshow(carina[:,:,:,3])
plt.colorbar()
plt.show()


N = carina.shape[3]

# ---------------------------------------------------------------------
# Now processing
# ---------------------------------------------------------------------

# Quick test on one dataset
x = carina[0:224,0:224,:,1]
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Calculate the predicitons for all data
labels = []
allpredictions = {}
N = carina.shape[3]

print('Calculating predictions')
for ii in range(N):
    print('\r\t{} of {}'.format(ii, N), end='')
    x = carina[0:224,0:224,:,ii]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    predictions = decode_predictions(preds, top=50)[0]
    
    allpredictions[ii] = [(tt[1], tt[2]) for tt in predictions if tt[2] > 0.001] 
print('\n')

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

# ----------------------------------------------------------
# Calculate L2 and Jaccard Similarities
# ----------------------------------------------------------

p1 = np.zeros((len(labels),))
p2 = np.zeros((len(labels),))
p1[ [labels.index(p[0]) for p in allpredictions[0]] ] = [p[1] for p in allpredictions[0]]
p2[ [labels.index(p[0]) for p in allpredictions[1]] ] = [p[1] for p in allpredictions[1]]

A_dotprod = np.zeros((N,N))
A_l1 = np.zeros((N,N))
A_l2 = np.zeros((N,N))
A_jaccard = np.zeros((N,N))
A_dice = np.zeros((N,N))
A_overlap = np.zeros((N,N))
print('Calculate the similarity measures')
for ii in np.arange(N):
    print('\r\t{} of {}'.format(ii,N), end='')
    p1 = np.zeros((len(labels),))
    p1[ [labels.index(p[0]) for p in allpredictions[ii]] ] = [p[1] for p in allpredictions[ii]]

    for jj in np.arange(N):
        p2 = np.zeros((len(labels),))
        p2[ [labels.index(p[0]) for p in allpredictions[jj]] ] = [p[1] for p in allpredictions[jj]]
        A_dotprod[ii,jj] = np.dot(p1,p2)

        # L1 Distance
        A_l1[ii,jj] = sum( abs(p1-p2) )

        # L2 Distance
        A_l2[ii,jj] = sqrt( sum( (p1-p2)**2 )  )

        # Jaccard similarity
        numerator = set([labels.index(p[0]) for p in allpredictions[ii]]) & set([labels.index(p[0]) for p in allpredictions[jj]])
        denominator = set(set([labels.index(p[0]) for p in allpredictions[ii]]) | set([labels.index(p[0]) for p in allpredictions[jj]]))
        A_jaccard[ii,jj] = len(numerator)/len(denominator)

        # Dice's Coefficient
        numerator = np.dot(p1,p2)
        denominator = sum(p1) + sum(p2)
        A_dice[ii,jj] = 2*numerator / denominator

        # Overlap Coefficient
        numerator = np.dot(p1,p2)
        denominator = min(sum(p1), sum(p2))
        A_overlap[ii,jj] = numerator / denominator

np.savez('session.npz', 
        A_dotprod= A_dotprod, A_l1= A_l1, A_l2= A_l2, A_jaccard= A_jaccard, 
        A_dice= A_dice, A_overlap = A_overlap , carina_orig=carina_orig, carina=carina, X=X, Y=Y)


asdf

A = A_l2

plt.figure(2)
plt.clf()
plt.imshow(A)
plt.colorbar()
plt.clim((0,1))

plt.figure(3)
plt.clf()
plt.plot(A[4])

def show_img(inds, IMG, carina_orig, title):
    plt.clf()
    subplot(4,4,1)
    tt = carina_orig[:,:,:,IMG]
    imshow(toimage(tt))
    axis('off')
    for ii in arange(15):
        subplot(4,4,ii+2)
        tt = carina_orig[:,:,:,inds[ii]]
        imshow(toimage(tt))
        axis('off')
    suptitle(title)

np.savez('session.npz', 
        A_dotprod= A_dotprod, A_l1= A_l1, A_l2= A_l2, A_jaccard= A_jaccard, 
        A_dice= A_dice, A_overlap = A_overlap , carina_orig=carina_orig, carina=carina, X=X, Y=Y)

IMG = 8

# Find the most similar to index = 100, as an example
inds = argsort(A_l2[IMG])[::-1] # want the largest number
plt.figure(4)
show_img(inds, IMG, carina_orig, 'L2 Norm [Slice {}]'.format(IMG))
figure(4).savefig('output/im_{}_l2norm.png'.format(IMG))

inds = argsort(A_jaccard[IMG])[::-1] # want the largest number
plt.figure(5)
show_img(inds, IMG, carina_orig, 'Jaccard Similarity [Slice {}]'.format(IMG))
figure(5).savefig('output/im_{}_jaccard.png'.format(IMG))

inds = argsort(sum((Y - Y[IMG])**2, axis=1)) # want the smallest number - smallest distance
plt.figure(6)
show_img(inds, IMG, carina_orig, 'tSNE Similarity [Slice {}]'.format(IMG))
figure(6).savefig('output/im_{}_tsne.png'.format(IMG))

inds = argsort(A_dice[IMG])[::-1] # want the largest number
plt.figure(7)
show_img(inds, IMG, carina_orig, 'Dice Similarity [Slice {}]'.format(IMG))
figure(7).savefig('output/im_{}_dice.png'.format(IMG))

inds = argsort(A_overlap[IMG])[::-1] # want the largest number
plt.figure(8)
show_img(inds, IMG, carina_orig, 'Overlap Similarity [Slice {}]'.format(IMG))
figure(8).savefig('output/im_{}_overlap.png'.format(IMG))

inds = argsort(A_l1[IMG])[::-1] # want the largest number
plt.figure(9)
show_img(inds, IMG, carina_orig, 'L1 Norm [Slice {}]'.format(IMG))
figure(9).savefig('output/im_{}_l1norm.png'.format(IMG))

inds = argsort(A_dotprod[IMG])[::-1] # want the largest number
plt.figure(10)
show_img(inds, IMG, carina_orig, 'L1 Norm [Slice {}]'.format(IMG))
figure(10).savefig('output/im_{}_dotprod.png'.format(IMG))
