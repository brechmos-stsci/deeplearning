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


# Load the data
A = np.load('{}/Box Sync/DeepLearning/similarity_testing/similarity_0p0_threshold.npz'.format(os.environ['HOME']))
for k,v in A.items():
    globals()[k] = v

# Choose the slice
IMG = 1125
IMGS = [1125, 1225, 1325, 1425, 1429, 1432, 1434, 1485, 152, 1885, 2025, 845, 865, 898]

for IMG in IMGS:

    output = '{}/Box Sync/DeepLearning/similarity_testing/output_similarity_0p0_threshold'.format(os.environ['HOME'])

    # Find the most similar to index = 100, as an example
    inds = argsort(A_l2[IMG])[::-1] # want the largest number
    plt.figure(4)
    show_img(inds, IMG, carina_orig, 'L2 Norm [Slice {}]'.format(IMG))
    figure(4).savefig('{}/im_{}_l2norm.png'.format(output, IMG))

    inds = argsort(A_jaccard[IMG])[::-1] # want the largest number
    plt.figure(5)
    show_img(inds, IMG, carina_orig, 'Jaccard Similarity [Slice {}]'.format(IMG))
    figure(5).savefig('{}/im_{}_jaccard.png'.format(output, IMG))

    inds = argsort(sum((Y - Y[IMG])**2, axis=1)) # want the smallest number - smallest distance
    plt.figure(6)
    show_img(inds, IMG, carina_orig, 'tSNE Similarity [Slice {}]'.format(IMG))
    figure(6).savefig('{}/im_{}_tsne.png'.format(output, IMG))

    inds = argsort(A_dice[IMG])[::-1] # want the largest number
    plt.figure(7)
    show_img(inds, IMG, carina_orig, 'Dice Similarity [Slice {}]'.format(IMG))
    figure(7).savefig('{}/im_{}_dice.png'.format(output, IMG))

    inds = argsort(A_overlap[IMG])[::-1] # want the largest number
    plt.figure(8)
    show_img(inds, IMG, carina_orig, 'Overlap Similarity [Slice {}]'.format(IMG))
    figure(8).savefig('{}/im_{}_overlap.png'.format(output, IMG))

    inds = argsort(A_l1[IMG])[::-1] # want the largest number
    plt.figure(9)
    show_img(inds, IMG, carina_orig, 'L1 Norm [Slice {}]'.format(IMG))
    figure(9).savefig('{}/im_{}_l1norm.png'.format(output, IMG))

    inds = argsort(A_dotprod[IMG])[::-1] # want the largest number
    plt.figure(10)
    show_img(inds, IMG, carina_orig, 'Dot Product [Slice {}]'.format(IMG))
    figure(10).savefig('{}/im_{}_dotprod.png'.format(output, IMG))

