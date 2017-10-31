import numpy as np
import os,sys
from scipy import misc
from matplotlib import pyplot as plt
import pickle as pk
from PIL import Image
from astropy.io import fits
from scipy import ndimage
from skimage import exposure

def plot_trc_box(ax, trc_in, clr, dd):
    ax.plot([trc_coords[trc_in, 1], trc_coords[trc_in, 1], 
                                                            trc_coords[trc_in, 1]+xs,
                                                            trc_coords[trc_in, 1]+xs,trc_coords[trc_in, 1]],
             [trc_coords[trc_in, 0], trc_coords[trc_in, 0]+xs, trc_coords[trc_in, 0]+xs,
             trc_coords[trc_in, 0],trc_coords[trc_in, 0]], color=clr, alpha=np.clip(1.0/dd, 0.2, 1))


def dist2D(a):
    N = a.shape[0]
    A = np.matlib.repmat(a[:,0], N,1)
    B = np.matlib.repmat(a[:,0][np.newaxis,:].T, 1,N)

    C = np.matlib.repmat(a[:,1], N,1)
    D = np.matlib.repmat(a[:,1][np.newaxis,:].T, 1,N)

    return np.sqrt((A-B)**2 + (C-D)**2)

# Load the TIFF file
im = Image.open('/Users/crjones/Box/Science/HargisDDRF/Carina/CarinaDisplay/carina.tiff')
imarr = np.asarray(im)
print(imarr.shape)

# Create trc_coords
xs = 224
nx =np.int(imarr.shape[0]/xs)-1
ny =np.int(imarr.shape[1]/xs)-1
stride = 4
trc_coords = np.zeros([nx*ny*stride*stride, 2])
images = np.zeros([xs, xs, 3, nx*ny*stride*stride])
for i in np.arange(nx*stride):
    for j in np.arange(ny*stride):
        trc_coords[int(j+i*ny*stride), 0] = int(i*xs/stride)
        trc_coords[int(j+i*ny*stride), 1] = int(j*xs/stride)

# Load the 
images = np.load('/Users/crjones/Box/Science/HargisDDRF/carina_224.npy')

# Load the coordinates.
tsne = np.load('/Users/crjones/Box/Science/HargisDDRF/Carina/carina_tsne_coords.npy')
#tsne = np.load('/Users/crjones/Box/Science/HargisDDRF/Carina/carina_nopreproc_rotations_Xception.npy')[0]

# Create mapping
dist_tsne_grid = dist2D(tsne)
dist_real_grid = dist2D(trc_coords)
diag = np.zeros([tsne.shape[0], tsne.shape[0]])
diag = 1-np.diag(np.ones((tsne.shape[0],)))

wh = np.asarray(np.where(np.ravel((dist_tsne_grid < 1) & (dist_real_grid > 1024) & (diag == 1.0))))
    
axarr = [None, None]

def display_plot():
    plt.figure(5, figsize=[2500/scale, 1500/scale])
    plt.clf()

    axarr[0] = plt.subplot(1,2,1)
    axarr[1] = plt.subplot(1,2,2)
    gcf().tight_layout()

    axarr[0].imshow(imarr, origin='upper')
    axarr[0].axis("off")

    axarr[1].scatter(tsne[:,0], tsne[:, 1], facecolors='black', edgecolor='black', alpha=0.2)
    plt.show()

    return axarr

nsimilar = 50
scale=200.0

def onclick(event):
    y,x = event.xdata, event.ydata

    print('x,y {} {}'.format(x,y))

    axarr = display_plot()

    print('event.x {}'.format(event.x))
    if event.inaxes == axarr[0]:
        x, y = int(x), int(y)
        x = x - 112
        y = y - 112
        # convert to point on carina
        trc_in = np.argmin((trc_coords[:,0]-x)**2+(trc_coords[:,1]-y)**2)
    else:
        trc_in = np.argmin((tsne[:,0]-y)**2+(tsne[:,1]-x)**2)
         
    print('trc_in {}'.format(trc_in))
    dists = np.ravel(dist_tsne_grid[trc_in, :])
    print('dists {} {}'.format(dists, sum(dists == 0)))
    closest = (np.argsort(dists))[0:nsimilar]
    print('closts {}'.format(closest))

    axarr = display_plot()

    # Plot on carina image
    for i in closest:
        plot_trc_box(axarr[0], i, 'yellow', dists[i])
    plot_trc_box(axarr[0], trc_in, 'red', 0.0001)

    # Plot on tSNE scatter plot
    for i in closest: 
        axarr[1].scatter(tsne[i,0], tsne[i, 1], facecolors=None, edgecolor='yellow', alpha=np.clip(2.0/dists[i], 0.2, 1))
    axarr[1].scatter(tsne[trc_in,0], tsne[trc_in, 1], facecolors='red', edgecolor='red')
    axarr[1].set_aspect(1.3)
    
    gcf().canvas.draw()

axarr = display_plot()
cid = figure(5).canvas.mpl_connect('button_press_event', onclick)
