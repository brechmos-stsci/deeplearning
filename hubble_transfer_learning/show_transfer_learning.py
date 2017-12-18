import time
import pickle
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt

# Get the data directory information
from config import Configuration
c = Configuration()
data_directory = c.data_directory

def rgb2plot(data):
    mindata, maxdata = np.percentile(data, (0.01, 99))
    return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

# Choose the NN model to use
#model_name = 'resnet50'
#model_name = 'inceptionresnetv2'
model_name = 'inceptionv3'

# Load the tSNE
print('Loading tSNE pre-calculated...')
Y, labels, process_result_filename_cutout_number = pickle.load(
    open('{}/tSNE/Y_labels_{}.pck'.format(data_directory, model_name), 'rb'))

start = time.time()

plt.figure(2)
plt.clf()

while True:
    # Plot the tSNE 
    plt.figure(2)
    plt.plot(Y[:,0], Y[:,1], 'b.')
    plt.figure(2).show()
    plt.figure(2).canvas.draw()

    # Find the points closest to the tsne_coord
    tsne_coord = plt.ginput(1, timeout=0)[0]
    tsne_coord = [int(x) for x in tsne_coord]
    print('Coordinates: {}'.format(tsne_coord))
    inds = np.argsort(sum( (Y-[tsne_coord[0],tsne_coord[1]])**2, axis=1))
    print('\t Indices of closest tSNE points {}...'.format(inds[:9]))

    # Display the cutouts corresponding to the closest in tSNE space.
    plt.figure(3)
    plt.clf()
    filenames = []
    subplots = 0
    ii = 0
    while subplots < 9:

        # Load the HST filename, cutout number and center point of the cutout
        filename, cutout, middle = process_result_filename_cutout_number[inds[ii]]

        ii = ii + 1
        if filename in filenames:
            continue
        else:
            filenames.append(filename)

        # Load the hubble data    
        pf = pickle.load(open(filename, 'rb'))
        ff = glob.glob('{}/../*/{}'.format(data_directory, pf['filename'].split('/')[-1]))
        data = rgb2plot(fits.open(ff[0])[1].data)

        # Plot the hubble data in a subplot
        plt.subplot(3,3,subplots+1)
        plt.imshow(data)
        plt.xticks([]), plt.yticks([])
        plt.gray()
        plt.plot([middle[0]-112, middle[0]-112], [middle[1]-112, middle[1]+112], 'y')
        plt.plot([middle[0]+112, middle[0]+112], [middle[1]-112, middle[1]+112], 'y')
        plt.plot([middle[0]-112, middle[0]+112], [middle[1]-112, middle[1]-112], 'y')
        plt.plot([middle[0]-112, middle[0]+112], [middle[1]+112, middle[1]+112], 'y')
        clow, chigh = np.percentile(data, (1, 99))
        plt.clim((clow, chigh))
        plt.title('{} {}'.format(filename.split('/')[-1].replace('hubble_cutouts_', ''), cutout ), fontsize=10)
        subplots = subplots + 1

    plt.figure(3).canvas.draw()
    plt.figure(3).show()

    # Plot a yellow circle around the tSNE points.
    plt.figure(2)
    plt.clf()
    plt.plot(Y[:,0], Y[:,1], 'b.')
    plt.plot(Y[inds[:9],0], Y[inds[:9],1], 'yo')
    plt.figure(2).show()
    print('\tDone display...')
