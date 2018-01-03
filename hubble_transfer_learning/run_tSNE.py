import time
import pickle
from itertools import chain
import sys
import glob
import os
import json
import numpy as np
from sklearn.manifold import TSNE
from config import Configuration

#inceptionresnetv2  inceptionv3  resnet50  vgg16  vgg19
#data_directory = '/home/craig/stsci/hubble/HST/cutouts/basic/results/inceptionv3'

data_directory = sys.argv[1]

if not os.path.isdir( os.path.join(data_directory, 'tsne') ):
    os.mkdir( os.path.join(data_directory, 'tsne') )

results_directory = os.path.join(data_directory, 'tsne')

files = glob.glob(os.path.join(data_directory, '*.json'))

# Find all the unique labels
labels = []
for filename in files:
    data = json.load(open(filename, 'rt'))
    for s,v in data['results'].items():
        labels.extend([list(x.keys()) for x in v['predictions']])
labels = list(set(chain.from_iterable(labels)))
print(labels)

# Calculate the X matrix which will be passed to tSNE
print('Calculate the tSNE...')
start = time.time()
prediction_results = []
process_result_filename_cutout_number = []
ii = 0
for ii, filename in enumerate(files):

    data = json.load(open(filename, 'rt'))
    for _,v in data['results'].items():
        tt = np.zeros((len(labels),))

        for prediction in v['predictions']:
            k = list(prediction.keys())[0]  #There is only one
            pred_value = prediction[k]
            tt[labels.index(k)] = pred_value

        prediction_results.append(tt)

        process_result_filename_cutout_number.append( (v['fits_filename'], v['filename'], v['cutout_number'], v['middle']) )

X = np.array(prediction_results)

# Compute the tSNE of the data.
Y = TSNE(n_components=2).fit_transform(X)
print('\tCalculate the tSNE took {} seconds'.format(time.time() - start))

# Write the data out to the tSNE sub directory
pickle.dump((Y, labels, process_result_filename_cutout_number),
            open('{}/Y_labels.pck'.format(results_directory), 'wb'))
