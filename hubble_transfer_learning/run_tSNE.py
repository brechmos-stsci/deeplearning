import time
import pickle
from itertools import chain
import sys
import numpy as np
from sklearn.manifold import TSNE
from config import Configuration

from blitzdb import Document
from blitzdb import FileBackend

# Load the configuration file information
c = Configuration()
data_directory = c.data_directory

# Setup the storage mechanism
backend = FileBackend("{}/prediction_database".format(data_directory))


class DataDescription(Document):
    pass


class ProcessDescription(Document):
    pass


class ProcessResult(Document):
    pass


print('Data Descriptions')
dds = backend.filter(DataDescription, {})
for dd in dds:
    print('\t', dd.name)

# Run through several different models in order to create the tSNE.
for model_name in ['resnet50', 'vgg16', 'vgg19', 'inceptionv3', 'inceptionresnetv2']:

    #  Data description part
    try:
        data_description = backend.get(DataDescription,{'name' : 'hubblecutouts'})
    except DataDescription.DoesNotExist:
        print("Data desription doesn't exist")
        sys.exit(0)
    except DataDescription.MultipleDocumentsReturned:
        #more than one 'Charlie' in the database
        pass
    print('Data description is {}'.format(data_description.name))

    #  Process description part
    try:
        process_description = backend.get(ProcessDescription,{'name' : model_name})
    except ProcessDescription.DoesNotExist:
        print("Process desription doesn't exist")
        sys.exit(0)
    except ProcessDescription.MultipleDocumentsReturned:
        pass
    print('Process description is {}'.format(process_description.name))

    # Grab the process results given the data description and process description information
    process_results = backend.filter(ProcessResult,{'data_description.pk' : data_description.pk, 'process_description.pk': process_description.pk})
    N = len(process_results)

    # Find all the unique labels
    labels = []
    for pr in process_results:
        labels.extend([list(x.keys()) for x in pr.predictions])
    labels = list(set(chain.from_iterable(labels)))
    print(labels)

    # Calculate the X matrix which will be passed to tSNE
    print('Calculate the tSNE...')
    start = time.time()
    X = np.zeros((N, len(labels)))
    process_result_filename_cutout_number = []
    ii = 0
    for x in process_results:
        for jj in x.predictions:
            k = list(jj.keys())[0]
            v = jj[k]
            X[ii,labels.index(k)] = v
        ii = ii + 1
        process_result_filename_cutout_number.append( (x.filename, x.cutout_number, x.middle) )
    print(X)

    # Compute the tSNE of the data.
    Y = TSNE(n_components=2).fit_transform(X)
    print('\tCalculate the tSNE took {} seconds'.format(time.time() - start))

    # Write the data out to the tSNE sub directory
    pickle.dump((Y,labels, process_result_filename_cutout_number),
                open('{}/tSNE/Y_labels_{}.pck'.format(data_directory, model_name), 'wb'))
