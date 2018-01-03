import pickle
import time
import sys
import os
import json
import numpy as np
import glob
from collections import OrderedDict
from config import Configuration

# run -i run_transfer_learning.py /Users/crjones/christmas/hubble/HST/cutouts/basic resnet50
data_directory = sys.argv[1]
model_name = sys.argv[2]
results_directory = os.path.join(data_directory, 'results')

def gray2rgb(data):
    """
    Convert 2D data set to 3D gray scale

    :param data:
    :return:
    """
    data_out = np.zeros((224, 224, 3))
    data_out[:, :, 0] = data
    data_out[:, :, 1] = data
    data_out[:, :, 2] = data

    return data_out


def rgb2plot(data):
    """
    Convert the 3 planes of data to a proper format for RGB imshow.

    :param data:
    :return:
    """
    mindata, maxdata = np.prctile(data, (0.01, 99))
    return np.clip((data - mindata) / (maxdata - mindata) * 255, 0, 255).astype(np.uint8)


data_description = {'name': 'acs_halpha', 'description': 'ACS Halpha from Josh'}

# Now run through all the models
if model_name == 'resnet50':
    # Add the process description information
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    model = ResNet50(weights='imagenet')
    model_name = 'resnet50'
elif model_name == 'vgg16':
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    model = VGG16(weights='imagenet')
    model_name = 'vgg16'
elif model_name == 'vgg19':
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input, decode_predictions
    model = VGG19(weights='imagenet')
    model_name = 'vgg19'
elif model_name == 'inceptionv3':
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input, decode_predictions
    model = InceptionV3(weights='imagenet')
    model_name = 'inceptionv3'
elif model_name == 'inceptionresnetv2':
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
    model = InceptionResNetV2(weights='imagenet')
    model_name = 'inceptionresnetv2'

process_description = {'name': model_name, 'description': '{} with imagenet'.format(model_name)}
print(process_description)

result = {}
result['data_descrption'] = data_description
result['process_descrption'] = process_description

# ---------------------------------------------------------------------
# Load and pre-processing
# ---------------------------------------------------------------------

files = glob.glob('{}/*_cutouts_*.pck'.format(data_directory))

# Load in the data
allpredictions = {}
for filename in files:

    # Load up the cutouts
    print('processing {}'.format(filename.split('/')[-1]))
    data_orig = pickle.load(open(filename, 'rb'))

    N = len(data_orig['cutouts'])

    # ---------------------------------------------------------------------
    # Now processing
    # ---------------------------------------------------------------------

    # Calculate the predicitons for all data
    path, fname = os.path.split(filename)

    start = time.time()
    file_results = {}
    for ii in range(N):
        print('\r\t{} of {}'.format(ii, N), end='')

        # Set the data into the expected format
        x = gray2rgb(data_orig['cutouts'][ii]['data'])
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            predictions = decode_predictions(preds, top=100)[0]
        else:
            predictions = [('test', 'beaver', 0.0000000000001),]

        # Save the results in the database.
        doc = {
            'filename': fname,
            'fits_filename': data_orig['filename'],
            'middle': data_orig['cutouts'][ii]['middle'],
            'cutout_number': ii,
            'predictions': [{tt[1]: np.float64(tt[2])} for tt in predictions]
            }
        file_results[ii] = doc

    result['results'] = file_results

    if not os.path.isdir(os.path.join(results_directory, model_name)):
        os.mkdir(os.path.join(results_directory, model_name))

    filename_results = os.path.join(results_directory, model_name, fname.replace('pck', 'json'))
    print('\tSaving to {}'.format(filename_results))
    json.dump(result, open(filename_results, 'wt'))

    print('\r\tCalculate the predictions took {} seconds'.format(time.time() - start))
