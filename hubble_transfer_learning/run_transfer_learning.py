import pickle
import time
import numpy as np
import glob
from collections import OrderedDict
from blitzdb import Document
from blitzdb import FileBackend
from config import Configuration


def gray2rgb(data):
    """
    Convert 2D data set to 3D gray scale

    :param data:
    :return:
    """
    data_out = np.zeros((224, 224, 3, data.shape[0]))
    data_out[:, :, 0] = data.transpose((1, 2, 0))
    data_out[:, :, 1] = data.transpose((1, 2, 0))
    data_out[:, :, 2] = data.transpose((1, 2, 0))

    return data_out


def rgb2plot(data):
    """
    Convert the 3 planes of data to a proper format for RGB imshow.

    :param data:
    :return:
    """
    mindata, maxdata = np.prctile(data, (0.01, 99))
    return np.clip((data - mindata) / (maxdata - mindata) * 255, 0, 255).astype(np.uint8)


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


#  Save the data description part to the database
try:
    data_description = backend.get(DataDescription, {'name': 'hubblecutouts'})
except DataDescription.DoesNotExist:
    doc = {'name': 'hubblecutouts', 'description': 'hubble cutouts from Josh'}
    data_description = DataDescription(doc)
    backend.save(data_description) 
    backend.commit()
except DataDescription.MultipleDocumentsReturned:
    # more than one in the database, ignore for now
    pass

# Now run through all the models
for model_name in ['resnet50', 'vgg16', 'vgg19', 'inceptionv3', 'inceptionresnetv2']:
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
    
    #  Save the process description part
    try:
        process_description = backend.get(ProcessDescription, {'name': model_name})
    except ProcessDescription.DoesNotExist:
        doc = {'name': model_name, 'description': '{} with imagenet'.format(model_name)}
        process_description = ProcessDescription(doc)
        backend.save(process_description) 
        backend.commit()
    except ProcessDescription.MultipleDocumentsReturned:
        pass
    print(process_description.name)
    
    # ---------------------------------------------------------------------
    # Load and pre-processing
    # ---------------------------------------------------------------------

    files = glob.glob('{}/hubble_cutouts_*.pck'.format(data_directory))
    
    # Load in the data
    allpredictions = OrderedDict()
    for filename in files:

        # Load up the cutouts
        print('processing {}'.format(filename.split('/')[-1]))
        data_orig = pickle.load(open(filename, 'rb'))

        N = len(data_orig['cutouts'])
    
        # ---------------------------------------------------------------------
        # Now processing
        # ---------------------------------------------------------------------
    
        # Calculate the predicitons for all data
    
        start = time.time()
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
                'data_description': data_description, 
                'process_description': process_description, 
                'filename': filename,
                'middle': data_orig['cutouts'][ii]['middle'],
                'cutout_number': ii,
                'predictions': [{tt[1]: np.float64(tt[2])} for tt in predictions]
                }
            process_result = ProcessResult(doc)
            backend.save(process_result) 
            backend.commit()
    
        print('\r\tCalculate the predictions took {} seconds'.format(time.time() - start))
