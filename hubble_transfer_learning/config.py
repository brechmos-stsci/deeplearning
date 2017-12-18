import os


class Configuration:

    # Default data directory
    dd = '{}/Box Sync/DeepLearning/hubble/HST/cutouts_v2/'.format(os.environ['HOME'])

    def __init__(self):
        pass

    @property
    def data_directory(self):

        data_dir = Configuration.dd

        # Check to see if the local_config.py file exists.  The local_config.py file
        # must contain a variable such as:
        #  DATA_DIRECTORY = '/home/bob/DeepLearning/hubble/HST/cutouts_v2/'
        if os.path.isfile('local_config.py'):
            print('Loading data directory from local configuration file...')
            try:
                from local_config import DATA_DIRECTORY
                data_dir = DATA_DIRECTORY
            except:
                print('Error loading from local_config file')

        return data_dir
