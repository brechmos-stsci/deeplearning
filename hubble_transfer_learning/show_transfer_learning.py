import sys
import pickle
import numpy as np
import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt

class tSNEInteract:

    def __init__(self, directory):

        #directory = '/home/craig/stsci/hubble/HST/cutouts/median_filtered/results/resnet50/tsne'
        self._tsne_directory = directory
        self._cutouts_directory = directory.split('results')[0]
        self._data_directory = directory.split('cutouts')[0] + '/data'

        print('tSNE directory {}'.format(self._tsne_directory))
        print('Cutouts directory {}'.format(self._cutouts_directory))
        print('Data directory {}'.format(self._data_directory))

        # Choose the NN model to use
        self._model_name = directory.split('/')[-2]

        print('Data directory is {}'.format(self._data_directory))
        data_files = glob.glob(os.path.join(self._data_directory, '*', '*drz*fits.gz'))
        self._main_window_filename = data_files[0]

        self._main_window = None
        self._tsne_window = None
        self._sub_windows = None
        self._sub_window_filenames = None

        self._setup_figure()

        # Load the tSNE
        #  Each line of Y_tsne correpsonds to a line of process_result_filename_cutout_number
        print('Loading tSNE pre-calculated...')
        self._Y_tsne, self._labels, self._process_result_filename_cutout_number = pickle.load( open(self._tsne_directory+'/Y_labels.pck', 'rb'))

        self._display_tsne()

        self._display_window(self._main_window, os.path.join(self._data_directory, self._main_window_filename))

    def _display_tsne(self):
        """
        Display the tSNE plot in the tsne_window.

        :return:
        """
        self._tsne_window.clear()
        self._tsne_window.plot(self._Y_tsne[:,0], self._Y_tsne[:,1], 'b.')

    def _setup_figure(self):
        """
        Setup the structure of the figure (axes and text)

        :return:
        """

        plt.figure(1)
        plt.clf()

        # Two main axes
        self._tsne_window = plt.axes([0.05, 0.05, 0.4, 0.4])
        self._main_window = plt.axes([0.05, 0.55, 0.4, 0.4])

        # Nine sub axes
        self._sub_windows = []
        for row in range(3):
            for col in range(3):
                tt = plt.axes([0.5+0.17*col, 0.75-0.25*row, 0.15, 0.15])
                tt.set_xticks([])
                tt.set_yticks([])
                self._sub_windows.append(tt)

        # Register the button click
        self._cid = plt.figure(1).canvas.mpl_connect('button_press_event', self._onclick)

        # Text
        plt.figure(1).text(0.6, 0.2, 'Click with 2nd or 3rd mouse button to select image...')
        plt.figure(1).text(0.05, 0.5, 'Click in main image or tSNE plot to find similar cutouts...')
        plt.figure(1).text(0.6, 0.05, 'The tSNE data reduction calculated from data run through {}'.format(self._model_name), fontsize=8)

        # Show
        plt.figure(1).show()
        plt.figure(1).canvas.draw()

    def _rgb2plot(self, data):
        """
        Convert the input data to RGB. This is basically clipping and cropping the intensity range for display

        :param data:
        :return:
        """

        mindata, maxdata = np.percentile(data[np.isfinite(data)], (0.01, 99))
        return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

    def _display_from_tsne(self, x, y):
        """
        Display the similar cutouts based on a location in the tSNE space.

        :param x, y: Location in tSNE space
        :return:
        """

        # Find the closest 9
        inds = np.argsort(np.sum( (self._Y_tsne-np.array([x, y]))**2, axis=1))
        print(inds[:10])

        # Plot the green circles on the tsne plot
        self._display_tsne()
        self._tsne_window.plot(self._Y_tsne[inds[:9],0], self._Y_tsne[inds[:9],1], 'yo')

        # Now run through the 9 sub axes and display the image data and cutout location.
        self._sub_window_filenames = []
        for ii, axis in enumerate(self._sub_windows):
            axis.clear()

            fits_filename, filename, sliceno, middle = self._process_result_filename_cutout_number[inds[ii]]
            print('display from tsne fits: {} filename: {}'.format(fits_filename, filename))

            # So, the filename actually contains the wrong path on it so we
            # need to take it off and use the proper path.
            pf = pickle.load(open(os.path.join(self._cutouts_directory, filename), 'rb'))
            ff = list(glob.iglob('{}/**/{}'.format(self._data_directory, pf['filename'].split('/')[-1])))[0]

            print(ff)
            self._display_window(axis, ff)
            self._sub_window_filenames.append(fits_filename)

            # Draw the line
            axis.plot([middle[0]-112, middle[0]-112], [middle[1]-112, middle[1]+112], 'y')
            axis.plot([middle[0]+112, middle[0]+112], [middle[1]-112, middle[1]+112], 'y')
            axis.plot([middle[0]-112, middle[0]+112], [middle[1]-112, middle[1]-112], 'y')
            axis.plot([middle[0]-112, middle[0]+112], [middle[1]+112, middle[1]+112], 'y')

        plt.figure(1).show()
        plt.figure(1).canvas.draw()

    def _tsne_window_callback(self, x, y):
        """
        Callback function if pressed in tSNE window.

        :param x:
        :param y:
        :return:
        """
        self._display_from_tsne(x,y)

    def _main_window_callback(self, x_image, y_image):
        """
        Callback function if pressed in Main window.

        :param x_image:
        :param y_image:
        :return:
        """

        x, y = x_image, y_image

        print('xy is {} {}'.format(x, y))
        # Find the closest cutout to this click. Each line of the process_result...
        # list corresponds to a line of Y_tsne, so the one that is closest to the middle
        # and has the same filename is the line we need to lookup in the Y_tsne.
        # distances = [np.sum((np.array([x, y]) - np.array(middle))**2)
        #              for filename, cutout_no, middle in self._process_result_filename_cutout_number
        #              if self._main_window_filename in filename]
        distances = []
        for fits_filename, filename, cutout_no, middle in self._process_result_filename_cutout_number:
            if self._main_window_filename.split('/')[-1] == fits_filename.split('/')[-1]:
                d = np.sum((np.array([x, y]) - np.array(middle))**2)
                distances.append(d)
        print(distances[:9])
        inds = np.argsort(np.array(distances))

        fits_filename, filename, cutoutnumber, middle = self._process_result_filename_cutout_number[inds[0]]

        axis = self._main_window
        axis.plot([middle[0] - 112, middle[0] - 112], [middle[1] - 112, middle[1] + 112], 'y')
        axis.plot([middle[0] + 112, middle[0] + 112], [middle[1] - 112, middle[1] + 112], 'y')
        axis.plot([middle[0] - 112, middle[0] + 112], [middle[1] - 112, middle[1] - 112], 'y')
        axis.plot([middle[0] - 112, middle[0] + 112], [middle[1] + 112, middle[1] + 112], 'y')

        self._display_from_tsne(self._Y_tsne[inds[0],0], self._Y_tsne[inds[0],1])

    def _onclick(self, event):
        """
        Main callback if a mouse button is clicked in the main window.

        :param event:
        :return:
        """

        # If clicked in tSNE
        if event.inaxes == self._tsne_window:
            self._tsne_window_callback(event.xdata, event.ydata)

        # If clicked in the main window
        if event.inaxes == self._main_window:
            self._main_window_callback(event.xdata, event.ydata)

        # If it is a middle mouse button click in one of the sub-windows
        # then we will copy it to the main window.
        if event.inaxes in self._sub_windows and event.button in [2, 3]:
            # get the filename for the main window
            index = self._sub_windows.index(event.inaxes)

            # TODO: Remove Hack
            print(self._sub_window_filenames[index])
            tf = self._sub_window_filenames[index].split('/')[-1]
            print(tf)
            filename = glob.glob(self._data_directory + '/**/' + tf)[0]
            print(filename)
            filename = filename.replace(self._data_directory, '').strip('/')
            print(filename)
            #self._main_window_filename = self._sub_window_filenames[index]
            self._main_window_filename = filename

            self._display_window(self._main_window, self._main_window_filename)
            plt.figure(1).canvas.draw()

    def _display_window(self, axes, filename, extra=None):
        """
        Display the hubble data in the file at filename and display
        into the axes. It could be main axes or one of the sub windows.

        :param axes:
        :param filename:
        :return:
        """

        fn = os.path.join(self._data_directory, filename)
        data = self._rgb2plot(fits.open(fn)[1].data)

        # Plot the hubble data in a subplot
        axes.clear()
        axes.imshow(data, cmap=plt.gray(), origin='upper')
        axes.set_xticks([]) 
        axes.set_yticks([])
        clow, chigh = np.percentile(data[np.isfinite(data)], (1, 99))
        axes.get_images()[0].set_clim((clow, chigh))

        # Display the filename as the title of the axes
        tt = filename.split('/')[-1]
        if extra:
            tt += ' ' + str(extra)
        axes.set_title(tt)
        plt.figure(1).canvas.draw()

if __name__ == "__main__":

    #directory = '/home/craig/stsci/hubble/HST/cutouts/median_filtered/results/resnet50/tsne'
    directory = sys.argv[1]

    tsnei = tSNEInteract(directory)
