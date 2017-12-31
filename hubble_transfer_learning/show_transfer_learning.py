import time
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
        self._tsne_window.clear()
        self._tsne_window.plot(self._Y_tsne[:,0], self._Y_tsne[:,1], 'b.')

    def _setup_figure(self):
        plt.figure(1)
        plt.clf()

        self._tsne_window = plt.axes([0.05, 0.05, 0.4, 0.4])
        self._main_window = plt.axes([0.05, 0.55, 0.4, 0.4])

        self._sub_windows = []
        for row in range(3):
            for col in range(3):
                tt = plt.axes([0.5+0.17*col, 0.05+0.25*row, 0.15, 0.15])
                tt.set_xticks([])
                tt.set_yticks([])
                self._sub_windows.append(tt)

        self._cid = plt.figure(1).canvas.mpl_connect('button_press_event', self._onclick)

        plt.figure(1).show()
        plt.figure(1).canvas.draw()

    def _rgb2plot(self, data):

        mindata, maxdata = np.percentile(data, (0.01, 99))

        return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

    def _display_from_tsne(self, x, y):
        print('display_from_tsne {} {}'.format(x, y))

        # Find the closest 9
        inds = np.argsort(np.sum( (self._Y_tsne-np.array([x, y]))**2, axis=1))
        print(inds[:10])

        # Plot the green circles on the tsne plot
        self._display_tsne()
        self._tsne_window.plot(self._Y_tsne[inds[:9],0], self._Y_tsne[inds[:9],1], 'go')

        self._sub_window_filenames = []
        for ii, axis in enumerate(self._sub_windows):
            axis.clear()

            filename, sliceno, middle = self._process_result_filename_cutout_number[inds[ii]]
            print('display from tsne {}'.format(filename))

            # So, the filename actually contains the wrong path on it so we
            # need to take it off and use the proper path.
            pf = pickle.load(open(os.path.join(self._cutouts_directory, filename), 'rb'))
            ff = list(glob.iglob('{}/**/{}'.format(self._data_directory, pf['filename'].split('/')[-1])))[0]

            print(ff)
            self._display_window(axis, ff)
            self._sub_window_filenames.append(filename)

            # Draw the line
            axis.plot([middle[0]-112, middle[0]-112], [middle[1]-112, middle[1]+112], 'y')
            axis.plot([middle[0]+112, middle[0]+112], [middle[1]-112, middle[1]+112], 'y')
            axis.plot([middle[0]-112, middle[0]+112], [middle[1]-112, middle[1]-112], 'y')
            axis.plot([middle[0]-112, middle[0]+112], [middle[1]+112, middle[1]+112], 'y')

        plt.figure(1).show()
        plt.figure(1).canvas.draw()

    def _tsne_window_callback(self, x, y):
        self._display_from_tsne(x,y)

    def _main_window_callback(self, x_image, y_image):

        x, y = x_image, y_image

        print('xy is {} {}'.format(x, y))
        # Find the closest cutout to this click. Each line of the process_result...
        # list corresponds to a line of Y_tsne, so the one that is closest to the middle
        # and has the same filename is the line we need to lookup in the Y_tsne.
        # distances = [np.sum((np.array([x, y]) - np.array(middle))**2)
        #              for filename, cutout_no, middle in self._process_result_filename_cutout_number
        #              if self._main_window_filename in filename]
        distances = []
        for filename, cutout_no, middle in self._process_result_filename_cutout_number:
            print(self._main_window_filename, filename)
            if self._main_window_filename == filename:
                d = np.sum((np.array([x, y]) - np.array(middle))**2)
                distances.append(d)
        inds = np.argsort(np.array(distances))

        filename, cutoutnumber, middle = self._process_result_filename_cutout_number[inds[0]]

        axis = self._main_window
        axis.plot([middle[0] - 112, middle[0] - 112], [middle[1] - 112, middle[1] + 112], 'y')
        axis.plot([middle[0] + 112, middle[0] + 112], [middle[1] - 112, middle[1] + 112], 'y')
        axis.plot([middle[0] - 112, middle[0] + 112], [middle[1] - 112, middle[1] - 112], 'y')
        axis.plot([middle[0] - 112, middle[0] + 112], [middle[1] + 112, middle[1] + 112], 'y')

        self._display_from_tsne(self._Y_tsne[inds[0],0], self._Y_tsne[inds[0],1])

    def _onclick(self, event):

        # If tSNE
        if event.inaxes == self._tsne_window:
            self._tsne_window_callback(event.xdata, event.ydata)

        if event.inaxes == self._main_window:
            self._main_window_callback(event.xdata, event.ydata)

        # If it is a middle mouse button click in one of the sub-windows
        # then we will copy it to the main window.
        if event.inaxes in self._sub_windows and event.button == 2:
            print('button 2')
            # get the filename for the main window
            index = self._sub_windows.index(event.inaxes)
            self._main_window_filename = self._sub_window_filenames[index]

            self._display_window(self._main_window, self._main_window_filename)
            plt.figure(1).canvas.draw()

    def _display_window(self, axes, filename):
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
        clow, chigh = np.percentile(data, (1, 99))
        axes.get_images()[0].set_clim((clow, chigh))
        axes.set_title(filename.split('_')[-1])


if __name__ == "__main__":

    #directory = '/home/craig/stsci/hubble/HST/cutouts/median_filtered/results/resnet50/tsne'
    directory = sys.argv[1]

    tsnei = tSNEInteract(directory)

