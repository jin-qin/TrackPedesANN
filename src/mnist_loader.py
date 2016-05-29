import numpy as np
import gzip
import pickle

def load_MNIST_data(pPath):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    """
    f = gzip.open(pPath + '/mnist.pkl.gz', 'rb');
    tmn, vmn, temn = pickle.load(f, encoding='latin1');
    f.close();
    tmn_x = [np.reshape(x, (784)) for x in tmn[0]];   #traning set data
    tmn_y = [y for y in tmn[1]];    #training set label
    tmn_x = np.asarray(tmn_x);
    tmn_y = np.asarray(tmn_y);
    vmn_x = [np.reshape(x, (784)) for x in vmn[0]]; #validation set data
    vmn_y = [y for y in vmn[1]];   #validation set label
    vmn_x = np.asarray(vmn_x);
    vmn_y = np.asarray(vmn_y);
    temn_x = [np.reshape(x, (784)) for x in temn[0]];  #test set data
    temn_y = [y for y in temn[1]];   #test set label
    temn_x = np.asarray(temn_x);
    temn_y = np.asarray(temn_y);
    return (tmn_x, tmn_y, vmn_x, vmn_y, temn_x, temn_y);