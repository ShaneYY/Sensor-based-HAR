import lasagne
import theano
import time

import numpy as np
import cPickle as cp
import theano.tensor as T
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128


def load_dataset(filename):

    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

assert NB_SENSOR_CHANNELS == X_train.shape[1]
def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

# Sensor data is segmented using a sliding window mechanism
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

# Data is reshaped since the input of the network is a 4 dimension tensor
X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))


net = {}
net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'], NUM_UNITS_LSTM)
net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'], NUM_UNITS_LSTM)
# In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
# to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
net['prob'] = lasagne.layers.DenseLayer(net['shp1'],NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
# Tensors reshaped back to the original shape
net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
# Last sample in the sequence is considered
net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)


# The model is populated with the weights of the pretrained network
all_params_values = cp.load(open('model_w.pkl'))
lasagne.layers.set_all_param_values(net['output'], all_params_values)
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(net['output'], param_values)

# Compilation of theano functions
# Obtaining the probability distribution over classes
test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
# Returning the predicted output for the given minibatch
test_fn =  theano.function([ net['input'].input_var], [T.argmax(test_prediction, axis=1)])


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Classification of the testing data
print("Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], BATCH_SIZE))
test_pred = np.empty((0))
test_true = np.empty((0))
start_time = time.time()
for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
    inputs, targets = batch
    y_pred, = test_fn(inputs)
    test_pred = np.append(test_pred, y_pred, axis=0)
    test_true = np.append(test_true, targets, axis=0)


# Results presentation
print("||Results||")
print("\tTook {:.3f}s.".format( time.time() - start_time))
import sklearn.metrics as metrics
print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(test_true, test_pred, average='weighted')))