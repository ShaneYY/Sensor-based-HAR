"""
Training/validating the DeepConvLSTM
"""
import lasagne
import theano
import time

import numpy as np
import cPickle as cp
import theano.tensor as T
from sliding_window import sliding_window

# from pudb import set_trace; set_trace()

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

    # The targets are cast to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

# This is identiavl to the test network
def build_cnn(input_var=None):
    net = {}
    net['inputs'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS),
                                             input_var=input_var)
    net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['inputs'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
    net['lstm1'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['shuff'], p=.5), NUM_UNITS_LSTM)
    net['lstm2'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['lstm1'], p=.5), NUM_UNITS_LSTM)
    # In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
    # to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
    net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
    net['prob'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(net['shp1'], p=.5), NUM_CLASSES,
                                            nonlinearity=lasagne.nonlinearities.softmax)
    # Tensors reshaped back to the original shape
    net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
    # Last sample in the sequence is considered
    net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
    return net['output']


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

# ############################## Main program ################################

def main(num_epochs=500):
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

    assert NB_SENSOR_CHANNELS == X_train.shape[1]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # train data
    print("Train with {0} instances in mini-batches of {1}".format(
        X_train.shape[0], BATCH_SIZE))

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(
        X_train.shape, y_train.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    print(" ..shape after reshaping: inputs {0}".format(
        X_train.shape))

    # test data
    print("Validate with {0} instances in mini-batches of {1}".format(
        X_test.shape[0], BATCH_SIZE))

    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(
        X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    print(" ..shape after reshaping: inputs {0}".format(
        X_test.shape))


    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    #params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
    #        loss, params, learning_rate=0.0001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    train_prediction_rnn = lasagne.layers.get_output(network, input_var, deterministic=False)
    train_loss_rnn = lasagne.objectives.categorical_crossentropy(train_prediction_rnn, target_var)
    train_loss_rnn = train_loss_rnn.mean()
    train_loss_rnn += .0001 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params_rnn = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.0001
    rho = 0.9
    updates_rnn = lasagne.updates.rmsprop(train_loss_rnn, params_rnn, learning_rate=lr, rho=rho)
    train_fn = theano.function([input_var, target_var], train_loss_rnn, updates=updates_rnn)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    #train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # The training+validatopn loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 1
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 1
        for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    file_1 = open('model_wb.pkl', 'wb')
    cp.dump(lasagne.layers.get_all_param_values(network), file_1)

    file_2 = open('model_w.pkl', 'w')
    cp.dump(lasagne.layers.get_all_param_values(network), file_2)

if __name__ == '__main__':
    main(5)  # training batch number


