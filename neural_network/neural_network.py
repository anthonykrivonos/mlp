import os, sys
import random
import numpy as np
from scipy.special import expit, softmax
from enum import Enum

"""
Abstract: Multi-Layer Perceptron (MLP) neural net with customizable activation and regularization layers.
"""


class Loss(Enum):
    """
    The type of loss to use during training.
    """
    cross_entropy = 'cross_entropy'
    mse = 'mse'


class LayerType(Enum):
    """
    The type of layer to use. Activation layers are (obviously) used for activating outputs.
    Regularization layers manipulate these outputs for better learning. The input to the neural network
    is a linear activation layer so, technically, you can have zero hidden layers.
    """
    activation = 'activation'
    regularization = 'regularization'


class Regularization(Enum):
    """
    The type of regularization for your regularization layer. Note that dropout, l1, and l2 require
    parameters such as the dropout rate and regularization lambdas, respectively.
    """
    dropout = 'dropout'
    l1 = 'l1'
    l2 = 'l2'
    normalize = 'normalize'


class Activation(Enum):
    """
    The activation function to use at the given layer. Google the name of each activation type to learn
    more about it. The sigmoid, relu, linear, and softmax do not require an extra parameter.
    The parameter for leaky_relu is leak, for noisy_relu is noise, for elu is the coefficient.
    """
    sigmoid = 'sigmoid'
    relu = 'relu'
    leaky_relu = 'leaky_relu'
    noisy_relu = 'noisy_relu'
    elu = 'elu'
    linear = 'linear'
    softmax = 'softmax'


class NeuralNetwork:
    """
    Multi-layer Perceptron (MLP) neural network.
    """

    def __init__(self, input_size):
        """
        Initializes the neural network.
        :param input_size: Scalar size of vector input into the first layer (a linear layer under the hood).
        """
        self.input_size = input_size
        self.layers = [ Layer(input_size) ]
        self.weights = [ ]
        self.biases = [ ]
        self.reg_layers = { }

    def reset(self):
        """
        Re-randomize all weights and biases. Uses 1 / sqrt(input_size) as the standard deviation of the
        Gaussian the random weights and biases are sampled from.
        """
        for i in range(len(self.weights)):
            input_size = self.weights[i].shape[0]
            scale = 1 / np.sqrt(input_size)
            self.weights[i] = np.random.normal(loc=0, scale=scale, size=self.weights[i].shape)
            self.biases[i] = np.random.normal(loc=0, scale=scale, size=self.biases[i].shape)

    def add(self, layer, verbose = False):
        """
        Add a layer to the network.
        :param layer: A Layer or RegLayer object.
        :param verbose: If True, prints a description of the neural network.
        """
        if layer.type == LayerType.activation:
            # Add an activation layer
            input_size = self.layers[-1].size
            scale = 1 / np.sqrt(input_size)
            self.weights.append(np.random.normal(loc=0, scale=scale, size=(input_size, layer.size)))
            self.biases.append(np.random.normal(loc=0, scale=scale, size=(layer.size,)))
            self.layers.append(layer)
        else:
            # Add a regularization layer.
            # Regularization layers are like add-ons to activation layers, performing weight and bias
            # manipulation during weight updates.
            idx = len(self.layers) - 1
            self.reg_layers[idx] = layer
        
        # Verbosity
        vprint(verbose, "Added '%s' layer" % layer.type)
        layers = self.layers.copy()
        reg_added = 0
        for idx in self.reg_layers:
            layers.insert(idx + 1 + reg_added, self.reg_layers[idx])
            reg_added += 1
        vprint(verbose, "  Layers:  %s" % str([ l.info for l in layers ]))
        vprint(verbose, "  Weights: %s" % str([ w.shape for w in self.weights ]))
        vprint(verbose, "  Biases:  %s" % str([ b.shape for b in self.biases ]))

    def feedforward(self, X):
        """
        Feeds an input through the neural network, returning outputs and activated outputs at each layer.
        :param X: The input matric to feed in
        :return: A matrix of outputs Z and a matrix of activated outputs A as a tuple.
        """
        # Feedforward vector
        a = np.copy(X)
        # Outputs
        Z = []
        # Activated outputs
        A = [ a ]
        # Feed forward
        for weight, bias, layer in zip(self.weights, self.biases, self.layers):
            # Get output at the layer
            z = np.array(a.dot(weight) + bias, dtype = np.float)
            Z.append(z)
            # Activate the output at the layer
            a = np.array(layer.activate(z), dtype = np.float)
            A.append(a)
        return Z, A

    def backpropagate(self, y, Z, A):
        """
        Given true outputs y, outputs matrix Z, and activated outputs matrix A, returns the derivatives
        of the cost function for weights and biases. Uses MSE as cost function regardless of loss
        function used during training.
        :param y: A list of true outputs, either numerical or categorical.
        :param Z: A matrix of outputs Z.
        :param A: A matrix of activated outputs A.
        :return: Returns a list of derivatives of the cost function with respect to weights and biases, respectively.
        """
        # Instantiate a list of errors for outputs of each layer
        dCdZ = [ 0.0 ] * len(self.weights)
        # Get error in the last layer)
        dCdZ[-1] = np.array((y - A[-1])) * self.layers[-1].derivative(Z[-1])
        # Do the same for all layers in reverse order
        for i in reversed(range(len(dCdZ) - 1)):
            dCdZ[i] = dCdZ[i + 1].dot(self.weights[i + 1].T) * self.layers[i].derivative(Z[i])
        num_outputs = y.shape[0]
        # Get the derivatives of the cost function wrt weights and biases
        dCdw = []
        dCdb = []
        for i, d in enumerate(dCdZ):
            dCdw.append(A[i].T.dot(d) / float(num_outputs))
            dCdb.append(np.ones((num_outputs, 1)).T.dot(d) / float(num_outputs))
        return dCdw, dCdb

    def train(self, X, y, batch_size = 8, epochs = 100, lr = 0.0001, loss_type = Loss.mse, validation_split = 0.1, use_best_weights = True, dynamic_lr_rate = None, dynamic_lr_epochs = 5, dynamic_lr_limit = None, shuffle = False, verbose = False):
        """
        Trains the neural network.
        :param X: An matrix of inputs.
        :param y: A matrix of outputs/labels for each input.
        :param batch_size: The size of a training batch. Must be <= len(X) * (1 - validation_split). (default 8)
        :param epochs: Number of epochs to train for. (default 100)
        :param lr: The initial (or constant) learning rate. (default 0.0001)
        :param loss_type: The type of loss function to use. (default Loss.mse)
        :param validation_split: The split of the input data to be used for validation. (default 0.1)
        :param use_best_weights: Should only the best weights be saved at each epoch? (default True)
        :param dynamic_lr_rate: Rate at which to increase/decrease the lr when the validation loss plateaus. None for no dynamic lr. (default None)
        :param dynamic_lr_epochs: Number of epochs validation loss must plateau before increasing/decreasing the lr. Does nothing if dynamic_lr_rate is None. (default 5)
        :param dynamic_lr_limit: Lower/upper bound on lr if dynamic_lr_rate is not None. (default None)
        :param shuffle: Shuffle the training data before training and creating the validation set? (default False)
        :param verbose: Should the result of each epoch be printed? (default False)
        :return:
        """
        assert(batch_size <= len(X) * (1 - validation_split))
        assert(epochs <= 999999)

        # Shuffle if required
        if shuffle:
            data = list(zip(X, y))
            random.shuffle(data)
            X, y = zip(*data)
        
        # Reshape input and output if flat
        if validation_split:
            lim = int((1 - validation_split) * len(y))
            X_val = np.array(X[lim:], dtype = np.float)
            y_val = np.array(y[lim:], dtype = np.float)
            X = np.array(X[:lim], dtype = np.float)
            y = np.array(y[:lim], dtype = np.float)
            m = y_val.shape[0]
        else:
            X = np.array(X, dtype = np.float)
            y = np.array(y, dtype = np.float)

        best_acc = 0
        best_loss = sys.maxsize
        best_weights = self.weights.copy()
        best_biases = self.biases.copy()

        # Number of epochs in a row loss hasn't changed
        loss_stagnant_epochs = 0
        last_loss = None

        def get_loss(output, true_output):
            size = len(output)
            if loss_type == Loss.mse:
                loss = np.sum((true_output - output)**2) / size
            else:
                eps = 1e-5
                shift = np.min(output)
                shifted_output = output + eps - shift
                loss = -np.sum(true_output * np.log(shifted_output)) / size
            return loss

        # Loop for number of epochs
        for e in range(epochs):
            i = 0
            loss = sys.maxsize

            # If we keep track of the highest loss across all epochs, we have a 'worst case' measure for epoch weights
            highest_epoch_loss = 0
            
            while i < len(y):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                Z, A = self.feedforward(X_batch)
                dCdw, dCdb = self.backpropagate(y_batch, Z, A)

                # Regularize function
                regularize = lambda j, x: self.reg_layers[j].regularize(x) if j in self.reg_layers else x

                # Update weights and biases
                self.weights = [ regularize(j, w + lr * d) for w, d, j in zip(self.weights, dCdw, range(len(self.weights))) ]
                self.biases = [ regularize(j, b + lr * d) for b, d, j in zip(self.biases, dCdb, range(len(self.biases))) ]

                # Get loss
                loss = get_loss(A[-1], y_batch)
                
                # Keep track of highest loss
                highest_epoch_loss = max(highest_epoch_loss, loss)

                # Verbosity
                vprint(verbose, "Epoch %6d (loss: %1.20f) %s%s" % (e + 1, loss, progress_bar(i / len(y)), " " * 20), end='\r' if i < len(y) - 1 else '')

                i += batch_size

            if validation_split:
                # Calculate validation accuracy
                y_pred = self.classify(X_val)
                acc = NeuralNetwork.get_accuracy(y_pred, y_val)

            # Get the best weights
            if use_best_weights:
                if validation_split:
                    # Rollback the weights and biases to their best values based on validation loss and validation accuracy
                    if highest_epoch_loss < best_loss or (highest_epoch_loss == best_loss and acc > best_acc):
                        best_weights = self.weights.copy()
                        best_biases = self.biases.copy()
                        best_loss = highest_epoch_loss
                        best_acc = acc
                    else:
                        self.weights = best_weights
                        self.biases = best_biases
                else:
                    # Rollback the weights and biases to their best values based on validation loss only
                    if highest_epoch_loss < best_loss:
                        best_weights = self.weights.copy()
                        best_biases = self.biases.copy()
                        best_loss = highest_epoch_loss
                    else:
                        self.weights = best_weights
                        self.biases = best_biases
            else:
                best_acc = acc
                best_loss = highest_epoch_loss

            # Verbosity
            if validation_split:
                vprint(verbose, "Epoch %6d (acc: %1.12f, loss: %1.20f) %s%s" % (e + 1, best_acc, best_loss, progress_bar(1), " " * 10))
            else:
                vprint(verbose, "Epoch %6d (loss: %1.20f) %s%s" % (e + 1, best_loss, progress_bar(1), " " * 10))

            # Dynamically reduce learning rate if it plateaus
            if dynamic_lr_rate is not None and e != epochs - 1:
                if use_best_weights:
                    highest_epoch_loss = best_loss
                if highest_epoch_loss == last_loss:
                    loss_stagnant_epochs += 1
                    if loss_stagnant_epochs >= dynamic_lr_epochs:
                        # Ensure dynamic_lr_rate doesn't exceed its limit
                        if dynamic_lr_limit is not None and ((dynamic_lr_rate > 1 and lr * abs(dynamic_lr_rate) > dynamic_lr_limit) or (dynamic_lr_rate <= 1 and lr * abs(dynamic_lr_rate) < dynamic_lr_limit)):
                            dynamic_lr_rate = None
                            lr = dynamic_lr_limit
                            continue
                        # Reduce or increase learning rate by factor
                        loss_stagnant_epochs = 1
                        lr *= abs(dynamic_lr_rate)
                        vprint(verbose, " - lr %s to %1.12f" % ("reduced" if abs(dynamic_lr_rate) < 1 else "increased", lr))
                else:
                    loss_stagnant_epochs = 1
                last_loss = highest_epoch_loss

    def classify(self, X):
        """
        Classify every data point in X with a label.
        :param X: A list of inputs.
        :return: A list of outputs (either numerical or one-hot for categorical).
        """
        Z, A = self.feedforward(X)
        output = A[-1]
        return output

    def save_weights(self, filename):
        """
        Save the NN's current weights to filename.npy.
        :param filename: The base name of the file to save to.
        """
        data_to_save = [ self.weights, self.biases ]
        np.save(filename, data_to_save)
    
    def load_weights(self, filename):
        """
        Loads the weights from the given file into the NN. The weights and biases must be the
        exact same dimensions as the current NN. Otherwise, the behavior is undefined.
        :param filename: The base name of the file to load.
        """
        if '.npy' not in filename:
            filename += '.npy'
        data_to_load = np.load(filename, allow_pickle=True)
        self.weights, self.biases = data_to_load[0], data_to_load[1]

    @staticmethod
    def one_hot_normalize(y_one_hot):
        """
        Given a list of n-dimensional predictions, turns each prediction into a one-hot value where
        the prediction at index argmin = 1 and all other components are 0. Must only be used for
        categorical data.
        :param y_one_hot: A list of inputs to normalize.
        :return: A list of one-hot predictions.
        """
        for i, y in enumerate(y_one_hot):
            y_new = [ 0 ] * y_one_hot.shape[1]
            y_new[np.argmax(y)] = 1
            y_one_hot[i] = y_new
        return y_one_hot

    @staticmethod
    def one_hot_encode(y_categorical, label_mapping=None):
        """
        Converts categorical labels into one-hot labels.
        :param y_categorical: A list of categorical labels.
        :param label_mapping: The desired mapping. For instance, if the mapping is [ 'red', 'blue', 'green' ],
        then all labels that are 'red' will be returned as [ 1, 0, 0 ]. If None, this function will
        generate a mapping from all the labels it sees. (default None)
        :return: The array of one-hot labels and the label mapping used, as a tuple.
        """
        if label_mapping is None:
            label_count = 0
            label_mapping = []
            for y in y_categorical:
                if y not in label_mapping:
                    label_mapping.append(y)
                    label_count += 1
        else:
            label_count = len(label_mapping)
        y_one_hot = [ [ 0 ] * label_count for _ in range(len(y_categorical)) ]
        for i, y in enumerate(y_categorical):
            hot_label = label_mapping.index(y)
            y_one_hot[i][hot_label] = 1
        return np.array(y_one_hot), label_mapping

    @staticmethod
    def one_hot_decode(y_one_hot, label_mapping):
        """
        Converts one-hot labels into categorical labels.
        :param y_one_hot: A list of one-hot labels.
        :param label_mapping: The desired mapping. For instance, if the mapping is [ 'red', 'blue', 'green' ],
        then all labels that are 'red' will be returned as [ 1, 0, 0 ]. If left blank, this function will
        generate a mapping from all the labels it sees.
        :return: The array of categorical labels.
        """
        y_categorical = []
        for y in y_one_hot:
            hot_label = np.argmax(y)
            categorical_label = label_mapping[hot_label]
            y_categorical.append(categorical_label)
        return y_categorical

    @staticmethod
    def get_accuracy(y_pred, y_true):
        """
        Calculates the prediction accuracy given predicted labels and true labels of the same dimension.
        :param y_pred: A list of numerical, categorical, or one-hot predicted labels.
        :param y_true: A list of numerical, categorical, or one-hot true labels.
        :return: A ratio from 0 to 1 of the number of correct labelings. 1 - get_accuracy(...) provides the error rate.
        """
        n = len(y_pred)
        assert(n > 0)
        is_one_hot = np.array(y_pred).ndim > 1
        if is_one_hot:
            # One hot encode predictions
            y_pred = NeuralNetwork.one_hot_normalize(y_pred)
        return np.sum([ int(np.equal(y_true[i], y_pred[i]).all()) if is_one_hot else int(y_true[i] == y_pred[i]) for i in range(n) ]) / n


class RegLayer:
    """
    A regularization layer, which provides methods for manipulating weights during learning.
    """

    def __init__(self, regularization = Regularization.l2, regularization_parameter = None):
        """
        Initializes a new regularization layer.
        :param regularization: The type of regularization. See Regularization. (default Regularization.l2)
        :param regularization_parameter: The parameter for regularization. If l1 or l2 is used, this is the
        regularization lambda. If dropout is used, this is the dropout rate. Normalization does not use a
        parameter (default None).
        """
        self.regularization = regularization
        self.regularization_parameter = regularization_parameter
        self.type = LayerType.regularization
        self.info = "{ %.2f, %s }" % (regularization_parameter, regularization) if regularization_parameter else "{ %s }" % regularization

    def regularize(self, x):
        """
        Performs the regularization on the input. In practice, x is either the weights or biases vector.
        :param x: The input matrix.
        :return: The output of the regularization applied on the input matrix.
        """
        if self.regularization == Regularization.dropout:
            assert(self.regularization_parameter is not None)
            # Get the dropout rate as a positive float
            rate = abs(float(self.regularization_parameter))
            # Sample a 0/1 matrix from a binomial with the given rate
            dropout_mat = np.random.binomial(1, 1 - rate, size=x.shape)
            # Multiply by the input to drop out the random components
            x *= dropout_mat
        elif self.regularization == Regularization.l1:
            assert(self.regularization_parameter is not None)
            # Get the lambda as a positive float
            λ = abs(float(self.regularization_parameter))
            # x <- x - (x matrix regularized into -1/1s divided by lambda)
            x -= np.vectorize(lambda χ: 1 if χ >= 0 else -1)(x) / λ
        elif self.regularization == Regularization.l2:
            assert(self.regularization_parameter is not None)
            # Get the lambda as a positive float
            λ = abs(float(self.regularization_parameter))
            # x <- x - (x matrix divided by lambda)
            x -= x / λ
        elif self.regularization == Regularization.normalize:
            # Make all x components positive
            x += np.ones(x.shape) * np.abs(np.min(x))
            # Normalize components
            x /= np.sum(x)
        return x


class Layer:

    def __init__(self, size, activation = Activation.linear, activation_parameter = None):
        """
        Initializes a new activation layer.
        :param size: The scalar input size of the layer.
        :param activation: The type of activation used. (default Activation.linear)
        :param activation_parameter: The parameter required for the given activation. (default None)
        """
        self.size = size
        self.activation = activation
        self.activation_parameter = activation_parameter
        self.type = LayerType.activation
        self.info = "{ %s, %s }" % (size, activation.name)

    def activate(self, x):
        """
        Call the activation function on the input.
        :param x: An input matrix.
        :return: The output of the activation function applied on the input matrix.
        """
        if self.activation == Activation.sigmoid:
            # Shift x around the maximum so not to overflow
            shifted_x = x - np.max(x, axis = 0)
            # Get the sigmoid function output for x
            return expit(shifted_x)
        elif self.activation == Activation.relu:
            # Make all negative values 0
            return np.vectorize(lambda χ: χ if χ > 0 else 0)(x)
        elif self.activation == Activation.leaky_relu:
            assert(self.activation_parameter is not None)
            # Multiply all negative values by the leak. Usually < 1.
            leak = self.activation_parameter
            return np.vectorize(lambda χ: χ if χ > 0 else χ * leak)(x)
        elif self.activation == Activation.noisy_relu:
            assert(self.activation_parameter is not None)
            # Add Gaussian noise to positive components, otherwise set to 0
            std_dev = self.activation_parameter
            get_noise = lambda: np.random.normal(scale = std_dev)
            # Noisy RELU helper
            def noisy(χ):
                noise = get_noise()
                noisy_χ = χ + noise
                return noisy_χ if noisy_χ > 0 else 0
            return np.vectorize(noisy)(x)
        elif self.activation == Activation.elu:
            assert(self.activation_parameter is not None)
            # Activate linearly if positive, otherwise activate by a * (e^(x_i) - 1)
            a = self.activation_parameter
            return np.vectorize(lambda χ: χ if χ > 0 else a * (np.exp(χ) - 1))(x)
        elif self.activation == Activation.softmax:
            assert(len(x) > 0)
            # Shift x around the maximum so not to overflow
            shifted_x = x - np.max(x, axis = 0)
            # Get the softmax function output for x
            return softmax(shifted_x)
        return x

    def derivative(self, x):
        """
        Call the derivative of the activation function on the input for backpropogation.
        :param x: An input matrix.
        :return: The output of the derivative of the activation function applied on the input matrix.
        """
        if self.activation == Activation.sigmoid:
            assert(len(x) > 0)
            # Calculate the sigmoid first (needed in the derivative)
            sig = self.activate(x)
            # Return the derivative of the sigmoid
            return sig * (np.ones(x.shape) - sig)
        elif self.activation == Activation.relu:
            # Piecewise derivative of RELU
            return np.vectorize(lambda χ: 1 if χ > 0 else 0)(x)
        elif self.activation == Activation.leaky_relu:
            assert(self.activation_parameter is not None)
            # Piecewise derivative of leaky RELU
            leak = self.activation_parameter
            return np.vectorize(lambda χ: 1 if χ > 0 else leak)(x)
        elif self.activation == Activation.noisy_relu:
            assert(self.activation_parameter is not None)
            std_dev = self.activation_parameter
            # Piecewise derivative of noisy RELU
            get_noise = lambda: np.random.normal(scale = std_dev)
            return np.vectorize(lambda χ: get_noise() if χ > 0 else 0)(x)
        elif self.activation == Activation.elu:
            assert(self.activation_parameter is not None)
            # Piecewise derivative of noisy ELU
            a = self.activation_parameter
            return np.vectorize(lambda χ: 1 if χ > 0 else a * np.exp(χ))(x)
        elif self.activation == Activation.softmax:
            assert(len(x) > 0)
            # Calculate the softmax first (needed in the derivative)
            softmax = self.activate(x)
            # Return the derivative of the softmax
            return softmax * (np.ones(x.shape) - softmax)
        return x


"""
Utility Functions
"""

def progress_bar(perc, width = 30):
    """
    Gets a progress bar for printing.
    :param perc: The percent completed.
    :param width: The entire width of the bar.
    :return: The progress bar string.
    """
    assert(width > 10)
    width -= 3
    prog = int(perc * width)
    bar = "[" + "=" * prog + (">" if perc < 1 else "=") + "." * (width - prog) + "]"
    return bar


def vprint(verbose, *args, **kwargs):
    """
    Conditional printing function.
    :param verbose: If true, performs the print.
    :param args: args to print(...).
    :param kwargs: kwargs to print(...).
    """
    if verbose:
        print(*args, **kwargs)
