import numpy.matlib as np

class Layer:
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out

        # weights is an nxm matrix, where
        #   n -- size of output layer
        #   m -- size of input layer
        self.weights = np.rand(size_out, size_in)*2-1

        self.biases = np.rand(size_out,1)*2-1


class NeuralNet:
    '''
    Stores the data of a neural net (mlp). This includes all layer weight/bias matrices, and activation function/derivative. Includes some convenience functions for calculating output, etc.
    '''
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, layer_size, activation, activation_derivative):
        '''
        Create a neural net with input, output, and hidden layers.
        '''
        if num_inputs < 1:
            raise ValueError('Net must have at least 1 input')
        elif num_outputs < 1:
            raise ValueError('Net must have at least one output')
        elif num_hidden_layers < 0:
            raise ValueError('Net must have at least 0 hidden layers')
        elif layer_size < 1:
            raise ValueError('Hidden layers must have at least one neuron')
        
        self.layers = [Layer(num_inputs, layer_size)]

        for _ in range(num_hidden_layers):
            self.layers.append(Layer(layer_size, layer_size))
        
        self.layers.append(Layer(layer_size, num_outputs))

        self.activation = activation
        self.activation_derivative = activation_derivative

    def evaluate(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = self.evaluate_layer(layer, outputs)
        return outputs

    def evaluate_layer(self, layer, inputs):
        return self.activation(self.evaluate_layer_without_activation(layer, inputs))

    def evaluate_layer_without_activation(self, layer, inputs):
        if type(inputs) is not np.matrix:
            raise TypeError('inputs must be numpy matrix')
        elif inputs.dtype != np.float64:
            raise TypeError('inputs must be a float type')
        elif inputs.shape != (layer.size_in, 1):
            raise ValueError('inputs has wrong shape')
        
        return layer.weights*inputs + layer.biases
    
    def num_inputs(self):
        return self.layers[0].size_in
    def num_outputs(self):
        return self.layers[-1].size_out