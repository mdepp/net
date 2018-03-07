import numpy.matlib as np

class GradientLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def __repr__(self):
        return 'Grad: weights, biases\nw={0}\nb={1}\n'.format(self.weights, self.biases)
    
    def __add__(self, other):
        return GradientLayer(self.weights + other.weights, self.biases + other.biases)
    def __sub__(self, other):
        return GradientLayer(self.weights - other.weights, self.biases - other.biases)

    def __iadd__(self, other):
        self.weights += other.weights
        self.biases += other.biases
        return self
    
    def __mul__(self, other):
        if type(other) is GradientLayer:
            return GradientLayer(np.multiply(self.weights, other.weights), np.multiply(self.biases, other.biases))
        else:
            return GradientLayer(np.multiply(self.weights, other), np.multiply(self.biases, other))
    
    def __truediv__(self, other):
        if type(other) is GradientLayer:
            return GradientLayer(np.divide(self.weights, other.weights), np.divide(self.biases, other.biases))
        else:
            return GradientLayer(np.divide(self.weights, other), np.divide(self.biases, other))
    
    def __imul__(self, other):
        '''
        >>> a = GradientComponent(np.matrix([-1, 1]), np.matrix([1, 1]))
        >>> a *= GradientComponent(np.matrix([-1, -1]), np.matrix([-1, -1]))
        >>> a
        Grad: weights, biases
        w=[[ 1 -1]]
        b=[[-1 -1]]
        <BLANKLINE>
        '''
        
        np.multiply(self.weights, other.weights, out=self.weights)
        np.multiply(self.biases, other.biases, out=self.biases)
        return self


class NumericApproximator:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def get_gradient(self, net, inputs, expected_outputs):
        '''
        Returns gradient as an np array of GradientLayers
        '''
        if type(expected_outputs) is not np.matrix:
            raise TypeError('expected outputs must be np matrix')
        elif expected_outputs.dtype != np.float64:
            raise TypeError('expected outputs must be a float type')
        elif inputs.shape != (net.num_outputs(), 1):
            raise ValueError('expected outputs has wrong shape')

        def cost(net_output):
            return np.sum(np.square(net_output-expected_outputs))
        
        gradient = np.array([GradientLayer(None, None) for _ in net.layers])

        for layer, grad_layer in zip(net.layers, gradient):
            # Calculate gradient for bias of this layer
            grad_layer.biases = np.zeros_like(layer.biases)
            for index, _ in np.ndenumerate(layer.biases):
                old_val = layer.biases[index]

                layer.biases[index] = old_val + self.epsilon
                right_cost = cost(net.evaluate(inputs))

                layer.biases[index] = old_val - self.epsilon
                left_cost = cost(net.evaluate(inputs))

                layer.biases[index] = old_val
                grad_layer.biases[index] = (right_cost - left_cost)/2
            
            # Calculate gradient for weights of this layer
            grad_layer.weights = np.zeros_like(layer.weights)
            for index, _ in np.ndenumerate(layer.weights):
                old_val = layer.weights[index]

                layer.weights[index] = old_val + self.epsilon
                right_cost = cost(net.evaluate(inputs))

                layer.weights[index] = old_val - self.epsilon
                left_cost = cost(net.evaluate(inputs))

                layer.weights[index] = old_val
                grad_layer.weights[index] = (right_cost - left_cost)/2

        return gradient

class Backprop:
    def __init__(self):
        pass

    def get_gradient(self, net, inputs, expected_outputs):
        '''
        Return gradient as a np array of gradient layers.
        '''
        if type(inputs) is not np.matrix:
            raise TypeError('inputs must be numpy matrix')
        elif inputs.dtype != np.float64:
            raise TypeError('inputs must be a float type')
        elif inputs.shape != (net.num_inputs(), 1):
            raise ValueError('inputs has wrong shape')

        if type(expected_outputs) is not np.matrix:
            raise TypeError('expected outputs must be np matrix')
        elif expected_outputs.dtype != np.float64:
            raise TypeError('expected outputs must be a float type')
        elif inputs.shape != (net.num_outputs(), 1):
            raise ValueError('expected outputs has wrong shape')
        
        zs = self.get_raw_output_cache(net, inputs)
        actual_outputs = net.activation(zs[-1])
        # How cost function changes w.r.t. current layer output
        dC_dx = actual_outputs - expected_outputs

        gradient = np.array([GradientLayer(None, None) for _ in net.layers])

        def reversed_enumerate(seq):
            for i, elem in enumerate(reversed(seq)):
                yield len(seq)-1-i, elem

        for index, layer in reversed_enumerate(net.layers):
            z = zs[index]
            # Get output from previous layer
            if index > 0:
                prev_x = net.activation(zs[index-1])
            else:
                prev_x = inputs
            
            num_outputs = layer.size_out # m
            num_inputs = layer.size_in # n

            if z.shape != (num_outputs, 1):
                raise RuntimeError('size of z is wrong')
            elif z.dtype != np.float64:
                raise RuntimeError('type of z is wrong')
            if prev_x.shape != (num_inputs, 1):
                raise RuntimeError('size of prev_x is wrong')
            elif prev_x.dtype != np.float64:
                raise RuntimeError('type of prev_x is wrong')
            
            # Output from this layer
            current_x = net.activation_derivative(z)

            gradient[index].biases = np.multiply(dC_dx, current_x)

            tiled = np.tile(current_x, (1, num_inputs))
            if tiled.shape != (num_outputs, num_inputs):
                raise RuntimeError('tiled matrix has wrong shape.')
            gradient[index].weights = np.multiply(tiled, dC_dx * prev_x.T)

            if gradient[index].weights.shape != (num_outputs, num_inputs):
                raise RuntimeError('Weight gradient is wrong shape')
            
            dC_dx = layer.weights.T * np.multiply(dC_dx, current_x)
        
        return gradient

    def get_raw_output_cache(self, net, inputs):
        '''
        Calculates raw output (i.e. without activation function) of net for each layer, and returns it.
        '''
        zs = []
        outputs = inputs
        for layer in net.layers:
            zs.append(net.evaluate_layer_without_activation(layer, outputs))
            outputs = net.activation(zs[-1])
        return zs