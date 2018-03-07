import numpy.matlib as np
import numpy.random
import random

import cProfile, pstats, io

class Layer:
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out

        # weights is an nxm matrix, where
        #   n -- size of output layer
        #   m -- size of input layer
        self.weights = np.rand(size_out, size_in)*2-1

        self.biases = np.rand(size_out,1)*2-1

class GradientComponent:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def __repr__(self):
        return 'Grad: weights, biases\nw={0}\nb={1}\n'.format(self.weights, self.biases)
    
    def __add__(self, other):
        return GradientComponent(self.weights + other.weights, self.biases + other.biases)
    
    def __iadd__(self, other):
        self.weights += other.weights
        self.biases += other.biases
        return self
    
    def __mul__(self, other):
        if other is GradientComponent:
            return GradientComponent(np.multiply(self.weights, other.weights), np.multiply(self.biases, other.biases))
        else:
            return GradientComponent(np.multiply(self.weights, other), np.multiply(self.biases, other))
    
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

class NNet:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, layer_size, activation=relu, activation_derivative=relu_derivative):
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
    

        #self.layers = [Layer(1, 1)]
        #self.layers[0].weights[0,0] = 1
        #self.layers[0].biases[0,0] = 1
        
        '''self.layers = [Layer(1, 2), Layer(2, 2), Layer(2, 1)]
        self.layers[0].weights[0,0] = 1
        self.layers[0].biases[0,0] = 0
        self.layers[-1].weights = np.matrix([[1.,-2.]])
        self.layers[-1].biases[0,0] = 0

        self.layers[1].weights = np.matrix([[1.,1.],[1.,1.]])
        self.layers[1].biases = np.matrix([[0.],[-0.5]])'''
        

        self.activation = activation
        self.activation_derivative = activation_derivative

        self.output_cache = [0]*len(self.layers)
    
    def calc_output(self, inputs):
        '''
        Get the output of the net in response to some input (an nx1 column numpy
        array.
        '''
        outputs = np.matrix(inputs, dtype=np.float64).transpose()
        for layer in self.layers:
            if layer.weights.shape != (layer.size_out, layer.size_in):
                raise RuntimeError('layer weights size is inconsistent.')
            elif outputs.shape != (layer.size_in, 1):
                raise RuntimeError('input size is inconsistent with layer')
            elif layer.biases.shape != (layer.size_out, 1):
                raise RuntimeError('Bias size is inconsistent with output size')
            elif (layer.weights*outputs).shape != layer.biases.shape:
                raise RuntimeError('multiplication not as expected!')
            
            outputs = self.activation(layer.weights*outputs + layer.biases)
        
        return outputs

    def populate_output_cache(self, inputs):
        '''
        Populate output_cache with the output from each layer, without activation
        function. Returns actual output.
        '''
        if len(self.output_cache) != len(self.layers):
            raise ValueError('Layer output cache is wrong size')
        
        if inputs is not np.matrix:
            inputs = np.matrix(inputs, dtype=np.float64).transpose()
        outputs = inputs
        for index, layer in enumerate(self.layers):
            self.output_cache[index] = layer.weights*outputs + layer.biases
            outputs = self.activation(self.output_cache[index])
        
        return outputs
        
    
    def backprop(self, inputs, expected_outputs):
        '''
        Get the gradient as calculated from a set of inputs and expected outputs.
        Gradient is a numpy array of GradientComponents.
        '''
        inputs = np.matrix(inputs, dtype=np.float64).transpose()
        expected_outputs = np.matrix(expected_outputs, dtype=np.float64).transpose()

        #if inputs.shape != (self.layers[0].size_in, 1):
        #    raise RuntimeError('Wrong input shape')
        #elif expected_outputs.shape != (self.layers[-1].size_out, 1):
        #    raise RuntimeError('Wrong expected output shape')

        actual_outputs = self.populate_output_cache(inputs)
        
        # How cost function changes w.r.t. current layer output
        dC_dx = 2*(actual_outputs - expected_outputs)

        gradient = np.array([GradientComponent(None, None) for _ in self.layers])

        #if len(self.layers) != len(self.output_cache):
        #    raise RuntimeError('Invalid size of output cache.')

        for reversed_index, layer in enumerate(reversed(self.layers)):
            index = len(self.layers)-(reversed_index+1)
            # get output from this layer due to previous inputs
            z = self.output_cache[index]
            if index > 0: # not first layer
                prev_x = self.activation(self.output_cache[index-1])
            else:
                prev_x = inputs

            num_outputs = layer.size_out # m
            num_inputs = layer.size_in # n

            if z.shape != (num_outputs, 1):
                raise RuntimeError('size of z is wrong')
            if prev_x.shape != (num_inputs, 1):
                raise RuntimeError('size of prev_x is wrong')


            gradient[index].biases = np.matrix(np.multiply(dC_dx, self.activation_derivative(z)), dtype=np.float64, copy=True)


            tiled = np.tile(self.activation_derivative(z), (1, num_inputs))
            if tiled.shape != (num_outputs, num_inputs):
                raise RuntimeError('tiled matrix has wrong shape.')
            gradient[index].weights = np.matrix(np.multiply(
                np.tile(self.activation_derivative(z), (1, num_inputs)),
                dC_dx * prev_x.T), dtype=np.float64, copy=True)
            
            if gradient[index].weights.shape != (num_outputs, num_inputs):
                raise RuntimeError('Weight gradient is wrong shape')

            dC_dx = layer.weights.T * np.multiply(dC_dx, self.activation_derivative(z))


        def distance(x, y):
            return np.sqrt(np.sum(np.absolute(x-y)))
        
        #approx_gradient = self.approximate_gradient(inputs, expected_outputs, 0.001)
        #for g, ag in zip(gradient, approx_gradient):
        #    print('<w: {0}, b: {1}>, '.format(distance(g.weights, ag.weights), distance(g.biases, ag.biases)), end='')
        #print()

        return gradient

    def approximate_gradient(self, inputs, expected_outputs, epsilon=0.001):
        '''
        Calculate and return a numeric approximation of the gradient (assumes
        net has only one output)
        '''
        gradients = np.array([GradientComponent(None, None) for _ in self.layers])

        def distance(x, y):
            return x-y
        
        def cost(output):
            return np.sum(np.square(output-expected_outputs))

        for layer_index, (layer, gradient) in enumerate(zip(self.layers, gradients)):
            # Calculate gradient of biases for this layer
            gradient.biases = np.zeros_like(layer.biases)
            for index, _ in np.ndenumerate(np.zeros_like(layer.biases)):
                old_val = layer.biases[index] # floating-point errors
                layer.biases[index] = old_val + epsilon
                right_cost = cost(self.calc_output(inputs))
                layer.biases[index] = old_val - epsilon
                left_cost = cost(self.calc_output(inputs))
                layer.biases[index] = old_val

                gradient.biases[index] = distance(right_cost, left_cost) / (2*epsilon)
                #if index != (0,0) and left_output != right_output:
                #    print(gradient.biases[index])
                #    raise RuntimeError()

            # Calculate gradient of weights for this layer
            gradient.weights = np.zeros_like(layer.weights)
            for index, _ in np.ndenumerate(np.zeros_like(layer.weights)):
                old_val = layer.weights[index] # floating-point errors
                layer.weights[index] = old_val + epsilon
                right_cost = cost(self.calc_output(inputs))
                layer.weights[index] = old_val - epsilon
                left_cost = cost(self.calc_output(inputs))
                layer.weights[index] = old_val

                gradient.weights[index] = distance(right_cost, left_cost) / (2*epsilon)

        return gradients

    def mimic_function(self, fn):
        '''
        Attempt to mimic a function from [0,1]->[0,1].
        '''


        num_passes = 1000
        num_sampled = 100
        learning_rate = 0.1

        profiler = cProfile.Profile()
        profiler.enable()


        from matplotlib import pyplot
        from matplotlib.colors import to_rgb
        fig = pyplot.figure()
        #_, axarr = fig.subplots(2)
        fn_subplot = fig.add_subplot(111)
        cost_subplot = fig.add_subplot(212)


        compare_axis = np.linspace(0,1)
        cost_fn = np.zeros_like(compare_axis) #np.zeros_like(time_axis)
        expected_fn = fn(compare_axis)
        actual_fn = np.zeros_like(expected_fn)

        #axarr[0].plot(time_axis, cost_at_time)
        #axarr[1].plot(compare_axis, expected_fn)
        #actual_fn_line, = axarr[1].plot(compare_axis, actual_fn)
        actual_fn_line, = fn_subplot.plot(compare_axis, actual_fn)
        fn_subplot.plot(compare_axis, expected_fn)
        cost_line, = cost_subplot.plot(compare_axis, cost_fn)

        pyplot.show(block=False)

        # Generate some example input test cases
        print('Generating test data...')
        input_data = [random.random() for _ in range(10000)]


        print('Training...')
        from tqdm import tqdm
        for pass_index in tqdm(range(num_passes)):
            first = True

            for inputs in np.random.choice(input_data, num_sampled):
                gradient = self.backprop([inputs], [fn(inputs)])
                #print(gradient)
                if first:
                    overall_gradient = gradient
                    first = False
                else:
                    #np.add(overall_gradient, gradient, out=overall_gradient)
                    overall_gradient += gradient


            if pass_index%10 == 0:
                avg_cost = 0
                for index in range(len(compare_axis)):
                    output = self.calc_output(np.matrix([[compare_axis[index]]], dtype=np.float64))[0,0]
                    actual_fn[index] = output
                    cost = (expected_fn[index]-output)**2
                    cost_fn[index] = cost
                    avg_cost += cost
                
                avg_cost /= len(compare_axis)
                #cost_at_time[pass_index] = avg_cost
                actual_fn_line.set_ydata(actual_fn)
                cost_line.set_ydata(cost_fn)
                fn_subplot.set_ylim([-1, 1])
                cost_subplot.set_ylim([0, 1])
                fig.canvas.draw()

            
            #np.multiply(overall_gradient, -learning_rate / len(input_data), out=overall_gradient)
            overall_gradient *= -(learning_rate / num_sampled)

            for layer_index, gradient in enumerate(overall_gradient):
                if gradient.weights.shape != self.layers[layer_index].weights.shape:
                    raise RuntimeError('Gradient weights have wrong shape.')
                if gradient.biases.shape != self.layers[layer_index].biases.shape:
                    raise RuntimeError('Gradient biases have wrong shape.')
                self.layers[layer_index].weights += gradient.weights
                self.layers[layer_index].biases += gradient.biases

            #print(abs(inputs - self.calc_output(inputs)[0,0]))
            #print(cost)

        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        print('Finished training...')


def profile(fn, *args):
    from time import time
    start = time()
    val = fn(*args)
    dur = time() - start
    return val, '{} seconds'.format(dur)


    

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # TODO: Change all of this code to be consistent for inputting matrices/arrays, and minimize converisons.

    import time
    net = NNet(num_inputs=1, num_outputs=1, num_hidden_layers=1, layer_size=20, activation=relu, activation_derivative=relu_derivative)
    #print(profile(net.calc_output, [1]))
    #print(profile(net.populate_output_cache, [1]))
    #print(profile(net.backprop, [1], [2]))

    def fn(x):
        return 0 + np.absolute(x-0.5)
    
    time_axis, cost_at_time = net.mimic_function(fn)
    #print(test_input)
    #print(test_output)
    #print([(i, o) for i, o in zip(test_input, test_output)])

    import subprocess
    subprocess.call('notify')

    from matplotlib import pyplot
    from matplotlib.colors import to_rgb
    f, axarr = pyplot.subplots(3)

    axarr[0].plot(time_axis[len(time_axis)//2:], cost_at_time[len(time_axis)//2:])
    axis_np = np.linspace(0,1)
    axis = axis_np.tolist()
    expected_output = fn(axis_np).tolist()
    actual_output = list(net.calc_output([x])[0,0] for x in axis)

    axarr[1].plot(axis, expected_output, 'b', linewidth=3, label='Function to mimic')
    axarr[1].plot(axis, actual_output, 'r-', linewidth=1, label='Output of net')
    handles, labels = axarr[1].get_legend_handles_labels()
    axarr[1].legend(handles, labels)

    axarr[2].plot(np.linspace(-1, 3).tolist(), list(net.calc_output([x])[0,0] for x in np.linspace(-1, 3)))
    #pyplot.plot(test_input, test_output)
    #pyplot.plot(test_input, test_input)
    pyplot.show()


"""
def backprop(self, inputs, expected_outputs):
        '''
        Get the gradient as calculated from a set of inputs and expected outputs.
        Gradient is a numpy array of GradientComponents.
        '''
        inputs = np.matrix(inputs).transpose()
        expected_outputs = np.matrix(expected_outputs).transpose()

        #if inputs.shape != (self.layers[0].size_in, 1):
        #    raise RuntimeError('Wrong input shape')
        #elif expected_outputs.shape != (self.layers[-1].size_out, 1):
        #    raise RuntimeError('Wrong expected output shape')

        actual_outputs = self.populate_output_cache(inputs)
        
        dC_da = 2*(actual_outputs - expected_outputs)

        gradient = np.array([GradientComponent(None, None) for _ in self.layers])

        #if len(self.layers) != len(self.output_cache):
        #    raise RuntimeError('Invalid size of output cache.')

        for reversed_index, layer in enumerate(reversed(self.layers)):
            index = len(self.layers)-(reversed_index+1)
            # get output from this layer due to previous inputs
            z = self.output_cache[index]
            if index > 0: # not first layer
                prev_output = self.activation(self.output_cache[index-1])
            else:
                prev_output = inputs

            num_outputs = len(z)
            num_inputs = len(prev_output)

            #if num_inputs != layer.size_in:
            #    raise RuntimeError('invalid number of inputs')
            #if num_outputs != layer.size_out:
            #    raise RuntimeError('invalid number of outputs')

            
            #if dC_da.shape != (num_outputs, 1):
            #    raise RuntimeError('dC_da has wrong shape.')
            

            #print('Num outputs: {0}\nNum inputs: {1}'.format(num_outputs, num_inputs))


            partial_bias = self.activation_derivative(z)
            #if partial_bias.shape != z.shape:
            #    raise RuntimeError('activation derivative changed shape of result.')
            #elif partial_bias.shape != (num_outputs, 1):
            #    raise RuntimeError('partial bias has wrong shape')
            
            # outer product of sigma(z) and a_{L-1}
            partial_weight = self.activation_derivative(z)*prev_output.transpose()
            #if partial_weight.shape != layer.weights.shape:
            #    raise RuntimeError('partial_weights has the wrong shape')
            #elif partial_weight.shape != (num_outputs, num_inputs):
            #    raise RuntimeError('assumptions about partial_weight.shape are wrong')

            # partial_prev_outputs (\del a_L / \del a_{L-1}) encodes da_i/da_j
            # as partial_prev_outputs[j][i] (this is to ensure multiplication with
            # prev_output works as expected).
            
            #if layer.weights.transpose().shape != (num_inputs, num_outputs):
            #    raise RuntimeError('weights transpose is invalid')
            #elif z.shape != (num_outputs, 1):
            #    raise RuntimeError('z somehow has changed and is now invalid (!!)')
            #elif z.shape != self.activation_derivative(z).shape:
            #    raise RuntimeError('activation derivative changes shape')
            
            column_scaling = np.eye(num_outputs)
            np.fill_diagonal(column_scaling, self.activation_derivative(z))
            partial_prev_output = layer.weights.transpose()*column_scaling
            
            #if partial_prev_output.shape != layer.weights.transpose().shape:
            #    raise RuntimeError('partial previous output has wrong shape.')
            #elif partial_prev_output.shape != (num_inputs, num_outputs):
            #    raise RuntimeError('assumptions about partial_prev_output.shape are wrong')

            # Calculate gradient weights and biases from these partial matrices
            gradient_bias = np.multiply(dC_da, partial_bias)
            #if gradient_bias.shape != (num_outputs, 1):
            #    raise RuntimeError('gradient_bias has wrong shape')

            column_scaling = np.eye(num_outputs)
            np.fill_diagonal(column_scaling, dC_da)
            gradient_weight = (partial_weight.transpose()*column_scaling).transpose()
            #if gradient_weight.shape != layer.weights.shape:
            #    raise RuntimeError('Gradient weights has wrong shape')
            
            gradient[index] = GradientComponent(np.matrix(gradient_weight, copy=True), np.matrix(gradient_bias, copy=True))

            # Update dC/da^(i) for next pass
            dC_da = partial_prev_output*dC_da

        return gradient
"""