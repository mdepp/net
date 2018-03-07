import numpy.matlib as np

class SingleFunctionApprox:
    '''
    A routine to mimic a function from [0,1] to [0,1].
    '''
    def __init__(self, function):
        self.function = function
        self.domain = np.linspace(0, 1)
    
    def descend(self, net, gradcalc, visualizer):
        num_passes = 1000
        num_sampled = 100
        learning_rate = 0.1

        from tqdm import tqdm
        for _ in tqdm(range(num_passes)):
            average_gradient = self.get_average_gradient(net,
                                                np.random.choice(self.domain, num_sampled),
                                                gradcalc)
            
            self.update_net(net, average_gradient * -learning_rate)
            
            visualizer.compare_functions(self.function, net)


    def get_average_gradient(self, net, input_samples, gradcalc):

        first = True
        for x in input_samples:
            net_inputs = np.matrix([[x]], dtype=np.float64)
            net_expected_outputs = self.function(net_inputs)
            gradient = gradcalc.get_gradient(net, net_inputs, net_expected_outputs)
            if first:
                average_gradient = gradient
                first = False
            else:
                average_gradient += gradient
            
        average_gradient /= len(input_samples)
        return average_gradient
    
    def update_net(self, net, gradient):
        for layer, grad_layer in zip(net.layers, gradient):
                if grad_layer.weights.shape != layer.weights.shape:
                    raise RuntimeError('Gradient weights have wrong shape.')
                if grad_layer.biases.shape != layer.biases.shape:
                    raise RuntimeError('Gradient biases have wrong shape.')
                layer.weights += grad_layer.weights
                layer.biases += grad_layer.biases