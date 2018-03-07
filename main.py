import GradientCalculator
import NeuralNet
import GradientDescent
import Visualizer
import activation

import numpy.matlib as np

net = NeuralNet.NeuralNet(num_inputs=1,
                         num_outputs=1,
                         num_hidden_layers=2,
                         layer_size=10,
                         activation=activation.relu,
                         activation_derivative=activation.relu_derivative)

grad_approx = GradientCalculator.NumericApproximator(epsilon=0.001)
grad_backprop = GradientCalculator.Backprop()

fn_approx = GradientDescent.SingleFunctionApprox(lambda x: 0.5+0.5*np.sin(x*10))#np.absolute(x-0.5) + 0.2)

visualizer = Visualizer.Visualizer()


inputs = np.matrix([[0.5]])
expected_outputs = np.matrix([[0.5]])

a = grad_approx.get_gradient(net, inputs, expected_outputs)
b  = grad_backprop.get_gradient(net, inputs, expected_outputs)
print(a)
print(b)
print('length: {}'.format(np.sum(np.square(a - b))))


fn_approx.descend(net, grad_backprop, visualizer)

print(net.layers[1].weights)