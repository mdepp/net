import numpy.matlib as np

def relu(inputs):
    '''
    Should work for inputs that are numbers, numpy arrays, or numpy matrices

    >>> inputs = np.array([[-2], [0], [2]])
    >>> relu(inputs)
    array([[0],
           [0],
           [2]])
    >>> inputs = np.matrix([[-2, 0, 2]])
    >>> relu(inputs)
    matrix([[0, 0, 2]])
    >>> inputs = np.matrix([[-2], [0], [2]])
    >>> relu(inputs)
    matrix([[0],
            [0],
            [2]])
    >>> relu(2)
    2
    >>> relu(-2)
    0
    >>> relu(0)
    0

    '''
    return np.maximum(inputs, 0)


def relu_derivative(inputs):
    '''
    Should work for numbers, numpy arrays, or numpy matrices. Because of
    implementation, this always returns floating-point numbers

    >>> inputs = np.array([[-2], [0], [2]])
    >>> relu_derivative(inputs)
    array([[0.],
           [0.],
           [1.]])
    >>> inputs = np.matrix([[-2, 0, 2]])
    >>> relu_derivative(inputs)
    matrix([[0, 0, 1]])
    >>> inputs = np.matrix([[-2], [0], [2]])
    >>> relu_derivative(inputs)
    matrix([[0.],
            [0.],
            [1.]])
    >>> relu_derivative(2)
    1.0
    >>> relu_derivative(-2)
    0.0
    >>> relu_derivative(0)
    0.0
    '''
    return np.heaviside(inputs, 0)

def sigmoid(x):
    #return 1/(1+np.exp(-x))
    return np.arctan(x)
def sigmoid_derivative(x):
    #return np.multiply(sigmoid(x), 1-sigmoid(x))
    return 1/(np.multiply(x,x)+1)
