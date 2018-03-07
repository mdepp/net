import numpy.matlib as np
from matplotlib import pyplot
from matplotlib.colors import to_rgb
from time import time

class Visualizer:
    def __init__(self):
        self.fig = pyplot.figure()
        self.function_subplot = self.fig.add_subplot(111)
        self.cost_subplot = self.fig.add_subplot(212)
        
        self.input_axis = np.linspace(0, 1) # TODO: Replace with domain
        self.ideal_fn = np.zeros_like(self.input_axis)
        self.net_fn = np.zeros_like(self.input_axis)
        self.cost_fn = np.zeros_like(self.input_axis)

        self.ideal_fn_line, = self.function_subplot.plot(self.input_axis, self.ideal_fn)
        self.net_fn_line, = self.function_subplot.plot(self.input_axis, self.net_fn)
        self.cost_fn_line, = self.cost_subplot.plot(self.input_axis, self.cost_fn)

        pyplot.show(block=False)

        self.last_compare_time = time()
        self.compare_period = 0.5

    def compare_functions(self, ideal_function, net):
        '''
        Compare ideal function and net approximation
        '''
        current_time = time()
        if current_time-self.last_compare_time > self.compare_period:
            self.last_compare_time = current_time
            for index, fn_input in enumerate(self.input_axis):
                net_input = np.matrix([[fn_input]])
                net_output = net.evaluate(net_input)
                self.ideal_fn[index] = ideal_function(fn_input)
                self.net_fn[index] = net_output[0,0]
                self.cost_fn[index] = (self.ideal_fn[index]-self.net_fn[index])**2
            
            self.ideal_fn_line.set_ydata(self.ideal_fn)
            self.net_fn_line.set_ydata(self.net_fn)
            self.cost_fn_line.set_ydata(self.cost_fn)
            
            self.function_subplot.set_ylim([-1, 1])
            self.cost_subplot.set_ylim([0, 1])

            self.fig.canvas.draw()