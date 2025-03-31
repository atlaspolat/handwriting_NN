import numpy as np

class Network(object):

    def __init__(self, layer_size: list):
        self.num_layers = len(layer_size)
        self.biases = [np.random.randn(y, 1) for y in layer_size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_size[:-1], layer_size[1:])]


    def feedforward(self, input_layer: np.array):
        """Return the output of the network if "a" is input.
            We loop over the layers and compute the output of each layer
            
        Args:
            input_layer (np.array): The input layer of the network
        
        Returns:
            np.array: The output layer of the network
            """
        next_layer = input_layer

        for i in range(self.num_layers - 1):
            next_layer = self.sigmoid(np.matmul(self.weights[i], next_layer) + self.biases[i])

        
        return next_layer
        
    
    def StochasticGradientDescent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """ Train the neural network using mini-batch stochastic gradient descent.
        The training_data is a list of tuples (x, y) representing the training inputs and the desired outputs.
        If test_data is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.
        This is useful for tracking progress, but slows things down substantially."""

        if test_data:
            n_test = len(test_data)


        for example in range(epochs):
            #Shuffle the training data
            np.random.shuffle(training_data)
            mini_batches = [training_data[batch_start: batch_start+mini_batch_size] for batch_start in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {example}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {example} complete")



    def update_mini_batch(self, mini_batch, learning_rate):
        """update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "learning_rate" is the learning rate."""

        grad_vector_bias = [np.zeros(bias.shape) for bias in self.biases]
        grad_vector_weight = [np.zeros(weight.shape) for weight in self.weights]

        for train_input, train_output in mini_batch:
            delta_grad_vector_weight, delta_grad_vector_bias = self.backpropagation(train_input, train_output)
            
            for index in range(len(grad_vector_bias)):
                grad_vector_bias[index] += delta_grad_vector_bias[index] 
            
            for index in range(len(grad_vector_weight)):
                grad_vector_weight[index] += delta_grad_vector_weight[index]

        for index in range(len(self.biases)):
            self.biases[index] -= (learning_rate/len(mini_batch))*grad_vector_bias[index]

        for index in range(len(self.weights)):
            self.weights[index] -= (learning_rate/len(mini_batch))*grad_vector_weight[index]

    def backpropagation(self, input_layer, output_layer):
        """Return a tuple (grad_vector_bias, grad_vector_weight) representing the gradient for the cost function C_x.
        grad_vector_bias and grad_vector_weight are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights."""
        grad_vector_bias = [np.zeros(bias.shape) for bias in self.biases]
        grad_vector_weight = [np.zeros(weight.shape) for weight in self.weights]

        #feedforward
        activation = input_layer
        activations = [input_layer]
        unactivated_layers = []
        for index in range(len(self.biases)):
            unactivated_layer = np.matmul(self.weights[index], activation) + self.biases[index]
            unactivated_layers.append(unactivated_layer)
            activation = self.sigmoid(unactivated_layer)
            activations.append(activation)
        
        #backward pass


        delta = self.cost_derivative(activations[-1], output_layer)*self.sigmoid_prime(unactivated_layers[-1])
        grad_vector_bias[-1] = delta
        grad_vector_weight[-1] = np.matmul(delta, activations[-2].transpose())

        for index in range(2, self.num_layers):
            unactivated_layer = unactivated_layers[-index]
            delta_unactivated_layer = self.sigmoid_prime(unactivated_layer)
            delta = np.matmul(self.weights[-index+1].transpose(), delta)*delta_unactivated_layer
            grad_vector_bias[-index] = delta
            grad_vector_weight[-index] = np.matmul(delta, activations[-index-1].transpose())
        
        return (grad_vector_weight, grad_vector_bias)
    

    def evaluate(self, test_data):
        """return the number of test inputs for which the neural network outputs the correct result."""
        """Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation."""

        test_results = [(np.argmax(self.feedforward(test_input)), test_output) for (test_input, test_output) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    


    def cost_derivative(self, output_activations, expected_result):
        """Return the vector of partial derivatives partial C_x partial a for the output activations."""
        return (output_activations - expected_result)


### Helper functions

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))