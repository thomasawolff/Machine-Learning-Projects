import random
import numpy as np

# Input array
for b in range(5):
    x = np.array([[random.randint(0,1) for i in range(4)],[random.randint(0,1) for i in range(4)],\
                  [random.randint(0,1) for i in range(4)],[random.randint(0,1) for i in range(4)]])
    print x
    ##print''
    ##
    ##x = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]])
    ##print x

    # Output
    y = np.array([[random.choice(x[0])],[random.choice(x[1])],[random.choice(x[2])],[random.choice(x[3])]])
    print''
    print y
    print''

    # Sigmoid function
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
        #return np.tanh(x) # tanh activation function

    # Derivative of Sigmoid Function
    def derivatives_sigmoid(x):
        return x * (1-x)
        #return 1. - x * x # derivative of tanh function

    # Variable initialization
    epoch = 1000000 # Setting training iterations
    lr = 0.1 # Setting learning rate
    inputLayer_nuerons = x.shape[1] # Number of features in data set
    hiddenLayer_nuerons = 4 # Number of hidden layer nuerons in each layer
    output_nuerons = 1 # number of nuerons at output layer

    # Weight and bias initialization
    wh = np.random.uniform(size=(inputLayer_nuerons,hiddenLayer_nuerons)) # weight matrix to hidden layer
    wout = np.random.uniform(size=(hiddenLayer_nuerons,output_nuerons)) # bias matrix to hidden layer

    bh = np.random.uniform(size=(1,hiddenLayer_nuerons)) # weight matrix to output layer
    bout = np.random.uniform(size=(1,output_nuerons)) # bias matrix to output layer

    for i in range(epoch):
        
        #### Forward propogation

        # Calculate hidden layer input
        hidden_layer_input2 = np.dot(x,wh) + bh # Hidden layer 1
        hidden_layer_input1 = np.dot(hidden_layer_input2,wh) + bh # Hidden layer 2
        hidden_layer_input = np.dot(hidden_layer_input1,wh) + bh # Hidden layer 3
        
        # Perform non-linear transformation on hidden linear input    
        hiddenlayer_activations = sigmoid(hidden_layer_input)

        # Perform linear and non-linear transformation of hidden layer activations of output layer
        output_layer_input = np.dot(hiddenlayer_activations,wout) + bout
        output = sigmoid(output_layer_input)
        

        #### Backpropagation

        # Calculate gradient of Error(E) of output layer
        E = y - output

        # Compute derivative of output and hidden layer
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

        # Compute delta (change factor) of output layer
        d_output = E * slope_output_layer

        # Compute error of hidden layer
        Error_at_hidden_layer = d_output.dot(wout.T)

        # Compute delta of hidden layer. Error of hidden layer * slope hidden layer activation
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

        # Update weight of both output and hidden layer
        wout += hiddenlayer_activations.T.dot(d_output) * lr
        wh += x.T.dot(d_hiddenlayer) * lr
        
        # Update biases of both output and hidden layer. The biases in network can be updated
        # from aggregated errors at that nueron
        bout += np.sum(d_output,axis=0,keepdims=True) * lr
        bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr


    print '{0:.8f}'.format(float(output[0]))
    print '{0:.8f}'.format(float(output[1]))
    print '{0:.8f}'.format(float(output[2]))
    print '{0:.8f}'.format(float(output[3]))
    print'-'*50,b+1
    print''

