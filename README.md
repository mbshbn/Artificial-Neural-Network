# Artificial-Neural-Network
Here, we start from a simple perception algorithm for a binary classification problem, and improve it to build a neural network.

## Perception algorithm (logistic regression, binary classification)
Update the weights and biases for a binary classification problem (the step function is used as the activation function):
```
for i in range(len(X)):
    y_hat = stepFunction((np.matmul(X[i],W)+b)[0])
    if y[i]-y_hat == 1:
        W[0] += X[i][0]*learn_rate
        W[1] += X[i][1]*learn_rate
        b += learn_rate
    elif y[i]-y_hat == -1:
        W[0] -= X[i][0]*learn_rate
        W[1] -= X[i][1]*learn_rate
        b -= learn_rate
```    

## The activation function

If we use a `stepFunction` as the activation function, the prediction is `0` or `1` indicating false or true.
It is difficult to optimize using this activation function.
The error function should be continuous and differentiable to be able to apply gradient descent.
So, the sigmiod function is a good choice.
If we use the sigmoid function as the activation function, the prediction values are between `0` and `1`, which is the probability of being true or false.
```
def sigmoid(x):
    return 1/(1+np.exp(-x))
```
If we have more classes than simply true or false, then we can normalize the scores (probability should be between 0 and 1).
Also, the scores need to be positive, so we use the exponential function.
So, instead of the sigmoid function, we use the softmax function.
The following function, takes a list of numbers and returns the softmax output of them:
```
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
```
The softmax function for classifications problems with two classes are the same as the sigmoid function.
For classification problem with 3 or more classes, we need more variables than just `0` and `1`. So, we use the One-Hot encoding.

## The error function (cross entropy)
A better model gives a higher value of the multiplication of the independent probabilities of the events/objects in its group.
To find a better model, maximizing the likelihood is equivalent to minimizing the error function.
Calculation of products of small numbers (probabilities are between 0 and 1) are problematic due to computational problems.
So, we use natural logarithm function to transfer multiplication to sum.
Remember, the logarithm of numbers between 0 and 1 are negative, and the logarithm of 1 is zero.
The result of `-log` is called the cross entropy.
A better model, has a smaller cross entropy.
In other words, a correct classified object has a probability close to 1, and its cross entropy `-log(1)` is close to `0`.
So to find an optimized model, the goal id to decrease the cross entropy.

In the following, cross entropy is calculated for the problem with 2 classes (binary classification problem, or a logistic regression); Y represents the category, P represents the probability
```
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```
For a multi-class, the cross-entropy is computed as follows, `m` is the number of classes:
```
TODO: write code (28)
```

## Minimizing the error function (gradient descent)

Now that we have calculated the model error, to minimize the error to obtain a better model, we use gradient descent for one output unit  as follows
```
# The neural network output (y-hat)
nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])
# or nn_output = sigmoid(np.dot(x, weights))
# output error (y - y-hat)
error = y - nn_output
# error term
error_term = error * sigmoid_prime(np.dot(x,weights))
# Gradient descent step
del_w = [ learnrate * error_term * x[0],
                 learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x

```
The derivate of sigmoid function is given by
```
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
```
This is another reason why the sigmoid function is widely used.

Note that the gradient descent in case of the binary classification becomes the same as perception algorithm.
For perception, if the point is misclassified, its weights are updated.
For perception, if a point classified correctly, the output error (y - y-hat) becomes 0, and weights are not changed.
But in gradient descent, all points are updated.
For a binary classification, if a point is misclassified, the weights are changed such that the line becomes closer to the point, and if it is classified correctly, the weights are changed such that the line goes further away from the point

## Nonlinear boundaries (Neural Networks)
Until here, we assumed the boundaries are linear.
To solve the problem with nonlinear boundaries, we use the Neural Networks.
It is actually a multi-layer perceptron.

The input layer determines the number of dimension.
If we have n input, it indicates that the data is in n-dimensional space.
If we want to have a higher nonlinear boundary, then we can increase the number of nodes in the hidden layer.
If we add mode hidden layers, then we call it deep neural networks, which result in more nonlinear models.
If we have a multi-class classification problem, the output layer has more than one node. For example, for classification with 3 classes, we have an output layer with 3 nodes.

For training the Neural Networks, we use the feedforward algorithm, which is the calculation of the output based on the weights and sigmoid function applied to the input layer.
For example, for a network size of `N_input = 4, N_hidden = 3, N_output = 2`, the weight matrices' size are defined as follows:
`weights_input_to_hidden`  and `weights_hidden_to_output`  are `N_input` by `N_hidden`, and `N_hidden` by `N_output`, respectively.
The feedforward algorithm is as follows:
```
# Hidden_layer output :
hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

# Output-layer Output:
output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

```

This repo is based on the Udacity Self-driving car engineering Nondegree course.
