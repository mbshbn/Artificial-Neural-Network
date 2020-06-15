# Artificial-Neural-Network

Perception algorithm to update the weights and biases using step function as the activation function:
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
If we use `stepFunction` as the activation function, the prediction is `0`, or `1` indicating false or true.

However, the error function should be continuous and differentiable to be able to apply gradient descent, for example the sigmiod function.
If we use the sigmoid function as the activation function, the prediction values are between `0` and `1`, which is the probability of being true or false.

If we have more classes than simply true or false, then the scores need to be positive (using exponential function) and all scores should sum up to 1.
So, instead of the sigmoid function, we use the Softmax function.
 following function, takes a list of numbers and returns the softmax output of them:
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
For classification problem with 3 or more classes, we need more variable than just `0` and `1`. so, we use the One-Hot encoding.

A better model gives a higher value of the multiplication of the independent probabilities of the events/objects in its group.
To find a better model, maximizing the likelihood is equivalent to minimizing the error function.
Products of small probabilities are problematic.
So, we use natural logarithm function to transfer multiplication to sum.
remember, the logarithm of numbers between 0 and 1 are negative, and the logarithm of 1 is zero.
The result of `-log' is called the cross entropy. A better model, has a smaller cross entropy.
In other words, a correct classified object, has a probability close to 1, and so `-log(1)` or its cross entropy is close to `0`.


In the following, cross entropy is calculated, Y represents the category, P represents the probability
```
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```


This repo is based on the Udacity Self-driving car engineering Nondegree course.
