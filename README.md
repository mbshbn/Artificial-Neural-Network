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
If we use `stepFunction` as the activation function, the prediction is `0`, or `1` indicating false or true..

However, the error function should be continuous and differentiable to be able to apply gradient descent, for example the sigmiod function.
If we use the sigmoid function as the activation function, the prediction values are between `0` and `1`, which is the probability of being true or false.

If we have more classes than simply true or false, then the scores need to be positive (using exponential function) and sum up to 1. So, instead of the sigmoid function, we use the Softmax function. The following function, takes a list of numbers and returns the softmax output of them:
```
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
```






This repo is based on the Udacity Self-driving car engineering Nondegree course.
