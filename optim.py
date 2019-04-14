import numpy as np
"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
def update(w, dw, config=None):
Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient of the
      loss with respect to w.
    - config: A dictionary containing hyperparameter values such as learning
      rate, momentum, etc. If the update rule requires caching values over many
      iterations, then config will also hold these cached values.
Returns:
    - next_w: The next point after the update.
    - config: The config dictionary to be passed to the next iteration of the
      update rule.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None

    ###########################################################################

    mu = 0.9
    v = mu * v - config['learning_rate'] * dw
    next_w = v + w

    ###########################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None

    ###########################################################################
 
    config['cache'] = config['decay_rate'] * config['cache'] + (
        1 - config['decay_rate']) * dx**2
    next_x = x - config['learning_rate'] * dx / (
        np.sqrt(config['cache']) + config['epsilon'])

    ###########################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None

    ###########################################################################

    config['t'] += 1
    config['m'] = config['m'] * config['beta1'] + (1 - config['beta1']) * dx
    config['v'] = config['v'] * config['beta2'] + (1 - config['beta2']) * dx**2
    mb = config['m'] / (1 - config['beta1']**config['t'])
    vb = config['v'] / (1 - config['beta2']**config['t'])
    next_x = x - config['learning_rate'] * mb / (
        np.sqrt(vb) + config['epsilon'])

    ###########################################################################

    return next_x, config