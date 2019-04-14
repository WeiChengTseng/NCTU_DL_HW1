from builtins import range
from builtins import object
import numpy as np

from layers import *

class FullyConnectedNet(object):
    def __init__(self,
                 hidden_dims,
                 input_dim=6,
                 num_classes=2,
                 dropout=0,
                 use_batchnorm=False,
                 reg=0.0,
                 weight_scale=1e-2,
                 dtype=np.float32,
                 seed=None):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################

        L = self.num_layers
        self.params_bn = {}

        # parameter setting
        self.params['W1'] = weight_scale * np.random.randn(
            input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        for i in range(2, L):
            self.params['W' + str(i)] = weight_scale * np.random.randn(
                hidden_dims[i - 2], hidden_dims[i - 1])
            self.params['b' + str(i)] = np.zeros(hidden_dims[i - 1])
        self.params['W' + str(L)] = weight_scale * np.random.randn(
            hidden_dims[L - 2], num_classes)
        self.params['b' + str(L)] = np.zeros(num_classes)

        # batch normalization
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])
            for i in range(2, L):
                self.params['gamma' + str(i)] = np.ones(hidden_dims[i - 1])
                self.params['beta' + str(i)] = np.zeros(hidden_dims[i - 1])

        ############################################################################

        # dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # batch normalize
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{
                'mode': 'train'
            } for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        ############################################################################

        cache, cache_relu, cache_nor, cache_dropout = {}, {}, {}, {}
        L = self.num_layers

        out, cache[1] = affine_forward(X, self.params['W1'], self.params['b1'])
        if self.use_batchnorm:
            out, cache_nor[1] = batchnorm_forward(out, self.params['gamma1'],
                                                  self.params['beta1'],
                                                  self.bn_params[0])
        if self.use_dropout:
            out, cache_dropout[1] = dropout_forward(out, self.dropout_param)
        next_input, cache_relu[1] = relu_forward(out)

        for i in range(2, L):
            out, cache[i] = affine_forward(next_input,
                                           self.params['W' + str(i)],
                                           self.params['b' + str(i)])
            if self.use_batchnorm:
                out, cache_nor[i] = batchnorm_forward(
                    out, self.params['gamma' + str(i)],
                    self.params['beta' + str(i)], self.bn_params[i - 1])
            if self.use_dropout:
                out, cache_dropout[i] = dropout_forward(
                    out, self.dropout_param)
            next_input, cache_relu[i] = relu_forward(out)
        scores, cache[L] = affine_forward(
            next_input, self.params['W' + str(L)], self.params['b' + str(L)])

        ############################################################################


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        for i in range(1, L + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)]**2)
        dh, grads['W' + str(L)], grads['b' + str(L)] = affine_backward(
            dscores, cache[L])
        grads['W' + str(L)] += self.params['W' + str(L)] * self.reg
        for i in range(L - 1, 0, -1):
            dh = relu_backward(dh, cache_relu[i])
            if self.use_batchnorm:
                dh, grads['gamma' +
                          str(i)], grads['beta' + str(i)] = batchnorm_backward(
                              dh, cache_nor[i])
            if self.use_dropout:
                dh = dropout_backward(dh, cache_dropout[i])
            dh, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(
                dh, cache[i])
            grads['W' + str(i)] += self.params['W' + str(i)] * self.reg

        ############################################################################

        return loss, grads