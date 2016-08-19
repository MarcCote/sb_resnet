from __future__ import division

import os
import sys
import string
import numpy as np
from time import time
from collections import OrderedDict

import theano
import theano.tensor as T


#########
# Utils #
#########

def calc_kumaraswamy_entropy(posterior_a, posterior_b):
    psi_b_taylor_approx = T.log(posterior_b) - 1./(2 * posterior_b) - 1./(12 * posterior_b**2)
    entropy = T.log(posterior_a*posterior_b)
    entropy += (posterior_a-1)/posterior_a * (-0.57721 - psi_b_taylor_approx - 1/posterior_b)
    entropy += -(posterior_b-1)/posterior_b
    return entropy


def psi_taylor_approx_at_zero(x):
    euler = -T.psi(1).eval()  # 0.57721
    polygamma_2_1 = -2.4041138063191  # T.polygamma(2, 1)
    polygamma_4_1 = -24.886266123440  # T.polygamma(4, 1)
    psi = -1./x - euler + (np.pi**2/6.)*x + (polygamma_2_1/2.)*x**2 + (np.pi**4/90.)*x**3  # + (polygamma_4_1/24.)*x**4
    return psi


def psi_taylor_approx_at_infinity(x):
    return T.log(x) - 1./(2*x) - 1./(12*x**2) + 1./(120*x**4) #- 1./(252*x**6)


def calc_kl_divergence(posterior_a, posterior_b, alpha, beta):
    # compute taylor expansion for E[log (1-v)] term
    # hard-code so we don't have to use Scan()
    # posterior_a.shape = (batch_size, sequence_length)
    # posterior_b.shape = (batch_size, sequence_length)
    kl = 1./(1+posterior_a*posterior_b) * Beta_fn(1./posterior_a, posterior_b)
    kl += 1./(2+posterior_a*posterior_b) * Beta_fn(2./posterior_a, posterior_b)
    kl += 1./(3+posterior_a*posterior_b) * Beta_fn(3./posterior_a, posterior_b)
    kl += 1./(4+posterior_a*posterior_b) * Beta_fn(4./posterior_a, posterior_b)
    kl += 1./(5+posterior_a*posterior_b) * Beta_fn(5./posterior_a, posterior_b)
    kl += 1./(6+posterior_a*posterior_b) * Beta_fn(6./posterior_a, posterior_b)
    kl += 1./(7+posterior_a*posterior_b) * Beta_fn(7./posterior_a, posterior_b)
    kl += 1./(8+posterior_a*posterior_b) * Beta_fn(8./posterior_a, posterior_b)
    kl += 1./(9+posterior_a*posterior_b) * Beta_fn(9./posterior_a, posterior_b)
    kl += 1./(10+posterior_a*posterior_b) * Beta_fn(10./posterior_a, posterior_b)
    kl *= (beta-1)*posterior_b

    # use another taylor approx for Digamma function
    euler = -T.psi(1).eval()  # 0.57721
    # psi_b_taylor_approx = psi_taylor_approx_at_infinity(posterior_b)
    # psi_b_taylor_approx = psi_taylor_approx_at_zero(posterior_b)
    psi_b_taylor_approx = T.switch(posterior_b < 0.53,
                                   psi_taylor_approx_at_zero(posterior_b),
                                   psi_taylor_approx_at_infinity(posterior_b))
    kl += (posterior_a-alpha)/posterior_a * (-euler - psi_b_taylor_approx - 1/posterior_b)
    # kl += (posterior_a-alpha)/posterior_a * (-euler - T.psi(posterior_b) - 1/posterior_b)

    # add normalization constants
    kl += T.log(posterior_a*posterior_b) + T.log(Beta_fn(alpha, beta))

    # final term
    kl += -(posterior_b-1)/posterior_b
    return kl.sum(axis=1)


########
# Misc #
########

def sharedX(value, name=None, borrow=True, broadcastable=None, keep_on_cpu=False):
    """ Transform value into a shared variable of type floatX """
    if keep_on_cpu:
        return T._shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow,
                         broadcastable=broadcastable)

    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow,
                         broadcastable=broadcastable)


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

    return path


chr2idx = {c: i for i, c in enumerate(' ' + string.ascii_lowercase)}
idx2chr = {i: c for i, c in enumerate(' ' + string.ascii_lowercase)}


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt, newline=False):
        self.txt = txt
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print self.txt + "... ",
            sys.stdout.flush()
        else:
            print self.txt + "... "

    def __exit__(self, type, value, tb):
        if self.newline:
            print self.txt + " done in ",

        print "{:.2f} sec.".format(time()-self.start)


########################
# Activation Functions #
########################

def Identity(x):
    return x


def ReLU(x):
    return T.nnet.relu(x)


def Sigmoid(x):
    return T.nnet.sigmoid(x)


def Softplus(x):
    return T.nnet.softplus(x)


def SoftmaxOld(x):
    return T.nnet.softmax(x)


def Beta_fn(a, b):
    return T.exp(T.gammaln(a) + T.gammaln(b) - T.gammaln(a+b))


def logsumexp(x, axis=None, keepdims=False):
    max_value = T.max(x, axis=axis, keepdims=True)
    res = max_value + T.log(T.sum(T.exp(x-max_value), axis=axis, keepdims=True))
    if not keepdims:
        if axis is None:
            return T.squeeze(res)

        slices = [slice(None, None, None)]*res.ndim
        slices[axis] = 0  # Axis being merged
        return res[tuple(slices)]

    return res


def Softmax(x, axis=None):
    return T.exp(x - logsumexp(x, axis=axis, keepdims=True))


##########
# Layers #
##########

class LSTM_Layer(object):
    def __init__(self, rng, input_size, hidden_size, activation, name="LSTM",
                 W=None, b=None, U=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation_fct = activation

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        # tmp = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_Utmp', values=U)
        # self.U = init_params_zeros(shape=(hidden_size, 4*hidden_size), name=self.name+'_U', values=np.repeat(tmp.get_value(), 4, axis=1))
        # self.U = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_U', values=U, repeat=4)
        self.U = init_params_orthogonal_rbn(rng, shape=(hidden_size, 4*hidden_size), name=self.name+'_U', values=U)
        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        # self.W = init_params_randn(rng, shape=(input_size, 4*hidden_size), name=self.name+'_W', values=W)
        self.W = init_params_orthogonal_rbn(rng, shape=(input_size, 4*hidden_size), name=self.name+'_W', values=W)

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = init_params_zeros(shape=(4*hidden_size,), name=self.name+'_b', values=b)
        tmp = self.b.get_value()
        tmp[hidden_size:2*hidden_size] = 1.
        self.b.set_value(tmp)

        self.betas = init_params_zeros(shape=(hidden_size,), name=self.name+'_betas')

        self.params = [self.W, self.U, self.b, self.betas]

    def _slice(self, x, no):
        if type(no) is str:
            no = ['m', 'f', 'i', 'o'].index(no)
        return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

    def fprop(self, last_h, last_m, X=None, proj_X=None):
        """ Compute the fprop of a LSTM unit.

        If proj_X is provided, X will be ignored. Specifically, proj_X
        represents: T.dot(X, W_in) + b_hid.

        Parameters
        ----------
        last_h : Last hidden state (batch_size, hidden_size)
        last_m : Last memory cell (batch_size, hidden_size)
        X : Input (batch_size, input_size)
        proj_X : Projection of the input plus bias (batch_size, hidden_size)
        """
        if proj_X is None:
            proj_X = self.project_X(X)

        preactivation = proj_X + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(self._slice(preactivation, 'i'))
        mi = self.activation_fct(self._slice(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(self._slice(preactivation, 'f'))
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(self._slice(preactivation, 'o'))
        h = gate_o * self.activation_fct(m + self.betas)

        return h, m

    def project_X(self, X):
        return T.dot(X, self.W) + self.b


class LstmWithPeepholes(object):
    def __init__(self, rng, input_size, hidden_size, activation, name="LSTM",
                 W=None, b=None, U=None, Vi=None, Vo=None, Vf=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation_fct = activation

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        # self.W = init_params_randn(rng, shape=(input_size, 4*hidden_size), name=self.name+'_W', values=W)
        self.W = init_params_orthogonal_rbn(rng, shape=(input_size, 4*hidden_size), name=self.name+'_W', values=W)

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = init_params_zeros(shape=(4*hidden_size,), name=self.name+'_b', values=b)

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        # tmp = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_Utmp', values=U)
        # self.U = init_params_zeros(shape=(hidden_size, 4*hidden_size), name=self.name+'_U', values=np.repeat(tmp.get_value(), 4, axis=1))
        # self.U = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_U', values=U, repeat=4)
        self.U = init_params_orthogonal_rbn(rng, shape=(hidden_size, 4*hidden_size), name=self.name+'_U', values=U)

        # Memory weights (peepholes) (i:input, o:output, f:forget)
        self.Vi = init_params_ones(shape=(hidden_size,), name=self.name+'_Vi', values=Vi)
        self.Vo = init_params_ones(shape=(hidden_size,), name=self.name+'_Vo', values=Vo)
        self.Vf = init_params_ones(shape=(hidden_size,), name=self.name+'_Vf', values=Vf)

        self.params = [self.W, self.U, self.b, self.Vi, self.Vo, self.Vf]

    def _slice(self, x, no):
        if type(no) is str:
            no = ['i', 'o', 'f', 'm'].index(no)
        return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

    def fprop(self, last_h, last_m, X=None, proj_X=None):
        """ Compute the fprop of a LSTM unit.

        If proj_X is provided, X will be ignored. Specifically, proj_X
        represents: T.dot(X, W_in) + b_hid.

        Parameters
        ----------
        last_h : Last hidden state (batch_size, hidden_size)
        last_m : Last memory cell (batch_size, hidden_size)
        X : Input (batch_size, input_size)
        proj_X : Projection of the input plus bias (batch_size, hidden_size)
        """
        if proj_X is None:
            proj_X = self.project_X(X)

        preactivation = proj_X + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(self._slice(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(self._slice(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(self._slice(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(self._slice(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m

    def project_X(self, X):
        return T.dot(X, self.W) + self.b


class FixedLSTM_Layer(object):
    def __init__(self, rng, n_timesteps, input_size, hidden_size, activation, name="LSTM",
                 W=None, b=None, U=None, Vi=None, Vo=None, Vf=None):
        self.input_size = input_size
        self.n_timesteps = n_timesteps
        self.hidden_size = hidden_size
        self.name = name
        self.activation_fct = activation

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        self.W = init_params_randn(rng, shape=(self.n_timesteps, self.input_size, 4*hidden_size), name=self.name+'_W', values=W)

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = init_params_zeros(shape=(self.n_timesteps, 4*hidden_size), name=self.name+'_b', values=b)

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        # tmp = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_Utmp', values=U)
        # self.U = init_params_zeros(shape=(hidden_size, 4*hidden_size), name=self.name+'_U', values=np.repeat(tmp.get_value(), 4, axis=1))
        self.U = init_params_orthogonal(rng, shape=(hidden_size, hidden_size), name=self.name+'_U', values=U, repeat=4)

        # Memory weights (i:input, o:output, f:forget, m:memory)
        self.Vi = init_params_ones(shape=(hidden_size,), name=self.name+'_Vi', values=Vi)
        self.Vo = init_params_ones(shape=(hidden_size,), name=self.name+'_Vo', values=Vo)
        self.Vf = init_params_ones(shape=(hidden_size,), name=self.name+'_Vf', values=Vf)

        self.params = [self.W, self.U, self.b, self.Vi, self.Vo, self.Vf]

    def _slice(self, x, no):
        if type(no) is str:
            no = ['i', 'o', 'f', 'm'].index(no)
        return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

    def fprop(self, last_h, last_m, timestep, X=None, proj_X=None):
        """ Compute the fprop of a LSTM unit.

        If proj_X is provided, X will be ignored. Specifically, proj_X
        represents: T.dot(X, W_in) + b_hid.

        Parameters
        ----------
        last_h : Last hidden state (batch_size, hidden_size)
        last_m : Last memory cell (batch_size, hidden_size)
        X : Input (batch_size, input_size)
        proj_X : Projection of the input plus bias (batch_size, hidden_size)
        """
        if proj_X is None:
            proj_X = self.project_X(X, timestep)

        preactivation = proj_X + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(self._slice(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(self._slice(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(self._slice(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(self._slice(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m

    def project_X(self, X, timestep):
        return T.dot(X, self.W[timestep]) + self.b[timestep]


########################
# Weights initializers #
########################

def guess_init_scale(shape):
    """ Provides appropriate scale for initialization of the weights. """
    if len(shape) == 2:
        # For feedforward networks (see http://deeplearning.net/tutorial/mlp.html#going-from-logistic-regression-to-mlp)
        return np.sqrt(6. / (shape[0] + shape[1]))
    elif len(shape) == 4:
        # For convnet (see http://deeplearning.net/tutorial/lenet.html)
        fan_in = np.prod(shape[1:])
        fan_out = shape[0] * np.prod(shape[2:])
        return np.sqrt(6. / (fan_in + fan_out))
    else:
        raise ValueError("Don't know what to do in this case!")


def init_params_zeros(shape=None, values=None, name=None):
    if values is None:
        values = np.zeros(shape, dtype=theano.config.floatX)

    return sharedX(value=values, name=name)


def init_params_ones(shape=None, values=None, name=None):
    if values is None:
        values = np.ones(shape, dtype=theano.config.floatX)

    return sharedX(value=values, name=name)


def init_params_randn(rng, shape=None, sigma=0.01, values=None, name=None):
    if sigma is None:
        sigma = guess_init_scale(shape)

    if values is None:
        values = sigma * rng.randn(*shape)

    return sharedX(values, name=name)


def init_params_uniform(rng, shape=None, scale=None, values=None, name=None):
    if scale is None:
        scale = guess_init_scale(shape)

    if values is None:
        values = rng.uniform(-scale, scale, shape)

    return sharedX(values, name=name)


def init_params_orthogonal(rng, shape=None, sigma=0.01, values=None, name=None, repeat=1):

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Only works for square matrices")

    if sigma is None:
        sigma = guess_init_scale(shape)

    if values is None:
        values = []

        for i in range(repeat):
            # initialize w/ orthogonal matrix.  code taken from:
            # https://github.com/mila-udem/blocks/blob/master/blocks/initialization.py
            M = np.asarray(rng.standard_normal(size=shape))
            Q, R = np.linalg.qr(M)
            values.append(Q * np.sign(np.diag(R)) * sigma)

        values = np.concatenate(values, axis=1)

    return sharedX(values, name=name)


def init_params_orthogonal_rbn(rng, shape=None, values=None, name=None):
    if values is None:
        # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
        """ benanne lasagne ortho init (faster than qr approach)"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        values = q[:shape[0], :shape[1]]

    return sharedX(values, name=name)


##############
# Optimizers #
##############

def get_adam_updates(cost, params, lr=0.0003, b1=0.95, b2=0.999, e=1e-8):
    """ Adam optimizer
    """
    gradients = dict(zip(params, T.grad(cost, params)))
    return get_adam_updates_from_gradients(gradients, lr=lr, b1=b1, b2=b2, e=e)


def get_adam_updates_from_gradients(gradients, lr=0.0003, b1=0.95, b2=0.999, e=1e-8, return_directions=False):
    """ Adam optimizer

    Parameters
    ----------
    gradients : dict
        Dict object where each entry has key: param, value: gparam.
    """
    directions = []
    updates = OrderedDict()
    i = theano.shared(np.asarray(0., dtype=theano.config.floatX))
    i_t = i + 1.
    updates[i] = i_t
    for p, g in gradients.items():
        m = sharedX(p.get_value()*0., broadcastable=p.broadcastable)
        v = sharedX(p.get_value()*0., broadcastable=p.broadcastable)

        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        m_t_hat = m_t / (1. - b1**i_t)
        v_t_hat = v_t / (1. - b2**i_t)
        p_t = p - lr * m_t_hat / (T.sqrt(v_t_hat) + e)
        updates[m] = m_t
        updates[v] = v_t
        updates[p] = p_t
        directions.append(lr * m_t_hat / (T.sqrt(v_t_hat) + e))

    if return_directions:
        return updates, directions

    return updates


def get_sgd_updates_from_gradients(gradients, lr=0.0003, b1=0.95, b2=0.999, e=1e-8):
    """ SGD optimizer

    Parameters
    ----------
    gradients : dict
        Dict object where each entry has key: param, value: gparam.
    """
    updates = OrderedDict()
    for p, g in gradients.items():
        p_t = p - lr * g
        updates[p] = p_t

    return updates


def calc_directions_norm(directions):
    return T.sqrt(sum(map(lambda d: T.sqr(d).sum(), directions)))


def clip_gradient(gradients, threshold, return_grad_norm=False):
    grad_norm = calc_directions_norm(gradients.values())
    # Note that rescaling is one if grad_norm <= threshold.
    rescaling = threshold / T.maximum(grad_norm, threshold)

    new_gradients = OrderedDict()
    for param, gparam in gradients.items():
        gparam_clipped = gparam * rescaling
        new_gradients[param] = gparam_clipped

    if return_grad_norm:
        return new_gradients, grad_norm

    return new_gradients
