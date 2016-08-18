"""
References
----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016)
        "Identity Mappings in Deep Residual Networks"
        http://arxiv.org/abs/1603.05027
    [2] He, K., Zhang, X., Ren, S., & Sun, J. (2015)
        "Deep Residual Learning for Image Recognition"
        http://arxiv.org/abs/1512.03385
"""
from __future__ import division

import numpy as np
import cPickle
import argparse
import os
import sys
import time
from os.path import join as pjoin
from collections import OrderedDict

import theano
import theano.tensor as T
import theano.printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except:
    print "* Cannot find cudNN for Conv2DDNNLayer. Falling back to Theano's implementation."
    from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne.layers import ConcatLayer, ElemwiseSumLayer
from lasagne.layers import FlattenLayer, SliceLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import NonlinearityLayer, ExpressionLayer
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import softmax, softplus, rectify as ReLU

from load_data import load_mnist, load_cifar10, load_mnist_w_rotations
import utils
from utils import mkdirs

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)
_srng = RandomStreams(1234)


def bound(layer, min_value, max_value):
    return ExpressionLayer(layer, lambda X: T.minimum(max_value, T.maximum(min_value, X)))


class RemainingStickLengthLayer(lasagne.layers.MergeLayer):
    def __init__(self, layer, stick, kumar_parameters=lasagne.init.Normal(0.0001), **kwargs):
        flatten_layer = FlattenLayer(layer, outdim=2)
        kumar_parameters = DenseLayer(flatten_layer, 2, W=kumar_parameters, b=None, nonlinearity=softplus)
        self.kumar_a = SliceLayer(kumar_parameters, indices=slice(0, 1), axis=1)  # Equivalent to [:, [0]]
        self.kumar_b = SliceLayer(kumar_parameters, indices=slice(1, 2), axis=1)  # Equivalent to [:, [1]]

        # Bound Kumaraswamy's parameters: 1e-6 <= a, b <= 30.
        self.kumar_a = bound(self.kumar_a, min_value=1e-6, max_value=30)
        self.kumar_b = bound(self.kumar_b, min_value=1e-6, max_value=30)
        super(RemainingStickLengthLayer, self).__init__([layer, stick, self.kumar_a, self.kumar_b], **kwargs)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        if False and deterministic:  # TODO
            raise NotImplementedError("What should we do at testing? Check Eric's draft.")
        else:
            input = inputs[0]
            stick = inputs[1]
            kumar_a = inputs[2]  # self.kumar_a.get_output_for(input)
            kumar_b = inputs[3]  # self.kumar_b.get_output_for(input)

            # Compute kumaraswamy sample.
            u = _srng.uniform(size=(input.shape[0], 1), low=0.01, high=0.99)
            v = (1-(u**(1/kumar_b)))**(1/kumar_a)

            # Compute remaining stick length.
            remaining_stick = (1-v) * stick
            return remaining_stick

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 1)


class WeightedByStickLengthLayer(lasagne.layers.MergeLayer):
    def __init__(self, layer, remaining_stick, **kwargs):
        super(WeightedByStickLengthLayer, self).__init__([layer, remaining_stick], **kwargs)

    def get_output_for(self, inputs, **kwargs):
        layer = inputs[0]
        remaining_stick = inputs[1]
        return remaining_stick.reshape((-1,) + (1,)*(layer.ndim-1)) * layer

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


def build_bottleneck_sb_residual_layer(prev_layer, n_out, stride, remaining_stick):
    size = n_out / 4

    # conv 1x1
    prev_layer_preact = batch_norm(prev_layer)
    prev_layer_preact = NonlinearityLayer(prev_layer_preact, nonlinearity=ReLU)
    layer = ConvLayer(prev_layer_preact, num_filters=size, nonlinearity=None,
                      filter_size=(1, 1), stride=stride, pad=(0, 0),
                      W=lasagne.init.HeNormal(gain='relu'))

    # conv 3x3
    layer = batch_norm(layer)
    layer = NonlinearityLayer(layer, nonlinearity=ReLU)
    layer = ConvLayer(layer, num_filters=size, nonlinearity=None,
                      filter_size=(3, 3), stride=(1, 1), pad=(1, 1),
                      W=lasagne.init.HeNormal(gain='relu'))

    # conv 1x1
    layer = batch_norm(layer)
    layer = NonlinearityLayer(layer, nonlinearity=ReLU)
    layer = ConvLayer(layer, num_filters=n_out, nonlinearity=None,
                      filter_size=(1, 1), stride=(1, 1), pad=(0, 0),
                      W=lasagne.init.HeNormal(gain='relu'))

    # Weigh layer by the remaining stick length.
    remaining_stick = RemainingStickLengthLayer(prev_layer, remaining_stick)
    layer = WeightedByStickLengthLayer(layer, remaining_stick)

    if prev_layer.output_shape[1] == n_out:
        shortcut_layer = prev_layer  # Identity shortcut.
    else:
        # Projection shortcut.
        shortcut_layer = ConvLayer(prev_layer_preact, num_filters=n_out, nonlinearity=None,
                                   filter_size=(1, 1), stride=stride, pad=(0, 0),
                                   W=lasagne.init.HeNormal(gain='relu'))

    output_layer = ElemwiseSumLayer([layer, shortcut_layer])
    return output_layer, remaining_stick


def build_sb_resnet_phase(prev_layer, n_out, count, stride):

    remaining_sticks = []
    # Initial stick length is 1.
    stick = ExpressionLayer(prev_layer, function=lambda X: T.ones((X.shape[0], 1)), output_shape=(None, 1))
    layer, remaining_stick = build_bottleneck_sb_residual_layer(prev_layer, n_out, stride, stick)
    remaining_sticks.append(remaining_stick)
    for _ in range(count-1):
        layer, remaining_stick = build_bottleneck_sb_residual_layer(layer, n_out, stride=(1, 1), remaining_stick=remaining_stick)
        remaining_sticks.append(remaining_stick)

    # Compute posteriors
    posterior_a = ConcatLayer([_remaining_stick.kumar_a for _remaining_stick in remaining_sticks], axis=1)
    posterior_b = ConcatLayer([_remaining_stick.kumar_b for _remaining_stick in remaining_sticks], axis=1)
    stick_lengths = ConcatLayer(remaining_sticks, axis=1)
    return layer, (posterior_a, posterior_b, stick_lengths)


def build_sb_resnet(input_layer, depth, output_size):
    """
    References
    ----------
    [1] https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    [2] https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
    """
    if (depth-2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (e.g. 164 or 1001 in He et al. (2016)")

    stages_sizes = [16, 64, 128, 256]
    depth_per_phase = (depth-2) // 9

    layer = ConvLayer(input_layer, num_filters=stages_sizes[0], nonlinearity=None,
                      filter_size=(3, 3), stride=(1, 1), pad=(1, 1),
                      W=lasagne.init.HeNormal())
    layer, infos1 = build_sb_resnet_phase(layer, stages_sizes[1], depth_per_phase, stride=(1, 1))
    layer, infos2 = build_sb_resnet_phase(layer, stages_sizes[2], depth_per_phase, stride=(2, 2))
    layer, infos3 = build_sb_resnet_phase(layer, stages_sizes[3], depth_per_phase, stride=(2, 2))
    layer = batch_norm(layer)
    layer = NonlinearityLayer(layer, nonlinearity=ReLU)

    layer = GlobalPoolLayer(layer, pool_function=T.mean)  # Average pooling.

    # fully connected layer
    network = DenseLayer(layer, num_units=output_size, W=lasagne.init.HeNormal(), nonlinearity=softmax)
    return network, (infos1, infos2, infos3)


def calc_kl_divergence(infos, alpha, beta):
    posterior_a = lasagne.layers.get_output(infos[0])
    posterior_b = lasagne.layers.get_output(infos[1])
    return utils.calc_kl_divergence(posterior_a, posterior_b, alpha, beta)


def calc_avg_n_layers(infos, epsilon=0.01):
    stick_lengths = lasagne.layers.get_output(infos[2])
    nb_layers = T.sum(T.cast(stick_lengths >= epsilon, dtype=theano.config.floatX), axis=1) * 3  # There are 3 convolution layers per residual unit.
    return T.mean(nb_layers)


########################
# Train & Evaluate Net #
########################
def run_ResNet(dataset,
               depth,
               n_epochs, batch_size, lookahead, alpha0,
               experiment_dir,
               epsilon,
               random_seed,
               output_file_base_name,
               gradient_clipping=None,
               force=False,
               n_validation_resamples=3., n_test_resamples=5.):

    # LOAD DATA
    if "mnist_plus_rot" in dataset:
        datasets = load_mnist_w_rotations(dataset, flatten=False, split=(70000, 10000, 20000))
        dataset_name = "mnist_w_rotation"
        input_layer = InputLayer(shape=(None, 1, 28, 28))
        output_size = 10

    elif "mnist" in dataset:
        # We follow the approach used in [2] to split the MNIST dataset.
        datasets = load_mnist(dataset, flatten=False, split=(45000, 5000, 10000))
        dataset_name = "mnist"
        input_layer = InputLayer(shape=(None, 1, 28, 28))
        output_size = 10

    elif "cifar10" in dataset:
        # We split the Cifar-10 dataset according to [2].
        datasets = load_cifar10(dataset, flatten=False, split=(45000, 5000, 10000))
        dataset_name = "cifar10"
        input_layer = InputLayer(shape=(None, 3, 32, 32))
        output_size = 10

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    train_set_size = int(train_set_y.shape[0].eval())
    valid_set_size = int(valid_set_y.shape[0].eval())
    test_set_size = int(test_set_y.shape[0].eval())
    print 'Dataset {} loaded ({:,}|{:,}|{:,})'.format(dataset_name, train_set_size, valid_set_size, test_set_size)

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(np.ceil(train_set_size / batch_size))
    n_valid_batches = int(np.ceil(valid_set_size / batch_size))
    n_test_batches = int(np.ceil(test_set_size / batch_size))

    # BUILD MODEL
    print 'Building the model ...'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    index.tag.test_value = 0

    # epoch = T.scalar()
    x = T.tensor4('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of [floatX] labels.

    # Test values are useful for debugging with THEANO_FLAGS="compute_test_value=warn"
    x.tag.test_value = train_set_x[:batch_size].eval()
    y.tag.test_value = train_set_y[:batch_size].eval()

    input_layer.input_var = x
    layers_per_phase = ((depth-2) // 9) * 3
    network, infos = build_sb_resnet(input_layer, depth, output_size)
    print "Number of parameters in model: {:,}".format(lasagne.layers.count_params(network, trainable=True))

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    ll_term = lasagne.objectives.categorical_crossentropy(prediction, T.cast(y, dtype="int32"))
    kl_term_1 = calc_kl_divergence(infos[0], alpha=1., beta=alpha0)
    kl_term_2 = calc_kl_divergence(infos[1], alpha=1., beta=alpha0)
    kl_term_3 = calc_kl_divergence(infos[2], alpha=1., beta=alpha0)
    kl_term = kl_term_1 + kl_term_2 + kl_term_3
    cost = T.mean(ll_term + kl_term)

    # Compute average number of layers that have a stick length >= 1% in each phase.
    avg_n_layers_phase1 = calc_avg_n_layers(infos[0])
    avg_n_layers_phase2 = calc_avg_n_layers(infos[1])
    avg_n_layers_phase3 = calc_avg_n_layers(infos[2])
    avg_kl_term_1 = T.mean(kl_term_1)
    avg_kl_term_2 = T.mean(kl_term_2)
    avg_kl_term_3 = T.mean(kl_term_3)

    # Build the expresson for the cost function.
    params = lasagne.layers.get_all_params(network, trainable=True)

    # If params already exist and 'force' is False, reload parameters.
    params_pkl_filename = pjoin(experiment_dir, 'conv_sb-resnet_params_' + output_file_base_name + '.pkl')
    print "Checking if '{}' already exists.".format(params_pkl_filename)
    if os.path.isfile(params_pkl_filename) and not force:
        print "Yes! Reloading existing parameters and resuming training (use --force to overwrite)."
        last_params = cPickle.load(open(params_pkl_filename, 'rb'))
        for param, last_param in zip(params, last_params):
            param.set_value(last_param)
    elif force:
        print "Yes! but --force was used. Starting from scratch."
    else:
        print "No! Starting from scratch."

    gradients = dict(zip(params, T.grad(cost, params)))

    if gradient_clipping is not None:
        grad_norm = T.sqrt(sum(map(lambda d: T.sqr(d).sum(), gradients.values())))
        # Note that rescaling is one if grad_norm <= threshold.
        rescaling = gradient_clipping / T.maximum(grad_norm, gradient_clipping)

        new_gradients = OrderedDict()
        for param, gparam in gradients.items():
            gparam_clipped = gparam * rescaling
            new_gradients[param] = gparam_clipped

        gradients = new_gradients

    updates = utils.get_adam_updates_from_gradients(gradients)

    # Compile theano function for training. This updates the model parameters and
    # returns the training nll term, kl term, and the avg. nb. of layers used in each phase.
    print 'Compiling train function ...'
    compiling_start = time.time()
    train_model = theano.function(inputs=[index], outputs=[ll_term.mean(), kl_term.mean(),
                                                           avg_n_layers_phase1, avg_n_layers_phase2, avg_n_layers_phase3,
                                                           avg_kl_term_1, avg_kl_term_2, avg_kl_term_3],
                                  updates=updates,
                                  givens={x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    print "{:.2f}".format((time.time()-compiling_start)/60.)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, T.cast(y, dtype="int32"))
    test_loss = test_loss.mean()
    test_error = T.sum(T.neq(T.argmax(test_prediction, axis=1), y), dtype=theano.config.floatX)

    print 'Compiling valid function ...'
    compiling_start = time.time()
    valid_model = theano.function(inputs=[index],
                                  outputs=[test_loss, test_error],
                                  givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    print "{:.2f}".format((time.time()-compiling_start)/60.)

    print 'Compiling test function ...'
    compiling_start = time.time()
    test_model = theano.function(inputs=[index],
                                 outputs=[test_loss, test_error],
                                 givens={x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    print "{:.2f}".format((time.time()-compiling_start)/60.)

    ###############
    # TRAIN MODEL #
    ###############
    print 'Training for {} epochs ...'.format(n_epochs)

    best_params = None
    best_valid_error = np.inf
    best_iter = 0
    start_time = time.clock()

    results_filename = pjoin(experiment_dir, "conv_sb-resnet_results_" + output_file_base_name + ".txt")
    if os.path.isfile(results_filename) and not force:
        last_result = open(results_filename, 'rb').readlines()[-1]
        idx_start = len("epoch ")
        idx_end = last_result.find(",", idx_start+1)
        start_epoch = int(last_result[idx_start:idx_end]) + 1
        results_file = open(results_filename, 'ab')
    else:
        start_epoch = 0
        results_file = open(results_filename, 'wb')

    stop_training = False
    for epoch_counter in range(start_epoch, n_epochs):
        if stop_training:
            break

        # Train this epoch
        epoch_start_time = time.time()
        avg_training_loss_tracker = 0.
        avg_training_kl_tracker = 0.
        avg_n_layers_phase1_tracker = 0.
        avg_n_layers_phase2_tracker = 0.
        avg_n_layers_phase3_tracker = 0.
        avg_kl_term_1_tracker = 0.
        avg_kl_term_2_tracker = 0.
        avg_kl_term_3_tracker = 0.

        for minibatch_index in xrange(n_train_batches):
            avg_training_loss, avg_training_kl, avg_n_layers_phase1, avg_n_layers_phase2, avg_n_layers_phase3, avg_kl_term_1, avg_kl_term_2, avg_kl_term_3 = train_model(minibatch_index)
            if minibatch_index % 1 == 0:
                results = "batch #{}-{}, avg n_layers per phase ({:.2f}|{:.2f}|{:.2f})/{}, training loss (nll) {:.4f}, training kl-div {:.4f} ({:.4f}|{:.4f}|{:.4f}), time {:.2f}m"
                results = results.format(epoch_counter, minibatch_index,
                                         float(avg_n_layers_phase1), float(avg_n_layers_phase2), float(avg_n_layers_phase3), layers_per_phase,
                                         float(avg_training_loss),
                                         float(avg_training_kl), float(avg_kl_term_1), float(avg_kl_term_2), float(avg_kl_term_3),
                                         (time.time()-epoch_start_time)/60.)
                print results

            if np.isnan(avg_training_loss):
                msg = "NaN detected! Stopping."
                print msg
                results_file.write(msg + "\n")
                results_file.flush()
                sys.exit(1)

            avg_training_loss_tracker += avg_training_loss
            avg_training_kl_tracker += avg_training_kl
            avg_n_layers_phase1_tracker += avg_n_layers_phase1
            avg_n_layers_phase2_tracker += avg_n_layers_phase2
            avg_n_layers_phase3_tracker += avg_n_layers_phase3
            avg_kl_term_1_tracker += avg_kl_term_1
            avg_kl_term_2_tracker += avg_kl_term_2
            avg_kl_term_3_tracker += avg_kl_term_3

        epoch_end_time = time.time()

        # Compute some infos about training.
        avg_training_loss_tracker /= n_train_batches
        avg_training_kl_tracker /= n_train_batches
        avg_n_layers_phase1_tracker /= n_train_batches
        avg_n_layers_phase2_tracker /= n_train_batches
        avg_n_layers_phase3_tracker /= n_train_batches
        avg_kl_term_1_tracker /= n_train_batches
        avg_kl_term_2_tracker /= n_train_batches
        avg_kl_term_3_tracker /= n_train_batches

        # Compute validation error --- sample multiple times to simulate posterior predictive distribution
        valid_errors = np.zeros((n_valid_batches,))
        valid_loss = np.zeros((n_valid_batches,))
        for idx in xrange(int(n_validation_resamples)):
            temp_valid_loss, temp_valid_errors = zip(*[valid_model(i) for i in xrange(n_valid_batches)])
            valid_errors += temp_valid_errors
            valid_loss += temp_valid_loss
        valid_loss = np.sum(valid_loss/n_validation_resamples) / n_valid_batches
        valid_nb_errors = np.sum(valid_errors/n_validation_resamples)
        valid_error = valid_nb_errors / valid_set_size

        results = ("epoch {}, avg n_layers per phase ({:.2f}|{:.2f}|{:.2f})/{}, train loss (nll) {:.4f}, "
                   "train kl-div {:.4f}, train kl-div per phase ({:.4f}|{:.4f}|{:.4f}), "
                   "valid loss {:.4f}, valid error {:.2%} ({:,}), time {:.2f}m")

        if valid_error < best_valid_error:
            best_iter = epoch_counter
            best_valid_error = valid_error
            results += " **"
            # Save progression
            best_params = [param.get_value().copy() for param in params]
            cPickle.dump(best_params, open(params_pkl_filename, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        elif epoch_counter-best_iter > lookahead:
            stop_training = True

        # Report and save progress.
        results = results.format(epoch_counter,
                                 avg_n_layers_phase1_tracker, avg_n_layers_phase2_tracker, avg_n_layers_phase3_tracker, layers_per_phase,
                                 avg_training_loss_tracker,
                                 avg_training_kl_tracker, avg_kl_term_1_tracker, avg_kl_term_2_tracker, avg_kl_term_3_tracker,
                                 valid_loss, valid_error, valid_nb_errors,
                                 (epoch_end_time-epoch_start_time)/60)
        print results

        results_file.write(results + "\n")
        results_file.flush()

    end_time = time.clock()

    # Reload best model.
    for param, best_param in zip(params, best_params):
        param.set_value(best_param)

    # Compute test error --- sample multiple times to simulate posterior predictive distribution
    test_errors = np.zeros((n_test_batches,))
    test_loss = np.zeros((n_test_batches,))
    for idx in xrange(int(n_test_resamples)):
        temp_test_loss, temp_test_errors = zip(*[test_model(i) for i in xrange(n_test_batches)])
        test_errors += temp_test_errors
        test_loss += temp_test_loss
    test_loss = np.sum(test_loss/n_test_resamples) / n_test_batches
    test_nb_errors = np.sum(test_errors/n_test_resamples)
    test_error = test_nb_errors / test_set_size

    results = "Done! best epoch {}, test loss {:.4f}, test error {:.2%} ({:,}), training time {:.2f}m"
    results = results.format(best_iter, test_loss, test_error, test_nb_errors, (end_time-start_time)/60)
    print results

    results_file.write(results + "\n")
    results_file.close()

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))


def build_argparser():
    DESCRIPTION = ("Train a Convolutional SB-ResNet using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    dataset = p.add_argument_group("Experiment options")
    dataset.add_argument('--dataset', default="mnist", choices=["mnist", "mnist_plus_rot", "cifar10"],
                         help='either mnist or mnist_plus_rot or cifar10. Default:%(default)s')

    dataset = p.add_argument_group("Model options")
    dataset.add_argument('--depth', type=int, default=164,
                         help='depth of the Residual Netowrk (usually 164 or 1001 in the paper). Default:%(default)s')
    dataset.add_argument('--prior-concentration-param', type=float, default=1.,
                         help="the Beta prior's concentration parameter: v ~ Beta(1, alpha0). The larger the alpha0, the deeper the net. Default:%(default)s")

    duration = p.add_argument_group("Training duration options")
    duration.add_argument('--max-epoch', type=int, metavar='N', default=1000,
                          help='train for a maximum of N epochs. Default: %(default)s')
    duration.add_argument('--lookahead', type=int, metavar='K', default=10,
                          help='use early stopping with a lookahead of K. Default: %(default)s')

    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, default=128,
                          help='size of the batch to use when training the model. Default: %(default)s.')
    training.add_argument('--clip-gradient', type=float,
                          help='if provided, gradient norms will be clipped to this value (if it exceeds it).')

    # optimizer = p.add_argument_group("Optimizer (required)")
    # optimizer = optimizer.add_mutually_exclusive_group(required=True)
    # optimizer.add_argument('--Adam', metavar="[LR=0.0003]", type=str, help='use Adam for training.')

    general = p.add_argument_group("General arguments")
    general.add_argument('--experiment-dir', default="./experiments/sb_resnet",
                         help='name of the folder where to save the experiment. Default: %(default)s.')

    general.add_argument('-f', '--force', action="store_true",
                         help='if specified, it will overwrite the experiment if it already exists.')

    return p


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    args_dict = vars(args)
    args_string = ''.join('{}_{}_'.format(key, val) for key, val in sorted(args_dict.items()) if key not in ['experiment_dir'])

    # Make sure our results are reproducible.
    lasagne.random.set_rng(np.random.RandomState(42))

    # Check the parameters are correct.
    if (args.depth-2) % 9 != 0:
        raise ValueError("Depth of this network should be 9*n+2 where n is the number of desired residual blocks.")

    print "Using Theano v.{}".format(theano.version.short_version)

    # LEARNING PARAMS
    epsilon = 0.01
    batch_size = args.batch_size

    # Create datasets and experiments folders is needed.
    dataset_dir = mkdirs("./datasets")
    mkdirs(args.experiment_dir)

    print "Datasets dir: {}".format(os.path.abspath(dataset_dir))
    print "Experiment dir: {}".format(os.path.abspath(args.experiment_dir))

    if args.dataset == 'mnist_plus_rot':
        dataset = pjoin(dataset_dir, args.dataset + ".pkl")
    else:
        dataset = pjoin(dataset_dir, args.dataset + ".npz")

    run_ResNet(dataset=dataset,
               depth=args.depth,
               n_epochs=args.max_epoch,
               batch_size=args.batch_size,
               lookahead=args.lookahead,
               alpha0=args.prior_concentration_param,
               experiment_dir=args.experiment_dir,
               epsilon=epsilon,
               gradient_clipping=args.clip_gradient,
               output_file_base_name=args_string,
               random_seed=1234,
               force=args.force)
