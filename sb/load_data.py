import numpy as np
import pickle
import random
import gzip
import zipfile
import tarfile
import os

import theano
import utils


def concatenate_images(images, shape=None, dim=None, border_size=0, clim=(-1, 1)):
    """
    Parameters
    ----------
    images : list of 1D, 2D, 3D arrays
    shape : (height, width)
        Shape of individual image
    dim : tuple (nrows, ncols)
    border_size : int
    """
    if dim is None:
        if type(images) is np.ndarray and images.ndim == 3:
            dim = images.shape[:2]
            images = images.reshape(-1, images.shape[2])
        else:
            dim = (int(np.ceil(np.sqrt(len(images)))), ) * 2

    if shape is None and images[0].ndim == 2:
        shape = images[0].shape

    img_shape = (dim[0] * (shape[0] + 2*border_size)), (dim[1] * (shape[1] + 2*border_size))
    img = np.ones(img_shape, dtype=float) * clim[0]

    for i, image in enumerate(images):
        row = i // dim[1]
        col = i % dim[1]
        starty, endy = row * (shape[0] + 2*border_size), (row+1) * (shape[0] + 2*border_size)
        startx, endx = col * (shape[1] + 2*border_size), (col+1) * (shape[1] + 2*border_size)

        img[starty+border_size:endy-border_size, startx+border_size:endx-border_size] = image.reshape(shape)
        pixels = img[starty+border_size:endy-border_size, startx+border_size:endx-border_size]
        pixels[:, :] = np.maximum(pixels, clim[0])
        pixels[:, :] = np.minimum(pixels, clim[1])

    return img


def display_datasets(datasets, image_shape, display_n_images=64, random=True):
    import pylab as plt

    for i, dataset in enumerate(datasets):
        plt.figure()
        data = dataset[0].get_value()
        nb_images = len(data)
        idx = np.arange(nb_images)
        if random:
            idx = np.random.choice(nb_images, size=display_n_images)

        data = concatenate_images(data[idx].reshape((len(idx), -1)), shape=image_shape, border_size=1, clim=(0, 1))
        plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
        plt.title("Dataset #{} containing {} images".format(i, nb_images))
        #print(dataset[1].get_value()[idx].reshape((8, 8)))

    plt.show()


def _split_data(data, split):
    starts = np.cumsum(np.r_[0, split[:-1]])
    ends = np.cumsum(split)
    splits = [data[s:e] for s, e in zip(starts, ends)]
    return splits


# def _split_into_sequences(dataset, sequence_length):
#     """ Prepare sequences of characters according to `sequence_length`. """
#     # Inspired by https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py#L47
#     # Following https://github.com/johnarevalo/blocks-char-rnn/blob/master/make_dataset.py
#     # Make sure we have the right amount of data (one extra character for the target).
#     if len(dataset) % sequence_length > 0:
#         dataset = dataset[:len(dataset) - len(dataset) % sequence_length + 1]
#     else:
#         dataset = dataset[:len(dataset) - sequence_length + 1]

#     inputs = dataset[:-1].reshape((-1, sequence_length, 1))
#     labels = dataset[1:].reshape((-1, sequence_length, 1))
#     return inputs, labels

def _split_into_sequences(dataset, sequence_length):
    """ Prepare sequences of characters according to `sequence_length`. """
    n_sequences = len(dataset) // sequence_length
    inputs = dataset[:n_sequences*sequence_length].reshape((-1, sequence_length, 1))

    # # Inspired by https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py#L47
    # # Following https://github.com/johnarevalo/blocks-char-rnn/blob/master/make_dataset.py
    # # Make sure we have the right amount of data (one extra character for the target).
    # if len(dataset) % sequence_length > 0:
    #     dataset = dataset[:len(dataset) - len(dataset) % sequence_length + 1]
    # else:
    #     dataset = dataset[:len(dataset) - sequence_length + 1]

    # inputs = dataset[:-1].reshape((-1, sequence_length, 1))
    # labels = dataset[1:].reshape((-1, sequence_length, 1))
    return inputs


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in xrange(0, len(sequence), n):
        yield sequence[i:i + n]


def gen_batches_for_epoch(dataset, sequence_length, batch_size, rng=None, start=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    if start is None:
        start = rng.randint(0, sequence_length)

    inputs = _split_into_sequences(dataset[start:], sequence_length)
    tmp = list(chunk(inputs, n=batch_size))
    # rng.shuffle(tmp)

    for x in tmp:
        yield x


def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, shared_y


def load_mnist(path, target_as_one_hot=False, flatten=False, split=(50000, 10000, 10000)):
    ''' Loads the MNIST dataset.

    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a ndarray of shape (28*28), i.e. (height*width).

    Example labels are integers between [0,9] respresenting one of the ten classes.

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (50000, 10000, 10000)

    References
    ----------
    This dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        mnist_picklefile = os.path.join(data_dir, 'mnist.pkl.gz')

        if not os.path.isfile(mnist_picklefile):
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print("Downloading data (16 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, mnist_picklefile)

        # Load the dataset and process it.
        inputs = []
        labels = []
        print("Processing data ...")
        with gzip.open(mnist_picklefile, 'rb') as f:
            trainset, validset, testset = pickle.load(f)

        inputs = np.concatenate([trainset[0], validset[0], testset[0]], axis=0).reshape((-1, 1, 28, 28))
        labels = np.concatenate([trainset[1], validset[1], testset[1]], axis=0).astype(np.int8)
        np.savez(path, inputs=inputs, labels=labels)

    print("Loading data ...")
    data = np.load(path)
    inputs, labels = data['inputs'], data['labels']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets


def load_mnist_w_rotations(path, target_as_one_hot=False, flatten=False, split=(70000, 10000, 20000)):
    ''' Loads the augmented MNIST dataset containing 50k regular MNIST digits and 50k rotated MNIST digits
    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a ndarray of shape (28*28), i.e. (height*width).

    Example labels are integers between [0,9] respresenting one of the ten classes.
    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (70000, 10000, 20000)

    References
    ----------
    The regular MNIST portion of this dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    The rotated MNIST portion comes from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        mnist_picklefile = os.path.join(data_dir, 'mnist_plus_rot.pkl.gz')

        if not os.path.isfile(mnist_picklefile):
            import urllib
            origin = 'http://www.ics.uci.edu/~enalisni/mnist_plus_rot.pkl.gz'
            print("Downloading data (100 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, mnist_picklefile)

        with gzip.open(mnist_picklefile, 'rb') as f:
            data = pickle.load(f)
        pickle.dump(data, open(os.path.join(data_dir, 'mnist_plus_rot.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    else:
        data = np.load(path)

    inputs, labels = data['inputs'], data['labels']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels.astype(int)] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets


def load_binarized_mnist(path, ordering=None, flatten=False, split=(50000, 10000, 10000)):
    ''' Loads the Binarized MNIST dataset.

    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a 1D array of shape (28*28), i.e. (height*width).

    Example labels are integers between [0,9] respresenting one of the ten classes.

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    ordering : list of int, optional
        If specified, order pixels following the provided list of indices. Default: natural ordering.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (50000, 10000, 10000)

    References
    ----------
    This dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)

        for name, filesize in zip(['train', 'valid', 'test'], [109, 22, 22]):
            mnist_textfile_name = 'mnist_' + name + '.txt'
            mnist_textfile = os.path.join(data_dir, mnist_textfile_name)

            if not os.path.isfile(mnist_textfile):
                import urllib
                origin = 'http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/' + mnist_textfile_name
                print("Downloading data ({} Mb) from {} ...".format(filesize, origin))
                urllib.urlretrieve(origin, mnist_textfile)

        print("Processing data ...")
        train_file, valid_file, test_file = [os.path.join(data_dir, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]
        rng = np.random.RandomState(42)

        def parse_file(filename):
            data = np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])
            data = data[:, :-1]  # Remove target
            data = (data > rng.rand(*data.shape)).astype('int8')
            return data

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        inputs = np.concatenate([trainset, validset, testset], axis=0)
        np.savez(path, inputs=inputs)

    print("Loading data ...")
    data = np.load(path)
    inputs = data['inputs']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    targets = inputs.copy()
    inputs = np.zeros_like(inputs)
    inputs[:, 1:] = targets[:, :-1]

    datasets_inputs = _split_data(inputs, split)
    datasets_targets = _split_data(targets, split)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_targets)]
    return datasets


def load_ocr_letter(path, ordering=None, flatten=False, split=(32152, 10000, 10000)):
    ''' Loads the OCR Letter dataset.

    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a 1D array of shape (28*28), i.e. (height*width).

    Example labels are integers between [0,9] respresenting one of the ten classes.

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    ordering : list of int, optional
        If specified, order pixels following the provided list of indices. Default: natural ordering.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (50000, 10000, 10000)

    References
    ----------
    This dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)

        mnist_gzipfile_name = 'letter.data.gz'
        mnist_gzipfile = os.path.join(data_dir, mnist_gzipfile_name)

        if not os.path.isfile(mnist_gzipfile):
            import urllib
            origin = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
            print("Downloading data (22 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, mnist_gzipfile)

        print("Processing data ...")
        # TODO: refactor the data processing.

        letters = 'abcdefghijklmnopqrstuvwxyz'
        all_data = []
        with gzip.open(mnist_gzipfile, 'rb') as f:
            # Putting all data in memory
            for line in f:
                tokens = line.strip('\n').strip('\t').split('\t')
                s = ''
                for t in range(6, len(tokens)):
                    s = s + tokens[t] + ' '

                target = letters.find(tokens[1])
                if target < 0:
                    print 'Target ' + tokens[1] + ' not found!'

                s = s + str(target) + '\n'
                all_data += [s]

        rng = np.random.RandomState(12345)
        perm = np.arange(len(all_data))
        rng.shuffle(perm)

        data = np.array([map(int, x.split()) for x in all_data])[perm]
        inputs = data[:, :-1]
        labels = data[:, [-1]]
        np.savez(path, inputs=inputs, labels=labels)

    print("Loading data ...")
    data = np.load(path)
    inputs = data['inputs']  # We ignore the labels for now.

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    targets = inputs.copy()
    inputs = np.zeros_like(inputs)
    inputs[:, 1:] = targets[:, :-1]

    datasets_inputs = _split_data(inputs, split)
    datasets_targets = _split_data(targets, split)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_targets)]
    return datasets

def load_svhn_pca(path, target_as_one_hot=False, train_valid_split=(65000, 8254)):
    ''' Loads the Street View House Numbers (SVHN) dataset pre-processed with PCA, reduced to 500 dimensions.
        Example labels are integers between [0,9] respresenting one of the ten classes.
        Parameters
        ----------
        path : str
        The path to the dataset file (.pkl).
        target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
        train_valid_split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (65000, 8254)
        References
        ----------
        The original dataset can be attained at http://ufldl.stanford.edu/housenumbers/
        '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        svhn_picklefile = os.path.join(data_dir, 'svhn_pca.pkl.gz')

        if not os.path.isfile(svhn_picklefile):
            import urllib
            origin = 'http://www.ics.uci.edu/~enalisni/svhn_pca.pkl.gz'
            print("Downloading data (370 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, svhn_picklefile)

        with gzip.open(svhn_picklefile, 'rb') as f:
            data = pickle.load(f)
        pickle.dump(data, open(os.path.join(data_dir, 'svhn_pca.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    else:
        data = pickle.load(open(path,'rb'))

    train_inputs = data['train_data']
    test_inputs = data['test_data']
    train_labels = data['train_labels'][:,0]
    test_labels = data['test_labels'][:,0]

    #shuffle
    idxs = range(train_inputs.shape[0])
    random.shuffle(idxs)
    train_inputs = train_inputs[idxs,:]
    train_labels = train_labels[idxs]

    if target_as_one_hot:
        one_hot_vectors_train = np.zeros((train_labels.shape[0], 10), dtype=theano.config.floatX)
        for idx in xrange(train_labels.shape[0]):
            one_hot_vectors_train[idx, train_labels[idx]] = 1.
        train_labels = one_hot_vectors_train

        one_hot_vectors_test = np.zeros((test_labels.shape[0], 10), dtype=theano.config.floatX)
        for idx in xrange(test_labels.shape[0]):
            one_hot_vectors_test[idx, test_labels[idx]] = 1.
        test_labels = one_hot_vectors_test

    datasets_inputs = [ train_inputs[:train_valid_split[0],:], train_inputs[-1*train_valid_split[1]:,:], test_inputs ]
    datasets_labels = [ train_labels[:train_valid_split[0]], train_labels[-1*train_valid_split[1]:], test_labels ]

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets


def load_penn_treebank(path, sequence_length=100, return_raw=False):
    ''' Loads the Penn Treebank dataset

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    sequence_length : int, optional
        All sequences of characters will have the same length.

    References
    ----------
    This dataset comes from https://github.com/GabrielPereyra/norm-rnn/tree/master/data.

    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        ptb_zipfile = os.path.join(data_dir, 'ptb.zip')

        if not os.path.isfile(ptb_zipfile):
            import urllib
            origin = 'https://www.dropbox.com/s/9hwo2392mfgnnlu/ptb.zip?dl=1'  # Marc's dropbox, TODO: put that somewhere else.
            print("Downloading data (2 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, ptb_zipfile)

        # Load the dataset and process it.
        print("Processing data ...")
        with zipfile.ZipFile(ptb_zipfile) as f:
            train = "\n".join((l.lstrip() for l in f.read('ptb.train.txt').split('\n')))
            valid = "\n".join((l.lstrip() for l in f.read('ptb.valid.txt').split('\n')))
            test = "\n".join((l.lstrip() for l in f.read('ptb.test.txt').split('\n')))

        chars = list(set(train) | set(valid) | set(test))
        data_size = len(train) + len(valid) + len(test)
        vocab_size = len(chars)
        print("Dataset has {:,} characters ({:,} | {:,} | {:,}), {:,} unique.".format(data_size, len(train), len(valid), len(test), vocab_size))

        words = train.split(), valid.split(), test.split()
        n_words = len(words[0]) + len(words[1]) + len(words[2])
        print("Dataset has {:,} words ({:,} | {:,} | {:,}), {:,} unique.".format(n_words, len(words[0]), len(words[1]), len(words[2]), len(set(words[0]) | set(words[1]) | set(words[2]))))
        chr2idx = {c: i for i, c in enumerate(chars)}
        idx2chr = {i: c for i, c in enumerate(chars)}

        train = np.array([chr2idx[c] for c in train], dtype=np.int8)
        valid = np.array([chr2idx[c] for c in valid], dtype=np.int8)
        test = np.array([chr2idx[c] for c in test], dtype=np.int8)

        np.savez(path,
                 train=train, valid=valid, test=test,
                 chr2idx=chr2idx, idx2chr=idx2chr)

    print("Loading data ...")
    ptb = np.load(path)
    if return_raw:
        return (ptb['train'], ptb['valid'], ptb['test']), ptb['idx2chr'].item()

    # datasets = [_shared_dataset(_split_into_sequences(d, sequence_length)) for d in [ptb['train'], ptb['valid'], ptb['test']]]
    datasets = [utils.sharedX(_split_into_sequences(d, sequence_length)) for d in [ptb['train'], ptb['valid'], ptb['test']]]
    return datasets, ptb['idx2chr'].item()


def load_cifar10(path, target_as_one_hot=False, flatten=False, split=(45000, 5000, 10000)):
    ''' Loads the CIFAR-10 dataset.

    Input examples are 32x32 pixels RGB images. Each input example is represented
    as an ndarray of shape (3, 32, 32), i.e. (n_channels, height, width). Pixel values
    are floats between [0,1].

    Example labels are integers between [0,9] respresenting one of the ten classes.

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (45000, 5000, 10000)

    Returns
    -------
    tuple of dataset splits
        The number of dataset splits depends on the parameter `split`. Each dataset
        split is a 2-tuple of ndarrays: (input, labels).

    References
    ----------
    This dataset comes from https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        cifar10_tarfile = os.path.join(data_dir, 'cifar-10-python.tar.gz')

        if not os.path.isfile(cifar10_tarfile):
            import urllib
            origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            print("Downloading data (163 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, cifar10_tarfile)

        # Load the dataset and process it.
        inputs = []
        labels = []
        print("Processing data ...")
        with tarfile.open(cifar10_tarfile) as f:
            # Read training/validation images and their labels.
            for i in range(1, 5+1):
                d = pickle.loads(f.extractfile("cifar-10-batches-py/data_batch_{}".format(i)).read())

                inputs.append(d['data'].reshape((-1, 3, 32, 32)))  # 32x32 pixels RGB images
                labels.append(d['labels'])

            # Read testing images and their labels.
            d = pickle.loads(f.extractfile("cifar-10-batches-py/test_batch").read())
            inputs.append(d['data'].reshape((-1, 3, 32, 32)))  # 32x32 pixels RGB images
            labels.append(d['labels'])

            # Get the label_names.
            d = pickle.loads(f.extractfile("cifar-10-batches-py/batches.meta").read())
            label_names = d['label_names']

        n_images = [len(e) for e in inputs]
        print("Dataset has {:,} images and {:,} classes.".format(sum(n_images), len(label_names)))
        np.savez(path,
                 inputs=np.concatenate(inputs, axis=0)/255., labels=np.concatenate(labels, axis=0),
                 label_names=label_names)

    print("Loading data ...")
    data = np.load(path)
    inputs, labels = data['inputs'], data['labels']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels.astype(int)] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets


def load_text8(path, sequence_length=180, split=(90000000, 5000000, 5000000)):
    ''' Loads the Text8 dataset.

    This dataset is derived from Wikipedia and consists of a sequence of 100M
    characters (only alphabetical and spaces). A character is represented as
    a number between 0 and 26 inclusively (i.e. 0 denotes a space).

    Conversion functions `chr2idx` and `idx2char` can be found in `sb.utils`.

    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    sequence_length : int, optional
        All sequences of characters will have the same length.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (90M, 5M, 5M)

    Returns
    -------
    tuple of dataset splits
        The number of dataset splits depends on the parameter `split`. Each
        dataset split is a ndarray.

    References
    ----------
    This dataset comes from http://mattmahoney.net/dc/text8.zip
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        data_zipfile = os.path.join(data_dir, 'text8.zip')

        if not os.path.isfile(data_zipfile):
            import urllib
            origin = 'http://mattmahoney.net/dc/text8.zip'
            print("Downloading data (30 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, data_zipfile)

        # Load the dataset and process it.
        inputs = []
        print("Processing data ...")
        with zipfile.ZipFile(data_zipfile) as f:
            text = f.read('text8')
            text = np.array([utils.chr2idx[c] for c in text], dtype=np.int8)

        np.savez(path,
                 inputs=text,
                 chr2idx=utils.chr2idx, idx2chr=utils.idx2chr,
                 vocabulary_size=len(utils.chr2idx))

    print("Loading data ...")
    data = np.load(path)
    inputs = data['inputs']

    datasets_inputs = _split_data(inputs, split)
    datasets = [_shared_dataset(_split_into_sequences(d, sequence_length)) for d in datasets_inputs]

    return datasets, data['idx2chr'].item()
