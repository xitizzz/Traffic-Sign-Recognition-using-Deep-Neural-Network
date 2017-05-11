NUM_CLASSES = 43
IMG_SIZE = 48

batch_size = 48
epochs = 10
lr = 0.01
decay = 1e-6

NETWORK_TYPE = ['baseline', 'vgg16', 'baseline with batch normalization']


def set_parameters(batch_size_p, epochs_p, lr_p, decay_p):
    global batch_size, epochs, lr, decay
    batch_size = batch_size_p
    epochs = epochs_p
    lr = lr_p
    decay = decay_p
