from preprocess import preprocess

from train_baseline import train_baseline
from train_enhanced_1 import train_vgg16
from train_enhanced_2 import train_batch_normalized
from test_baseline import test_model


from settings import set_parameters, NETWORK_TYPE

import time as t
import keras.backend as K


def main():
    batch_size = 32
    epochs = 15
    lr = 0.01
    decay = 1e-6
    vgg16_pretrained = True

    set_parameters(batch_size, epochs, lr, decay)

    network = input("Choose model: 0 for baseline, 1 for enhanced_1(VGG16), 2 for enhanced_2(batch normalization) :")
    if K.image_data_format() != 'channels_last':
        print "This program needs image_data_format to be channels_last please update ~/keras/keras.json"
        return

    if K.backend() != 'tensorflow':
        print "This program has been developed and tested on tensorflow backend. For best performance use tensorflow."
        print "Execution will continue, but problems may occur"

    print "Preprocessing images..."
    program_start = t.clock()
    start = t.clock()
    [X, Y] = preprocess()
    preprocess_time = t.clock()-start

    print "Training Model..."
    start = t.clock()
    if network == 0:
        print 'Using Baseline Network'
        model = train_baseline(X, Y)
    elif network == 1:
        print 'Using VGG16 Network'
        model = train_vgg16(X, Y, vgg16_pretrained)
    elif network == 2:
        print 'Using Baseline Network with Batch Normalization '
        model = train_batch_normalized(X, Y)
    else:
        print 'Invalid network, using baseline instead'
        network = 0
        model = train_baseline(X, Y)

    training_time = t.clock()-start
    print "Testing..."
    start = t.clock()
    accuracy = test_model(model)
    test_time = t.clock()-start
    total_time = t.clock() - program_start

    print "\nSystem Details"
    print "Backend = "+K.backend()
    print "Data format = "+K.image_data_format()
    # TODO: make this dynamic if possible (low priority)
    print "Target = GPU"
    print "Device = GTX 1070"

    print "\nTraining Details"
    print "Network = "+NETWORK_TYPE[network]
    print "Batch size = {}".format(batch_size)
    print "Number of Epochs = {}".format(epochs)
    print "Learning rate = {}".format(lr)
    print "Decay = {}".format(decay)

    print "\nTest accuracy = {:.2%}".format(accuracy)

    print "\nTimings"
    print "Pre-processing Time = {:.2f} seconds".format(preprocess_time)
    print "Training Time = {:.2f} seconds".format(training_time)
    print "Time per epoch = {:.2f} seconds".format(training_time/float(epochs))
    print "Testing Time =  {:.2f} seconds".format(test_time)
    print "Total time of execution = {:.0f} minutes {:.2f} seconds".format(total_time/60, total_time % 60)

    f_name = str(NETWORK_TYPE[network]+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(lr)+'.txt')
    w_file=open('./training_details/'+f_name, 'w')
    w_file.write("System Details")
    w_file.write("\nBackend = "+K.backend())
    w_file.write("\nData format = "+K.image_data_format())
    # TODO: make this dynamic if possible (low priority))
    w_file.write("\nTarget = GPU")
    w_file.write("\nDevice = GTX 1070")

    w_file.write("\n\nTraining Details")
    w_file.write("\nNetwork = "+NETWORK_TYPE[network])
    w_file.write("\nBatch size = {}".format(batch_size))
    w_file.write("\nNumber of Epochs = {}".format(epochs))
    w_file.write("\nLearning rate = {}".format(lr))
    w_file.write("\nDecay = {}".format(decay))
    w_file.write("\nTest accuracy = {:.2%}".format(accuracy))

    w_file.write("\n\nTimings")
    w_file.write("\nPre-processing Time = {:.2f} seconds".format(preprocess_time))
    w_file.write("\nTraining Time = {:.2f} seconds".format(training_time))
    w_file.write("\nTime per epoch = {:.2f} seconds".format(training_time/float(epochs)))
    w_file.write("\nTesting Time =  {:.2f} seconds".format(test_time))
    w_file.write("\nTotal time of execution = {:.0f} minutes {:.2f} seconds".format(total_time/60, total_time % 60))
    w_file.close()

if __name__ == "__main__":
    main()
