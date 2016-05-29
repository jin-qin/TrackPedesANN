import tensorflow as tf
import data_utils as du
import mnist_loader as ml
import conv_net as cnn
import log
import numpy as np
import sys
import os
import gc

#cf = config
#cfc = user config constant => do not change manually
cf_dataset = 1 # 0 => MNIST, 1 => CIFAR-100
cf_timeout_minutes = 60 * 3 - 10 # maximum number of minutes used for training. 0=unlimited
cf_min_max_scaling = True #turn on either this or cf_standardization
cf_standardization = True #turn on either this or cf_min_max_scaling
cf_log_auto_save = True #if True, the log file will be saved automatically as soon as all calculations have been finished correctly
cf_log_dir = cf_log_dir_init = "logs"
cf_batch_size = 100 #must divide number of images in all used datasets
cf_learning_rate = 0.015
cf_size_fully_connected_layer = 1024
cf_num_iters = 30000 #30.000 46 45 45
cf_regularization_strength = 0.1 #0 means no regularization L2
cf_learning_rate_decay = 0.99 #1 means no decay
#cf_verbose = True
#cf_gradient_checking = False
cf_dropout_rate = 0.75 # 1.0 = no dropout
#cf_greedy = True #if false, we will undo changes as soon as we detect that they have decreased the accuracy
cf_optimizer = 0 # 0=GradientDescentOptimizer, 1=AdamOptimizer, 2=MomentumOptimizer(see also cf_momentum)
cf_momentum = 0 #0 deactivates the momentum update. when activating/increasing you may want to decrease cf_learning_rate (the other way around, too).
cf_conv_filter_size = 3 # dim = cf_conv_filter_size x cf_conv_filter_size
cf_number_of_conv_layers = 2
cf_number_of_filters_conv1 = 16 # special number for very first convolution layer
cf_number_of_filters_conv2 = 16 # if cf_number_of_conv_layers > 1, cf_number_of_filters_conv2 will be used for all layers > 1

if cf_dataset == 0:
    cfc_dataset_name = 'MNIST'
else:
    cfc_dataset_name = 'CIFAR-100'


def redo_finalize(Xtrain, Ytrain, Xval, Yval, saveLog):

    global cfc_dataset_name, net

    # if not already done, load test data
    if (cf_dataset != 0):
        Xtest, Ytest = du.load_CIFAR100_test(cfc_datasetpath)
        log.log('Loaded testset, which includes {} images.'.format(Xtest.shape[0]))
        net.preprocessData(Xtest)

    val_acc = net.accuracy(Xval, Yval)
    log.log('FINAL Accuracy validation {0:.3f}%'.format(val_acc * 100))

    test_acc = net.accuracy(Xtest, Ytest)
    log.log('FINAL Accuracy test {0:.3f}%'.format(test_acc * 100))

    train_acc = net.accuracy(Xtrain, Ytrain)
    log.log('FINAL Accuracy training {0:.3f}%'.format(train_acc * 100))

    net.closeSession()
    log.log('.. training finished.')


    runtime = net.getTrainingRuntimeSeconds()
    minutes = np.floor(runtime / 60)
    seconds = round(((runtime / 60) - minutes) * 60, 2)
    runtime_text = "{} minutes and {} seconds".format(minutes, seconds)
    log.log("total runtime used for training: " + runtime_text)

    log.log('########################  END  ################################')
    log.log('###############################################################')

    if saveLog:
        log.logSetName(cfc_dataset_name + '-{}p'.format(round(test_acc * 100, 2)))
        log.logSave(cf_log_dir)



# configurations to automatically evaluate hyperparameters:
param_test_cf_regularization_strength = [
              "cf_regularization_strength", #string name
              [0.25, 0.5, 0.001, 0.002, 0.005, 0.01, 0.1]] #values #0.0001,
param_test_cf_learning_rate = [
    "cf_learning_rate",  # string name
    [0.0001, 0.02, 0.03, 0.04, 0.05, 0.001, 0.01, 0.1, 0.15, 0.2]]  # values
param_test_cf_conv_filter_size = [
    "cf_conv_filter_size",  # string name
    [2, 3, 4, 5]]  # values
param_test_cf_number_of_filters_conv_both = [
    "cf_number_of_filters_conv_both",  # string name
    [8,16,32,64,128]]  # values
param_test_cf_number_of_filters_conv1 = [
    "cf_number_of_filters_conv1",  # string name
    [8,16,32,64,128]]  # values
param_test_cf_number_of_filters_conv2 = [
    "cf_number_of_filters_conv2",  # string name
    [8, 16, 32, 64,128]]  # values
param_test_cf_number_of_conv_layers = [
    "cf_number_of_conv_layers",  # string name
    [2,1,3,4,5,6]]  # values
param_test_cf_dropout_rate = [
    "cf_dropout_rate",  # string name
    [0.25, 0.75,0.5,1.0]]  # values
param_test_cf_optimizer = [
    "cf_optimizer",  # string name
    [1,0,2]]  # values
param_test_cf_size_fully_connected_layer = [
    "cf_size_fully_connected_layer",  # string name
    [256, 512, 1024, 2048]]  # values



no_tests = ["", [0]]
testvals = no_tests
eval_i_max = len(testvals[1])
i=0
while i < eval_i_max: # don't use a for-loop, as we want to manipulate i inside the loop

    try:

        # Loading data
        log.log('###############################################################')
        log.log('########################  BEGIN  ##############################')

        if(testvals[0] != ""):

            log.log('DEBUG param ' + testvals[0] + " ({})".format(testvals[1][i]))
            cf_log_dir = os.path.join(cf_log_dir_init, testvals[0]) # save param evals in extra folder

            if testvals[0] == "cf_regularization_strength":
                cf_regularization_strength = testvals[1][i]
            elif testvals[0] == "cf_learning_rate":
                cf_learning_rate = testvals[1][i]
            elif testvals[0] == "cf_conv_filter_size":
                cf_conv_filter_size = testvals[1][i]
            elif testvals[0] == "cf_number_of_filters_conv_both":
                cf_number_of_filters_conv1 = testvals[1][i]
                cf_number_of_filters_conv2 = testvals[1][i]
            elif testvals[0] == "cf_number_of_filters_conv1":
                cf_number_of_filters_conv1 = testvals[1][i]
            elif testvals[0] == "cf_number_of_filters_conv2":
                cf_number_of_filters_conv2 = testvals[1][i]
            elif testvals[0] == "cf_number_of_conv_layers":
                cf_number_of_conv_layers = testvals[1][i]
            elif testvals[0] == "cf_dropout_rate":
                cf_dropout_rate = testvals[1][i]
            elif testvals[0] == "cf_optimizer":
                cf_optimizer = testvals[1][i]
            elif testvals[0] == "cf_size_fully_connected_layer":
                cf_size_fully_connected_layer = testvals[1][i]



        if len(sys.argv) == 1:
            # use default file path
            cfc_datasetpath = "/media/th/6C4C-2ECD/ml_datasets"
        else:
            #use user specified file path
            cfc_datasetpath = sys.argv[1]

        log.log('Loading ' + cfc_dataset_name + ' dataset..')
        if(cf_dataset == 0):
            cfc_datasetpath = os.path.join(cfc_datasetpath, "mnist")
            (Xtrain, Ytrain, Xval, Yval, Xtest, Ytest) = ml.load_MNIST_data(cfc_datasetpath)
            size_output = 10

            # reshape => unflatten
            mnist_dim = [-1, 28, 28, 1]
            Xtrain = Xtrain.reshape(mnist_dim)
            Xval = Xval.reshape(mnist_dim)
            Xtest = Xtest.reshape(mnist_dim)
        else:
            cfc_datasetpath = os.path.join(cfc_datasetpath, "cifar-100-python")
            Xall, Yall = du.load_CIFAR100_train(cfc_datasetpath)
            size_output = 20

            # resample training data to gain validation dataset
            log.log(".. resampling training and validation data")
            indices = np.random.permutation(Xall.shape[0])
            train_ids, val_ids = indices[:45000], indices[45000:]
            Xtrain, Xval = Xall[train_ids, :], Xall[val_ids, :]
            Ytrain = Yall[train_ids]
            Yval = Yall[val_ids]

            del Xall
            del Yall
            gc.collect()


        log.log('.. Trainingset includes {} images.'.format(Xtrain.shape[0]))

        log.log('.. Validationset includes {} images.'.format(Xval.shape[0]))



        # Creating network
        net = cnn.ConvolutionalNetwork(Xtrain, Ytrain, Xval, Yval,
                                       size_output,
                                       cf_min_max_scaling,
                                       cf_standardization,
                                       cf_batch_size,
                                       cf_learning_rate,
                                       cf_size_fully_connected_layer,
                                       cf_num_iters,
                                       cf_regularization_strength,
                                       cf_learning_rate_decay,
                                       cf_momentum,
                                       cf_timeout_minutes,
                                       cf_conv_filter_size,
                                       cf_number_of_filters_conv1,
                                       cf_number_of_filters_conv2,
                                       cf_log_dir,
                                       cf_number_of_conv_layers,
                                       cf_dropout_rate,
                                       cf_optimizer)

        # Training
        log.log('Start Training..')
        log.log('.. timeout after {} minutes'.format(cf_timeout_minutes))
        log.log('.. learning rate: {}'.format(cf_learning_rate))
        log.log('.. learning rate decay: {}'.format(cf_learning_rate_decay))
        log.log('.. L2 regularization active: {}'.format(cf_regularization_strength != 0))
        if cf_regularization_strength != 0:
            log.log('.. L2 regularization strength: {}'.format(cf_regularization_strength))
        log.log('.. drop out active: {}'.format(cf_dropout_rate > 0 and cf_dropout_rate < 1))
        if cf_dropout_rate > 0 and cf_dropout_rate < 1:
            log.log('.. drop out rate: {}'.format(cf_dropout_rate))
        log.log('.. total number of iterations: {}'.format(cf_num_iters))
        log.log('.. batch size in each iteration: {}'.format(cf_batch_size))
        log.log('.. apply standardization (mean + std): {}'.format(cf_standardization))
        log.log('.. min-max-scaling: {}'.format(cf_min_max_scaling))
        log.log('.. momentum update: {}'.format(cf_momentum))

        log.log('.. number of convolutional layers: {}'.format(cf_number_of_conv_layers))
        log.log('.. number of filters in first convolutional layer: {}'.format(cf_number_of_filters_conv1))
        log.log('.. number of filters in following convolutional layer: {}'.format(cf_number_of_filters_conv2))


        try:
            net.train()
        except KeyboardInterrupt:
            log.log("WARNING: User interrupted progess. Saving latest results.")

            finalizeAndSave = input("Do you want to save the latest data? [y/n]")
            if finalizeAndSave != "n":
                log.log("Saving latest results.")
                redo_finalize(Xtrain, Ytrain, Xval, Yval, True)
                sys.exit()
            else:
                log.log("Results deleted.")

        redo_finalize(Xtrain, Ytrain, Xval, Yval, cf_log_auto_save)

    except Exception:
        log.log("crash detected. auto repairing.. redo.. " + Exception)
        i -= 1 # on error: redo this iteration

    i += 1