import tensorflow as tf
import log
import numpy as np
import math
import time
import os

class ConvolutionalNetwork:
    def __init__(self, XtrainPrevious, XtrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval, min_max_scaling=True, standardization=True,
                 batch_size=200,
                 learning_rate=0.01,
                 size_full=300,
                 iterations=3000,
                 regularization_strength=0.01,
                 learning_rate_decay=0.96,
                 momentum=0,
                 timeout_minutes=0,
                 conv_filter_size=5,
                 number_of_filters_conv1=32,
                 number_of_filters_conv2=32,
                 log_dir="logs",
                 number_of_conv_layers=2,
                 dropout_rate=0.5,
                 optimizer=0):

        log.log('Creating network..')

        # save given params
        self.XtrainPrevious = XtrainPrevious
        self.XtrainCurrent = XtrainCurrent
        self.Ytrain = Ytrain
        self.XvalPrevious = XvalPrevious
        self.XvalCurrent = XvalCurrent
        self.Yval = Yval
        self.batch_size = batch_size
        self.starter_learning_rate = learning_rate
        self.size_full = size_full
        self.iterations = iterations
        self.regularization_strength = regularization_strength
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.timeout_minutes = timeout_minutes # maximum number of minutes used for training. 0=unlimited
        self.timeout_seconds = self.timeout_minutes * 60
        self.conv_filter_size = conv_filter_size
        self.number_of_filters_conv1 = number_of_filters_conv1
        self.number_of_filters_conv2 = number_of_filters_conv2
        self.log_dir = log_dir
        self.number_of_conv_layers = number_of_conv_layers
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer

        # derive further params

        # use original image shape, but resize the number of images to a single batch
        self.size_input = []
        orig_img_shape = tf.TensorShape(self.XtrainPrevious.shape).as_list()
        for i in range(len(orig_img_shape)):
            if i > 0:
                self.size_input.append(orig_img_shape[i])
            else:
                self.size_input.append(self.batch_size)

        log.log('.. Input dimension: {}.'.format(self.size_input))
        log.log( '.. Fully-connected layer dimension: {}.'.format(self.size_full) )
        # log.log( '.. Standard deviation W-init: {}.'.format(cf_standard_deviation_w_init) )

        # preprocessing (will change original data, too!)
        log.log(".. preprocessing data") #TODO add self.XtrainCurrent and XvalCurrent
        self.preprocessInit(min_max_scaling, standardization, self.XtrainPrevious)  # call this independently of the values of min_max_scaling or standardization!
        if min_max_scaling or standardization:
            self.preprocessData(self.XtrainPrevious)
            self.preprocessData(self.XvalPrevious)
            #self.preprocessData(Xtest)
        log.log(".. finished preprocessing data")


    def conv2d(self, x, W):
        # keep strides=[1, 1, 1, 1] to copy filter to all possible positions.
        # parameters got nothing to do with the size of the used neighborhood
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

    def vars_W_b(self, shape_W):
        W_conv1 = tf.Variable(tf.truncated_normal(shape=shape_W, stddev=1.0 / math.sqrt(float(shape_W[0]))))
        b_conv1 = tf.Variable(tf.constant(0.1, tf.float32, [shape_W[-1]]))
        return W_conv1, b_conv1

    def train(self):

        # runtime evaluation
        self.runtime_training_start = time.time()


        ## NEW TRACKING BEGIN ####################################################


        # input data: two batches of images. equal indices of the two batches must form valid pairs.
        # e.g. x_previous[i] and x_current[i] a one valid frame pair. Each image has the size = 48 x 128.
        # Each pixel got 5 channels: R, G, B, Dx, Dy
        # dims(self.x_previous) = dims(self.x_current) = <batch_size>, <image_height=128>, <image_width=48>, <channels=5>
        # reminder: "image is flipped": the image format is the one of opencv
        #       => <number_of_rows> x <number_of_cols> = 128 x 48
        #       => this is exactly what we need as the required input tensors need to be: [batch, in_height, in_width, in_channels]
        self.x_previous = tf.placeholder(tf.float32, shape=self.size_input, name="x_previous")
        self.x_current = tf.placeholder(tf.float32, shape=self.size_input, name="x_current")
        self.placeholder_labels = tf.placeholder(tf.float32, shape=(self.size_input[0],64,24), name="y_pos_probs")



        # Layer C1: convolutional layer with 10 feature maps
        # each feature map will be created independently
        # number of feature maps (=number of filters) in the complete conv layer C1: 10
        # => define either all at once with number of filters = 10
        # => OR define 10 times a partially-connected conv layer with number of filters (in each) layer = 1
        # size of a single filter: 5x5 (neighborhood used as input)
        # output size of the feature map must be 44 x 124px
        self.conv1_independent_parts = 10
        self.number_of_filters_conv1 = 1
        self.conv_filter_width = 5
        self.conv_filter_height = 5
        single_image_channels = self.size_input[-1]  # 5

        W_conv, b_conv, h_pool, S1 = [], [], [], []
        for i in range(self.conv1_independent_parts):
            with tf.name_scope("C1_part{}".format(i + 1)):

                W_conv1, b_conv1 = self.vars_W_b(
                    [self.conv_filter_width, self.conv_filter_height, single_image_channels,
                     self.number_of_filters_conv1])
                W_conv.append(W_conv1)
                b_conv.append(b_conv1)

                # the very first 5 feature maps are using the previous image only
                # the remaining 5 feature maps are using the current image only
                if i < 5:
                    input_source = self.x_previous
                else:
                    input_source = self.x_current

                # build actual (part of) convolutional layer
                # TODO no activation function ?? e.g.: h_conv = tf.nn.relu(h_conv)
                h_conv = self.conv2d(input_source, W_conv1) + b_conv1

            # Layer S1 is a pooling layer with 10 feature maps using max operation
            # each of this feature maps is only connected to exactly one particular
            # feature map of C1. So we can just connect each part of C1 right after
            # creating to the corresponding part of S1
            # => 2x2 pooling (as only 4 values are used)
            # TODO ensure that those params are equivalent to the given formula in the paper (they should fit)
            with tf.name_scope("S1_part{}".format(i + 1)):
                S1.append(self.max_pool_2x2(h_conv))

        S1 = np.asarray(S1)

        ##### GLOBAL BRANCH Begin #####
        c2size = 33
        C2, S2 = [], []
        for i in range(c2size):
            with tf.name_scope("C2_part{}".format(i + 1)):

                if i == 0:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2]])
                elif i == 1:
                    c2_input = self.mergeChannels([S1[1], S1[2], S1[3]])
                elif i == 2:
                    c2_input = self.mergeChannels([S1[2], S1[3], S1[4]])
                elif i == 3:
                    c2_input = self.mergeChannels([S1[0], S1[3], S1[4]])
                elif i == 4:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[4]])
                elif i == 5:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2], S1[3]])
                elif i == 6:
                    c2_input = self.mergeChannels([S1[1], S1[2], S1[3], S1[4]])
                elif i == 7:
                    c2_input = self.mergeChannels([S1[0], S1[2], S1[3], S1[4]])
                elif i == 8:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[3], S1[4]])
                elif i == 9:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2], S1[4]])
                elif i == 10:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2], S1[3], S1[4]])
                elif i == 11:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[7]])
                elif i == 12:
                    c2_input = self.mergeChannels([S1[6], S1[7], S1[8]])
                elif i == 13:
                    c2_input = self.mergeChannels([S1[7], S1[8], S1[9]])
                elif i == 14:
                    c2_input = self.mergeChannels([S1[5], S1[8], S1[9]])
                elif i == 15:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[9]])
                elif i == 16:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[7], S1[8]])
                elif i == 17:
                    c2_input = self.mergeChannels([S1[6], S1[7], S1[8], S1[9]])
                elif i == 18:
                    c2_input = self.mergeChannels([S1[5], S1[7], S1[8], S1[9]])
                elif i == 19:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[8], S1[9]])
                elif i == 20:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[7], S1[9]])
                elif i == 21:
                    c2_input = self.mergeChannels([S1[5], S1[6], S1[7], S1[8], S1[9]])
                elif i == 22:
                    c2_input = self.mergeChannels([S1[0], S1[2], S1[3], S1[5], S1[7], S1[8]])
                elif i == 23:
                    c2_input = self.mergeChannels([S1[1], S1[3], S1[4], S1[6], S1[8], S1[9]])
                elif i == 24:
                    c2_input = self.mergeChannels([S1[0], S1[2], S1[4], S1[5], S1[7], S1[9]])
                elif i == 25:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[3], S1[5], S1[6], S1[7]])
                elif i == 26:
                    c2_input = self.mergeChannels([S1[1], S1[2], S1[4], S1[6], S1[7], S1[9]])
                elif i == 27:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2], S1[5], S1[8], S1[9]])
                elif i == 28:
                    c2_input = self.mergeChannels([S1[1], S1[2], S1[3], S1[5], S1[6], S1[9]])
                elif i == 29:
                    c2_input = self.mergeChannels([S1[2], S1[3], S1[4], S1[5], S1[6], S1[7]])
                elif i == 30:
                    c2_input = self.mergeChannels([S1[0], S1[3], S1[4], S1[6], S1[7], S1[8]])
                elif i == 31:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[4], S1[7], S1[8], S1[9]])
                elif i == 32:
                    c2_input = self.mergeChannels([S1[0], S1[1], S1[2], S1[3], S1[4], S1[5], S1[6], S1[7], S1[8], S1[9]])

                self.conv2_filter_height = 7
                self.conv2_filter_width = 3
                self.number_of_filters_conv2 = 1
                c2number_channels = c2_input.get_shape()[-1].value
                W_conv2, b_conv2 = self.vars_W_b(
                    [self.conv2_filter_width, self.conv2_filter_height, c2number_channels,
                     self.number_of_filters_conv2])

                # build actual (part of) convolutional layer
                # TODO no activation function ?? e.g.: h_conv = tf.nn.relu(h_conv)
                h_conv = self.conv2d(c2_input, W_conv2) + b_conv2
                C2.append(h_conv)

                # Layer S2 is a pooling layer with 33 feature maps using max operation
                # each of this feature maps is only connected to exactly one particular
                # feature map of C2. So we can just connect each part of C2 right after
                # creating to the corresponding part of S2
                # => 2x2 pooling (as only 4 values are used)
                # TODO ensure that those params are equivalent to the given formula in the paper (they should fit)
                with tf.name_scope("S2_part{}".format(i + 1)):
                    S2.append(self.max_pool_2x2(h_conv))

        C2 = np.asarray(C2)
        S2 = np.asarray(S2)



        ##

        c3size = 80
        with tf.name_scope("C3"):

            # merge all existing inputs to simulate further 3D convolutional ops
            # TODO in the paper they propose using "10 random choices" instead of all 33. How to implement?
            c3_input = self.mergeChannels([
                S2[0], S2[1], S2[2], S2[3], S2[4], S2[5], S2[6], S2[7], S2[8], S2[9],
                S2[10], S2[11], S2[12], S2[13], S2[14], S2[15], S2[16], S2[17], S2[18], S2[19],
                S2[20], S2[21], S2[22], S2[23], S2[24], S2[25], S2[26], S2[27], S2[28], S2[29],
                S2[30], S2[31], S2[32]
            ])

            self.conv3_filter_height = 7
            self.conv3_filter_width = 3
            self.number_of_filters_conv3 = 1
            c3number_channels = c3_input.get_shape()[-1].value
            W_conv3, b_conv3 = self.vars_W_b(
                [self.conv3_filter_width, self.conv3_filter_height, c3number_channels,
                 self.number_of_filters_conv3])

            # build actual (part of) convolutional layer
            # TODO no activation function ?? e.g.: h_conv = tf.nn.relu(h_conv)
            C3 = self.conv2d(c3_input, W_conv3) + b_conv3

        # (4 times??) upsamling to 24 x 64
        C3 = tf.image.resize_images(C3, 64, 24)

        ##### GLOBAL BRANCH End #######

        ##### LOCAL BRANCH Begin ######

        c4size = 10
        with tf.name_scope("C4"):

            # merge all existing inputs to simulate further 3D convolutional ops
            c4_input = self.mergeChannels([
                S1[0], S1[1], S1[2], S1[3], S1[4], S1[5], S1[6], S1[7], S1[8], S1[9]
            ])

            self.conv4_filter_height = 7
            self.conv4_filter_width = 7
            self.number_of_filters_conv4 = 1
            c4number_channels = c4_input.get_shape()[-1].value
            W_conv4, b_conv4 = self.vars_W_b(
                [self.conv4_filter_width, self.conv4_filter_height, c4number_channels,
                 self.number_of_filters_conv4])

            # build actual (part of) convolutional layer
            # TODO no activation function ?? e.g.: h_conv = tf.nn.relu(h_conv)
            C4 = self.conv2d(c4_input, W_conv4) + b_conv4


        # TODO the paper mentiones a translation transfrom at this point. Check out what this means and if we need
        W_conv4, b_conv4 = self.vars_W_b(
            [self.conv4_filter_width, self.conv4_filter_height, c4number_channels,
             self.number_of_filters_conv4])
        # to do anything

        ##### LOCAL BRANCH End ########

        # Output

        # remove last dimension of shape 1 from tensors
        C3 = tf.squeeze(C3)
        C4 = tf.squeeze(C4)

        finalW1, finalB1 = self.vars_W_b([64, 24]) #finalB1 will not be used
        finalW2, finalB2 = self.vars_W_b([64, 24])
        self.scores =  tf.sigmoid(tf.mul(C3, finalW1) + tf.mul(C4, finalW2) + finalB2)

        # loss function
        with tf.name_scope("loss"):

            # tensorflow can only evaluate 2D results. So we just handle each pixel of the probability map as a single
            # class => flattening needed
            scoresFlattened = tf.reshape(self.scores, [self.batch_size, -1])
            targetProbsFlattened = tf.reshape(self.placeholder_labels, [self.batch_size, -1])

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scoresFlattened,
                                                                    targetProbsFlattened,
                                                                           name="xentropy")
            loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

        # add L2 regularization #TODO
        #if self.regularization_strength != 0:
        #    regularizers = tf.nn.l2_loss(W_full) + tf.nn.l2_loss(b_full) + tf.nn.l2_loss(W_softmax) + tf.nn.l2_loss(b_softmax)
        #    reg = tf.constant(self.regularization_strength, dtype=tf.float32)
        #    loss += reg * regularizers

        # Create a variable to track the global step (should be equal to the index var "step" in the following for loop)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # automatically decay learning rate
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                        self.iterations / 20, self.learning_rate_decay, staircase=True)

        # create optimizer
        if self.optimizer == 2 and self.momentum != 0:
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        elif self.optimizer == 1:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)


        train_op = optimizer.minimize(loss, global_step=global_step)

        # keep track of loss values
        tf.scalar_summary(loss.op.name, loss)

        # and maybe of the learning rate
        tf.scalar_summary(self.learning_rate.op.name, self.learning_rate)

        summary_op = tf.merge_all_summaries()

        # create saver to be able saving current parameters at certain checkpoints
        saver = tf.train.Saver()

        # start a session
        self.session = tf.InteractiveSession()

        # initialize all vars
        init = tf.initialize_all_variables()
        self.session.run(init)

        # this method will be used to calculate the current accuracy (create only once)
        # TODO reimplement for non-integer values
        #self.check_results = tf.nn.in_top_k(self.scores, self.placeholder_labels, 1)

        # allow saving results to file
        summary_writer = tf.train.SummaryWriter(os.path.join(self.log_dir, "tf-summary"), self.session.graph)

        interrupt_every_x_steps = min(self.iterations / 2.5, 1000)
        interrupt_every_x_steps_late = self.iterations / 2;
        for step in range(self.iterations):

            # get a batch of training samples
            offset = (step * self.batch_size) % (self.Ytrain.shape[0] - self.batch_size)
            batch_data_previous = self.XtrainPrevious[offset:(offset + self.batch_size), :]
            batch_data_current = self.XtrainCurrent[offset:(offset + self.batch_size), :]
            batch_labels = self.Ytrain[offset:(offset + self.batch_size)]

            # finally start training with current batch
            feed_dict ={self.x_previous:batch_data_previous,
                        self.x_current: batch_data_current,
                        self.placeholder_labels:batch_labels} #TODO add dropout again ,self.dropout_prob: self.dropout_rate
            _, loss_value = self.session.run([train_op, loss],
                                     feed_dict)

            # write the summaries and print an overview quite often
            if True or  step % 100 == 0 or (step + 1) == self.iterations: #TODO remove "True or"
                # Print status
                log.log('Iteration {0}/{1}: loss = {2:.2f}, learning rate = {3:.4f}'.format(step + 1, self.iterations, loss_value, self.session.run(self.learning_rate)))
                # Update the events file.
                summary_str = self.session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # print current accuracies less often (sometimes in between and at the end)
            # + save checkpoint
            if (step + 1) % interrupt_every_x_steps == 0 or (step + 1) == self.iterations:

                log.log("Saving checkpoint..")
                saver.save(self.session, os.path.join(self.log_dir, "tf-checkpoint"), global_step=step)

                # don't print in last run as this will be done anyway
                if (step + 1) != self.iterations:
                    log.log("Updated accuracies after {}/{} iterations:".format((step + 1), self.iterations))

                    # 1/3: validation data TODO reuse
                    #acc_val = self.accuracy(self.Xval, self.Yval)
                    #log.log(" - validation: {0:.3f}%".format(acc_val * 100))

            if (step + 1) % interrupt_every_x_steps_late == 0 and (step + 1) != self.iterations: #don't print in last run as this will be done anyway

                # 2/3: training data TODO reuse
                True
                #acc_train = self.accuracy(self.Xtrain, self.Ytrain)
                #log.log(" - training: {0:.3f}%".format(acc_train * 100))

                # 3/3: test data
                # acc_test = self.accuracy(self.Xtest, self.Ytest)
                # log.log(" - test: {0:.3f}%".format(acc_test * 100))

            # check timeout
            if self.timeout_minutes > 0:
                self.runtime_training_end = time.time() - self.runtime_training_start

                if self.runtime_training_end > self.timeout_seconds:
                    log.log("TIMEOUT: stopping earlier. saving current work.")
                    break

        # save final runtime one last time
        # (intermediate updates might have already calculated this value, but only if timeout_minutes > 0)
        self.runtime_training_end = time.time() - self.runtime_training_start



    def accuracy(self, x, y):
        num_iter = int(math.ceil(x.shape[0] / self.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * self.batch_size
        step = 0
        while step < num_iter:
            feed_dict = {self.placeholder_images: x[step * self.batch_size: (step + 1) * self.batch_size],
                         self.placeholder_labels: y[step * self.batch_size: (step + 1) * self.batch_size],
                         self.dropout_prob: 1.0} #deactivate dropout for testing
            predictions = self.session.run(self.check_results, feed_dict)
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        return precision


    # this method needs to be called exactly ONCE before any calls of preprocessData()
    # X must be the training data, no validation data allowed!
    # the calculated values will be reused to preprocess any following data
    # input data X will be changed as if preprocessData(X) has been called, so no further call for this object required
    def preprocessInit(self, min_max_scaling, standardization, Xorig):

        X = Xorig.copy()

        # Min-Max scaling
        if min_max_scaling:
            self.preMin = np.min(X)
            xmax = np.max(X)
            self.preMinMaxScale = xmax - self.preMin

            # we need to apply minmaxscaling once to the temporary data to ensure that the effect for standardization is considered
            if self.preMinMaxScale != 0:
                X -= self.preMin
                X /= self.preMinMaxScale

        else:  # ensure that preprocessData() will not apply min_max_scaling
            self.preMin = 0
            self.preMinMaxScale = 1.0

        # standardization
        if standardization:
            # generate mean image
            self.preMeanImage = np.mean(X, axis=0)

            # normalize by dividing by the standard deviation => unit std
            self.preStd = np.std(X, axis=0)
            self.preStd[self.preStd == 0] = 0.001  # prevent divide by 0

        else:  # ensure that preprocessData() will not apply standardization
            self.preMeanImage = 0
            self.preStd = 1.0


    # apply preprocessing on X
    # preprocessInit needs to be called before
    # will only use params of previously processed training data. so X can be whatever you want, validation data, too.
    def preprocessData(self, X):
        # Min-Max scaling
        if self.preMinMaxScale != 0:
            X -= self.preMin
            X /= self.preMinMaxScale

        # standardization:
        # subtract mean image from all data => zero mean
        # (this will change the original dataset, too)
        X -= self.preMeanImage

        # normalize by dividing by the standard deviation => unit std
        X /= self.preStd


    def closeSession(self):
        # we're done here
        self.session.close()
        tf.reset_default_graph()


    def getTrainingRuntimeSeconds(self):
        return self.runtime_training_end

    def mergeChannels(self,inputs):
        return tf.concat(3, inputs)


