import tensorflow as tf
import log
import numpy as np
import math
import time
import os
import cv2 as cv
import random
import string

class ConvolutionalNetwork:
    def __init__(self, calLoader,
                 batch_size_training=200,
                 learning_rate=0.01,
                 iterations=3000,
                 learning_rate_decay=0.96,
                 momentum=0,
                 timeout_minutes=0,
                 log_dir="logs",
                 dropout_rate=0.5,
                 optimizer=0,
                 head_rel_pos_prev_row=0.25,
                 head_rel_pos_prev_col=0.5,
                 accuracy_weight_direction=0.8,
                 accuracy_weight_distance=0.2,
                 learning_rate_min=0.01,
                 max_batch_size=None):


        # save given params
        self.calLoader = calLoader
        self.batch_size_training = batch_size_training #static value during training != self.batch_size (the latter is a dynamic tensor)
        self.starter_learning_rate = learning_rate
        self.iterations = iterations
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.momentum = momentum
        self.timeout_minutes = timeout_minutes # maximum number of minutes used for training. 0=unlimited
        self.timeout_seconds = self.timeout_minutes * 60
        self.log_dir = log_dir
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.head_rel_pos_prev_row = head_rel_pos_prev_row
        self.head_rel_pos_prev_col = head_rel_pos_prev_col
        self.output_width = 24
        self.output_height = 64
        self.accuracy_weight_distance = accuracy_weight_distance
        self.accuracy_weight_direction = accuracy_weight_direction
        self.max_batch_size = max_batch_size

        # create session name that is (in the best case) unique. this name will be used for file names etc.
        # => timestamp, underscore, 3 random letters
        self.session_name = "{}_".format( time.time()) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters)

        # load training and validation data
        self.XtrainPrevious, self.XtrainCurrent, self.Ytrain = calLoader.getTrainingData()
        self.XvalPrevious, self.XvalCurrent, self.Yval = calLoader.getValidationData()
        log.log('.. Trainingset includes {} images.'.format(self.XtrainPrevious.shape[0]))
        log.log('.. Validationset includes {} images.'.format(self.XvalPrevious.shape[0]))

        # get input dimension:
        # original image shape, but use dynamic size for first dimension in order to allow different batch sizes
        self.size_input = []
        orig_img_shape = tf.TensorShape(self.XtrainPrevious.shape).as_list()
        for i in range(len(orig_img_shape)):
            if i > 0:
                self.size_input.append(orig_img_shape[i])
            else:
                self.size_input.append(None)

        self.size_input = [None, self.XtrainPrevious.shape[1], self.XtrainPrevious.shape[2], self.XtrainPrevious.shape[3]]


            # some output
        log.log('Creating network..')
        log.log('.. Input dimension: {}.'.format(self.size_input))
        log.log('.. drop out between S2 and C3 active: {}'.format(self.dropout_rate > 0 and self.dropout_rate < 1))
        if self.dropout_rate > 0 and self.dropout_rate < 1:
            log.log('.. drop out rate: {}'.format(self.dropout_rate))

        # begin with actual network creation
        self.setUpArchitecture()




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

        # allow saving results to file
        summary_writer = tf.train.SummaryWriter(os.path.join(self.log_dir, self.session_name + "-tf-summary"), self.session.graph)

        interrupt_every_x_steps = min(self.iterations / 2.5, 1000, 10) #TODO remove ", 10" on computer with higher performance
        interrupt_every_x_steps_late = max(self.iterations / 4, 1)
        for step in range(self.iterations):

            # get a batch of training samples
            offset = (step * self.batch_size_training) % (self.Ytrain.shape[0] - self.batch_size_training)
            batch_data_previous = self.XtrainPrevious[offset:(offset + self.batch_size_training), :]
            batch_data_current = self.XtrainCurrent[offset:(offset + self.batch_size_training), :]
            batch_labels = self.Ytrain[offset:(offset + self.batch_size_training)]

            # finally start training with current batch
            feed_dict ={self.x_previous:batch_data_previous,
                        self.x_current: batch_data_current,
                        self.placeholder_labels:batch_labels,
                        self.dropout_prob: self.dropout_rate}
            _, loss_value = self.session.run([self.train_op, self.loss],
                                     feed_dict)

            # write the summaries and print an overview quite often
            if True or step % 100 == 0 or (step + 1) == self.iterations: #TODO remove "True or"
                # Print status
                log.log('Iteration {0}/{1}: loss = {2:.2f}, learning rate = {3:.4f}'.format(step + 1, self.iterations, loss_value, self.session.run(self.learning_rate)))
                # Update the events file.
                summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # print current accuracies less often (sometimes in between and at the end)
            # + save checkpoint
            if (step + 1) % interrupt_every_x_steps == 0 or (step + 1) == self.iterations:

                log.log("Saving checkpoint..")
                self.saver.save(self.session, os.path.join(self.log_dir, self.session_name + "-tf-checkpoint-loss_{}".format(loss_value)), global_step=step)

                # don't print in last run as this will be done anyway
                if (step + 1) != self.iterations:
                    log.log("Updated accuracies after {}/{} iterations:".format((step + 1), self.iterations))

                    # 1/3: validation data
                    acc_val = self.accuracy(self.XvalPrevious, self.XvalCurrent, self.Yval)
                    log.log(" - validation: {0:.3f}%".format(acc_val * 100))

            if (step + 1) % interrupt_every_x_steps_late == 0 and (step + 1) != self.iterations: #don't print in last run as this will be done anyway

                # 2/3: training data
                acc_train = self.accuracy(self.XtrainPrevious, self.XtrainCurrent, self.Ytrain)
                log.log(" - training: {0:.3f}%".format(acc_train * 100))

                # 3/3: test data
                # acc_test = self.accuracy(self.XtestPrevious, self.XtestCurrent, self.Ytest)
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

        log.log('.. training finished.')


    def setUpArchitecture(self):

        # input data: two batches of images. equal indices of the two batches must form valid pairs.
        # e.g. x_previous[i] and x_current[i] a one valid frame pair. Each image has the size = 48 x 128.
        # Each pixel got 5 channels: R, G, B, Dx, Dy
        # dims(self.x_previous) = dims(self.x_current) = <batch_size>, <image_height=128>, <image_width=48>, <channels=5>
        # reminder: "image is flipped": the image format is the one of opencv
        #       => <number_of_rows> x <number_of_cols> = 128 x 48
        #       => this is exactly what we need as the required input tensors need to be: [batch, in_height, in_width, in_channels]
        self.x_previous = tf.placeholder(tf.float32, shape=self.size_input, name="x_previous")
        self.x_current = tf.placeholder(tf.float32, shape=self.size_input, name="x_current")
        self.placeholder_labels = tf.placeholder(tf.float32,
                                                 shape=(self.size_input[0], self.output_height, self.output_width),
                                                 name="y_pos_probs")

        # save batch size as dynamic tensor
        self.batch_size = tf.shape(self.x_previous)[0]

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
        S2 = []
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
                    c2_input = self.mergeChannels(
                        [S1[0], S1[1], S1[2], S1[3], S1[4], S1[5], S1[6], S1[7], S1[8], S1[9]])

                self.conv2_filter_height = 7
                self.conv2_filter_width = 3
                self.number_of_filters_conv2 = 1
                c2number_channels = c2_input.get_shape()[-1].value
                W_conv2, b_conv2 = self.vars_W_b(
                    [self.conv2_filter_width, self.conv2_filter_height, c2number_channels,
                     self.number_of_filters_conv2])

                # build actual (part of) convolutional layer
                # TODO no activation function ?? e.g.: h_conv = tf.nn.relu(h_conv)
                h_conv = self.conv2d(c2_input, W_conv2) + b_conv2 #C2

                # Layer S2 is a pooling layer with 33 feature maps using max operation
                # each of this feature maps is only connected to exactly one particular
                # feature map of C2. So we can just connect each part of C2 right after
                # creating to the corresponding part of S2
                # => 2x2 pooling (as only 4 values are used)
                # TODO ensure that those params are equivalent to the given formula in the paper (they should fit)
                with tf.name_scope("S2_part{}".format(i + 1)):
                    S2.append(self.max_pool_2x2(h_conv))

        S2 = np.asarray(S2)

        ##

        c3size = 80
        with tf.name_scope("C3"):

            # merge all existing inputs to simulate further 3D convolutional ops
            c3_input = self.mergeChannels([
                S2[0], S2[1], S2[2], S2[3], S2[4], S2[5], S2[6], S2[7], S2[8], S2[9],
                S2[10], S2[11], S2[12], S2[13], S2[14], S2[15], S2[16], S2[17], S2[18], S2[19],
                S2[20], S2[21], S2[22], S2[23], S2[24], S2[25], S2[26], S2[27], S2[28], S2[29],
                S2[30], S2[31], S2[32]
            ])

            # in the paper they propose using "10 random choices" instead of all 33. How to implement?
            # dropout comes closest to this definition without keeping data untouched
            if self.dropout_rate != 1.0:
                with tf.name_scope("dropout"):
                    self.dropout_prob = tf.placeholder(tf.float32)
                    c3_input = tf.nn.dropout(c3_input, self.dropout_prob)

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
        C3 = tf.image.resize_images(C3, self.output_height, self.output_width)

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

        finalW1, finalB1 = self.vars_W_b([self.output_height, self.output_width])  # finalB1 will not be used
        finalW2, finalB2 = self.vars_W_b([self.output_height, self.output_width])
        self.scores = tf.sigmoid(tf.mul(C3, finalW1) + tf.mul(C4, finalW2) + finalB2)

        # loss function
        with tf.name_scope("loss"):

            # tensorflow can only evaluate 2D results. So we just handle each pixel of the probability map as a single
            # class => flattening needed
            # tf.pack() allows using a dynamic batch_size
            self.scoresFlattened = tf.reshape(self.scores, tf.pack([self.batch_size, -1]))
            self.targetProbsFlattened = tf.reshape(self.placeholder_labels, tf.pack([self.batch_size, -1]))

            # apply softmax on target map
            self.targetProbsFlattened = tf.nn.softmax(self.targetProbsFlattened)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.scoresFlattened,
                                                                    self.targetProbsFlattened,
                                                                    name="xentropy")
            self.loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

        # Create a variable to track the global step (should be equal to the index var "step" in the following for loop)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # automatically decay learning rate
        self.learning_rate_calc = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                        self.iterations / 20, self.learning_rate_decay, staircase=True)
        self.learning_rate = tf.maximum(self.learning_rate_calc, self.learning_rate_min)

        # create optimizer
        if self.optimizer == 2 and self.momentum != 0:
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        elif self.optimizer == 1:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        # keep track of loss values
        tf.scalar_summary(self.loss.op.name, self.loss)

        # and maybe of the learning rate
        tf.scalar_summary(self.learning_rate.op.name, self.learning_rate)

        self.summary_op = tf.merge_all_summaries()

        # create saver to be able saving current parameters at certain checkpoints
        self.saver = tf.train.Saver()

        # start a session
        self.session = tf.InteractiveSession()

        # initialize all vars
        init = tf.initialize_all_variables()
        self.session.run(init)


        ## accuracy Begin ###

        # the (supposed) position of a pedestrian's head will always be the same for all PREVIOUS frames
        # => once prepare same value for one batch size, so we can use this during training
        # => update: instead of copying the same value, we just use an array with shape [1,2] which will be broadcasted
        # TODO x and y correct order? especially compare data in caltech and the loader
        row = int(round(self.output_height * self.head_rel_pos_prev_row))
        col = int(round(self.output_width * self.head_rel_pos_prev_col))
        self.position_previous_2D_batch = np.array([[row, col]])

        self.position_predicted_2D = self.get_target_position(self.scoresFlattened)
        position_real_2D = self.get_target_position(self.targetProbsFlattened)
        movement_real = self.position_previous_2D_batch - position_real_2D
        movement_predicted = self.position_previous_2D_batch - self.position_predicted_2D

        # calculate difference in vector length (=distance of proposed movement)
        # (+ 0.0001 to prevent division by zero, which could occur in angle_between_movements() as well as calc of diff_distance)
        distance_real = self.tf_norm_batch(movement_real) + 0.0001
        distance_predicted = self.tf_norm_batch(movement_predicted) + 0.0001

        # keep value always between 0 and 1: 1=best fit
        diff_distance = tf.minimum(distance_predicted, distance_real) / tf.maximum(distance_predicted, distance_real)

        # calculate difference in rad
        angle_rad = self.angle_between_movements(movement_real, movement_predicted, distance_real, distance_predicted)
        diff_direction = 1 - (angle_rad / math.pi)  # 180 degrees = PI = 0%, 0 = 100%

        # combine angle and distance differences to total accuracy
        self.check_results = self.accuracy_weight_direction * diff_direction + self.accuracy_weight_distance * diff_distance

        # calculate a single mean value instead of <batch_size> values
        self.calc_accuracy = tf.reduce_mean(tf.cast(self.check_results, tf.float32))

        ## accuracy End ###


    def get_target_position(self, probs_1d):

        # TODO does this give the correct position?
        # TODO (not needed for accuracy, but for "real" tracking:) scale is still 0.5, so we need to multiply by 2
        position_predicted_1D = tf.reshape(tf.argmax(probs_1d, 1),tf.pack([self.batch_size, 1]))
        row = position_predicted_1D / self.output_width  # needs to be floored
        column = position_predicted_1D % self.output_width
        position_predicted_2D = tf.concat(1,[row,column])

        return position_predicted_2D


    # if self.max_batch_size = None => do all at once
    # otherwise split up dataset in batches of size max_batch_size
    def accuracy(self, x_prev, x_curr, y):

        if self.max_batch_size is None: #do all at once
            feed_dict = {self.x_previous: x_prev,
                         self.x_current: x_curr,
                         self.placeholder_labels: y,
                         self.dropout_prob: 1.0} #deactivate dropout for testing
            precision = self.session.run(self.calc_accuracy, feed_dict)

        #split up dataset in batches of size max_batch_size
        else:

            total_sample_count = x_prev.shape[0]
            num_iter = int(math.ceil(total_sample_count / float(self.max_batch_size)))
            true_count = 0  # sum up all single accuracies
            step = 0
            while step < num_iter:
                i_from = step * self.max_batch_size
                i_to = (step + 1) * self.max_batch_size  # we use this for the input to ensure the fixed batch size
                i_to_real = min(i_to,
                                total_sample_count)  # but our actual dataset might be smaller, so we only use data up to here
                feed_dict = {self.x_previous: x_prev[i_from: i_to],
                             self.x_current: x_curr[i_from: i_to],
                             self.placeholder_labels: y[i_from: i_to],
                             self.dropout_prob: 1.0}  # deactivate dropout for testing
                predictions = self.session.run(self.check_results, feed_dict)
                true_count += np.sum(predictions[0: i_to_real - i_from])
                step += 1

            # Compute precision @ 1.
            precision = true_count
            if total_sample_count > 0:
                precision /= float(total_sample_count)

        return precision


    def closeSession(self):
        # we're done here
        self.session.close()
        tf.reset_default_graph()


    def getTrainingRuntimeSeconds(self):
        return self.runtime_training_end

    def mergeChannels(self,inputs):
        return tf.concat(3, inputs)

    # source: http://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2
    # returns angle between the two lines in range of 0 to 180
    def angle_between_movements(self, vA, vB, magA, magB):

        # Get nicer vector form / THIS STEP IS ALREADY DONE BEFORE
        #vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        #vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]

        # Get dot prod
        dot_prod = vA * vB
        dot_prod = tf.reduce_sum(dot_prod, 1, True)
        dot_prod = tf.to_float(dot_prod)

        # Get magnitudes (already precalculated, so we take those)
        #magA = self.dot(vA, vA) ** 0.5
        #magB = self.dot(vB, vB) ** 0.5

        # Get cosine value
        cos = tf.truediv(tf.truediv(dot_prod, magA), magB) #dot_prod / magA / magB

        # Get angle in radians
        angle = self.tf_arccos(cos)

        return angle

    def get_session_name(self):
        return self.session_name


    # calculate the element-wise arcuscosinus.
    # return value is between 0 and PI
    # Tensorflow doesn't provide this feature yet, but will, see:
    # https: // github.com / tensorflow / tensorflow / issues / 1108
    # TODO use correct implementation instead of approximation
    def tf_arccos(self, tensor):

        # theoretically, we should be able to use numpy for this..
        # but it's not working..
        #angle = np.arccos(cos) #this complains that a Tensor doesn't got an attribute called arccos
        #angle = np.arccos(cos.eval()) #this should convert Tensor to np before, but doesn't work either

        # use simple approximation with max error of 10 degrees
        # source: http://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
        angle = (-0.69813170079773212 * tensor * tensor - 0.87266462599716477) * tensor + 1.5707963267948966;

        return angle

    '''
    Calculates "a norm" for a batch of tensors.
    So tensor's shape must be [self.batch_size, x, y]
    Returning value will have shape [self.batch_size, 1]
    and each entry got the value sqrt(x*x + y*y)
    '''
    def tf_norm_batch(self, tensor):

        result = tf.square(tensor)
        result = tf.reduce_sum(result, 1, True)

        # TODO we might skip the sqrt, as the "real scale" doesn't matter to us for camparing distance ratios,
        # but we need to ensure that angle_between_movements is working correctly, too
        result = tf.to_float(result) #everything before might have been integer only
        result = tf.sqrt(result)

        return result

    # this method might get called after the training has been finished
    def final_evaluation(self):

        log.log("starting final evaluation")

        # if not already done, load test data
        log.log("No testset available yet, start loading it..")
        XtestPrev, XtestCurr, Ytest = self.calLoader.getTestData()
        log.log('Loaded testset, which includes {} images.'.format(XtestPrev.shape[0]))

        val_acc = self.accuracy(self.XvalPrevious, self.XvalCurrent, self.Yval)
        log.log('FINAL Accuracy validation {0:.3f}%'.format(val_acc * 100))

        test_acc = self.accuracy(XtestPrev, XtestCurr, Ytest)
        log.log('FINAL Accuracy test {0:.3f}%'.format(test_acc * 100))

        train_acc = self.accuracy(self.XtrainPrevious, self.XtrainCurrent, self.Ytrain)
        log.log('FINAL Accuracy training {0:.3f}%'.format(train_acc * 100))

        log.log("final evaluation is done.")

        self.closeSession()
        log.log("session closed")

        return val_acc, test_acc, train_acc


    # track one or multiple pedestrians in a (complete) video.
    # => starting with a given initial position for each pedestrian, use that position to predict the position in the
    # next frame. Use this position as a new init for the next frame. And so on..
    # frames: array containing consecutive frames of one (or multiple) videos
    # ped_pos_init: array containing all initial pedestrian positions. First dimension got exactly one element
    # for each frame in frames. This element is another array containing the location of a single pedestrian in each
    # of it's elements. TSo:
    # ped_pos_init[frame_index][ped_index] = [x,y,width,height]
    #   [x,y] are the supposed coordinates of the pedestrian's head. The head's relative position to a bounding box of
    #   width x height pixels is determined by self.head_rel_pos_prev_row and self.head_rel_pos_prev_col.
    def live_tracking_video(self, frames, ped_pos_init, visualize_file_name=None):

        log.log("tracking {} frames".format(len(frames)))

        if not visualize_file_name is None:

            output_folder = 'plots'

            # create output folder if it doesn't exist yet
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            wri = cv.VideoWriter(
                os.path.join(output_folder, visualize_file_name + '.avi'),
                cv.cv.CV_FOURCC(*'XVID'), 30, (640, 480))

        frame_index = 0
        for frame in frames:

            # if this frame doesn't provide any information yet, we need to add an empty list before extending it
            # this assumes, that there are information about the previous frame
            if len(ped_pos_init) <= frame_index:
                ped_pos_init.append([])

            if frame_index > 0:

                log.log("tracking in frame {}".format(frame_index + 1))

                # start tracking (use position from PREVIOUS frame!)
                ped_pos_predicted = self.live_tracking_frame(frame_prev, frame, ped_pos_init[frame_index -1])

                # merge ped_pos_predicted and ped_pos_init[frame_index]
                if len(ped_pos_predicted) > 0:

                    if len(ped_pos_init[frame_index]) > 0:
                        ped_pos_init[frame_index] = np.append(ped_pos_init[frame_index], ped_pos_predicted, axis=0)
                    else:
                        ped_pos_init[frame_index] = ped_pos_predicted



                # visualize results
                if not visualize_file_name is None:
                    for ped_pos in ped_pos_predicted:
                        x, y, w, h = [int(v) for v in ped_pos]
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

                    # TODO write the current frame rate (of processing) into the video

                    wri.write(frame)

            frame_prev = frame
            frame_index += 1

        if not visualize_file_name is None:
            wri.release()

        return ped_pos_init


    # track one or multiple pedestrians in a single frame.
    # => predict their position in the (unknown) next frame
    # return value: array containing the predicted positions: ret[ped_index] = [x,y,width,height]
    def live_tracking_frame(self, frame_prev, frame_curr, ped_pos_prev):


        # ensure that positions are given as ints only,
        # so we can use them for direct accessing
        ped_pos_prev = np.asarray(ped_pos_prev, np.int16)

        # we will add all pedestrians of a single frame to one (or multiple) batch(es),
        # so we can evaluate all of them "at once"
        num_samples = len(ped_pos_prev)

        if num_samples > 0:

            log.log("found {} pedestrian(s) in current frame".format(num_samples))

            patch_shape = [num_samples, self.size_input[1], self.size_input[2], 5]
            patches_prev = np.zeros(patch_shape, np.float16)
            patches_curr = np.zeros(patch_shape, np.float16)

            # collect information about single pedestrians
            ped_index = 0
            for ped_pos in ped_pos_prev:

                # transform head position to top left corner
                corner_x = ped_pos[0] - int(round(self.head_rel_pos_prev_col * ped_pos[2]))
                corner_y = ped_pos[1] - int(round(self.head_rel_pos_prev_row * ped_pos[3]))

                # other required corners
                other_corner_x = corner_x + ped_pos[2]
                other_corner_y = corner_y + ped_pos[3]

                # only go on, if all corners are inside of the image (assuming both frames have the same size)
                # ideally this should happen, if the pedestrian "moves out of the image"
                # on the other hand, is that possible at all with the current cnn output??
                if corner_x >= 0 and other_corner_x >= 0 and corner_y >= 0 and other_corner_y >= 0 and corner_y < frame_prev.shape[0] and other_corner_y < frame_prev.shape[0] and corner_x < frame_prev.shape[1] and other_corner_x < frame_prev.shape[1]:

                    # get pedestrians image patch from frame_prev
                    patch_prev_temp = frame_prev[corner_y:other_corner_y, corner_x:other_corner_x]

                    # get pedestrians image patch from frame_curr
                    patch_curr_temp = frame_curr[corner_y:other_corner_y, corner_x:other_corner_x]

                    try:

                        # resize patches
                        patch_prev_temp = cv.resize(patch_prev_temp, (self.size_input[2], self.size_input[1]))
                        patch_curr_temp = cv.resize(patch_curr_temp, (self.size_input[2], self.size_input[1]))

                        # add gradient channels
                        patches_prev[ped_index] = self.calLoader.wrapImage(patch_prev_temp)
                        patches_curr[ped_index] = self.calLoader.wrapImage(patch_curr_temp)

                        ped_index += 1
                    except Exception as e:
                        var_temp = 0
                        # resizing might fail for too small image patches. just ignore/skip those

            # preprocess
            pre = self.calLoader.get_preprocessor()
            pre.preprocessData([patches_prev, patches_curr])


            # run complete batch
            ped_pos_predicted = self.predict(patches_prev, patches_curr)

            # stretch x and y
            ped_pos_predicted *= 2

            # add two new dimension for width and height (2 => 4)
            dim_width = np.zeros([ped_pos_predicted.shape[0], 1])
            dim_height = np.zeros([ped_pos_predicted.shape[0], 1])
            ped_pos_predicted = np.append(ped_pos_predicted, dim_width, axis=1)
            ped_pos_predicted = np.append(ped_pos_predicted, dim_height, axis=1)

            # add static width and height from given input again
            # and convert position to absolute coordinates
            ped_pos_predicted += ped_pos_prev

            # check for lost persons
            lost_persons = num_samples - (ped_index + 1)
            if lost_persons > 0:
                # get rid of empty data for lost/skipped persons
                ped_pos_predicted = ped_pos_predicted[0: num_samples - lost_persons -1]
                log.log("{} person(s) have left the video".format(lost_persons))


        else: #nothing to do here
            ped_pos_predicted = []


        return ped_pos_predicted

    # predict positions for next frame from based on patches_prev and patches_curr
    # patches_prev.shape == patches_curr.shape == [<num_samples>,self.size_input[1], self.size_input[2], self.size_input[3]]
    # return.shape == [<num_samples>, 2]
    # requires data to be preprocessed and extended by gradient values
    # returned coordinates are relative to output prob map of size self.output_width x self.output_height
    #   => you need to transform them to absolute coordinates again
    def predict(self, patches_prev, patches_curr):

        feed_dict = {self.x_previous: patches_prev,
                     self.x_current: patches_curr,
                     # self.placeholder_labels: y,
                     self.dropout_prob: 1.0}  # deactivate dropout for live system

        predictions = self.session.run(self.position_predicted_2D, feed_dict)

        return predictions