import tensorflow as tf
import log
import numpy as np
import math
import time
import os

class ConvolutionalNetwork:
    def __init__(self, XtrainPrevious, XtrainCurrent, Ytrain, XvalPrevious, XvalCurrent, Yval,
                 batch_size=200,
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
                 accuracy_weight_distance=0.2):

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
        self.iterations = iterations
        self.learning_rate_decay = learning_rate_decay
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

        # derive further params

        # the (supposed) position of a pedestrian's head will always be the same for all PREVIOUS frames
        # => once copy same value for one batch size, so we can use this during training
        self.position_previous_2D_batch = np.zeros([self.batch_size, 2])
        for x in self.position_previous_2D_batch:
            x[0] = self.output_height * self.head_rel_pos_prev_row
            x[1] = self.output_width * self.head_rel_pos_prev_col

        # use original image shape, but resize the number of images to a single batch
        self.size_input = []
        orig_img_shape = tf.TensorShape(self.XtrainPrevious.shape).as_list()
        for i in range(len(orig_img_shape)):
            if i > 0:
                self.size_input.append(orig_img_shape[i])
            else:
                self.size_input.append(self.batch_size)

        log.log('.. Input dimension: {}.'.format(self.size_input))
        # log.log( '.. Standard deviation W-init: {}.'.format(cf_standard_deviation_w_init) )




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
        self.placeholder_labels = tf.placeholder(tf.float32, shape=(self.size_input[0],self.output_height,self.output_width), name="y_pos_probs")



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

        finalW1, finalB1 = self.vars_W_b([self.output_height, self.output_width]) #finalB1 will not be used
        finalW2, finalB2 = self.vars_W_b([self.output_height, self.output_width])
        self.scores =  tf.sigmoid(tf.mul(C3, finalW1) + tf.mul(C4, finalW2) + finalB2)

        # loss function
        with tf.name_scope("loss"):

            # tensorflow can only evaluate 2D results. So we just handle each pixel of the probability map as a single
            # class => flattening needed
            self.scoresFlattened = tf.reshape(self.scores, [self.batch_size, -1])
            self.targetProbsFlattened = tf.reshape(self.placeholder_labels, [self.batch_size, -1])

            # apply softmax on target map
            self.targetProbsFlattened = tf.nn.softmax(self.targetProbsFlattened)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.scoresFlattened,
                                                                    self.targetProbsFlattened,
                                                                           name="xentropy")
            loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")


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

        ## accuracy Begin ###
        position_predicted_2D = self.get_target_position(self.scoresFlattened)
        position_real_2D = self.get_target_position(self.targetProbsFlattened)
        movement_real = self.position_previous_2D_batch - position_real_2D
        movement_predicted = self.position_previous_2D_batch - position_predicted_2D

        # calculate difference in vector length (=distance of proposed movement)
        # (+ 0.0001 to prevent division by zero, which could occur in angle_between_movements() as well as calc of diff_distance)
        distance_real = self.tf_norm_batch(movement_real) + 0.0001
        distance_predicted = self.tf_norm_batch(movement_predicted) + 0.0001

        # keep value always between 0 and 1: 1=best fit
        diff_distance = tf.minimum(distance_predicted, distance_real) / tf.maximum(distance_predicted, distance_real)

        # calculate difference in rad
        angle_rad = self.angle_between_movements(movement_real, movement_predicted, distance_real, distance_predicted)
        diff_direction = 1 - (angle_rad / math.pi) #180 degrees = PI = 0%, 0 = 100%

        # combine angle and distance differences to total accuracy
        self.check_results = self.accuracy_weight_direction * diff_direction + self.accuracy_weight_distance * diff_distance

        ## accuracy End ###

        # allow saving results to file
        summary_writer = tf.train.SummaryWriter(os.path.join(self.log_dir, "tf-summary"), self.session.graph)

        interrupt_every_x_steps = min(self.iterations / 2.5, 1000, 10) #TODO remove ", 10" on computer with higher performance
        interrupt_every_x_steps_late = max(self.iterations / 4, 1)
        for step in range(self.iterations):

            # get a batch of training samples
            offset = (step * self.batch_size) % (self.Ytrain.shape[0] - self.batch_size)
            batch_data_previous = self.XtrainPrevious[offset:(offset + self.batch_size), :]
            batch_data_current = self.XtrainCurrent[offset:(offset + self.batch_size), :]
            batch_labels = self.Ytrain[offset:(offset + self.batch_size)]

            # finally start training with current batch
            feed_dict ={self.x_previous:batch_data_previous,
                        self.x_current: batch_data_current,
                        self.placeholder_labels:batch_labels,
                        self.dropout_prob: self.dropout_rate}
            _, loss_value = self.session.run([train_op, loss],
                                     feed_dict)

            # write the summaries and print an overview quite often
            if True or step % 100 == 0 or (step + 1) == self.iterations: #TODO remove "True or"
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



    def get_target_position(self, probs_1d):

        # TODO does this give the correct position?
        # TODO (not needed for accuracy, but for "real" tracking:) scale is still 0.5, so we need to multiply by 2
        position_predicted_1D = tf.reshape(tf.argmax(probs_1d, 1),[self.batch_size, 1])
        row = position_predicted_1D / self.output_width  # needs to be floored
        column = position_predicted_1D % self.output_width
        position_predicted_2D = tf.concat(1,[row,column])

        return position_predicted_2D

    def accuracy(self, x_prev, x_curr, y):
        num_iter = int(math.ceil(x_prev.shape[0] / self.batch_size))
        true_count = 0  # sum up all single accuracies
        total_sample_count = num_iter * self.batch_size
        step = 0
        while step < num_iter:
            feed_dict = {self.x_previous: x_prev[step * self.batch_size: (step + 1) * self.batch_size],
                         self.x_current: x_curr[step * self.batch_size: (step + 1) * self.batch_size],
                         self.placeholder_labels: y[step * self.batch_size: (step + 1) * self.batch_size],
                         self.dropout_prob: 1.0} #deactivate dropout for testing
            predictions = self.session.run(self.check_results, feed_dict)
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count
        if total_sample_count > 0:
            precision /= total_sample_count

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

