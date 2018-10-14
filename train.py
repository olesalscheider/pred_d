#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import json
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
DATASET = 'contextpathpred_daimler_dataset_eccv14.json'
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
EPOCHS = 1000
DRAW_PLOTS = True
PLOT_DIR = 'plots'

if DRAW_PLOTS:
    shutil.rmtree(PLOT_DIR)
    os.makedirs(PLOT_DIR)

# Read dataset
def remove_list_idxs_and_convert(old_list, idxs):
    new_list = old_list
    for idx in reversed(idxs):
        del new_list[idx]
    return np.array(new_list, dtype=np.float32)

train_tracks_observed = []
train_tracks_gt = []
test_tracks_observed = []
test_tracks_gt = []
with open(DATASET, 'r') as dataset_file:
    json_dict = json.load(dataset_file)
    for track in json_dict['eccv14_tracks']:
        pos = track['gt_pos_e']
        x = pos[0]
        y = pos[1]
        # Handle nans in pos measurements
        nan_idx = []
        for idx in range(len(pos[0])):
            if x[idx] == '_NaN_' or y[idx] == '_NaN_':
                nan_idx.append(idx)
        x = remove_list_idxs_and_convert(x, nan_idx)
        y = remove_list_idxs_and_convert(y, nan_idx)

        dist_to_curb = track['curb_dist']
        dist_to_curb = remove_list_idxs_and_convert(dist_to_curb, nan_idx)

        sees_vehicle = track['sv']
        sees_vehicle = remove_list_idxs_and_convert(sees_vehicle, nan_idx)

        has_seen_vehicle = track['hsv']
        has_seen_vehicle = remove_list_idxs_and_convert(has_seen_vehicle, nan_idx)

        all_data = np.stack([x, y, dist_to_curb, sees_vehicle, has_seen_vehicle], axis=-1)
        x = np.expand_dims(x, axis=-1)

        # keep 70% of the tracks as training samples and use 30% for testing
        is_train_example = np.random.randint(0, 10) > 2

        # We want to predict the next 16 positions given the last 16 positions.
        # Extract as many subtracks as possible.
        for offset in range(len(dist_to_curb) - 2 * 16):
            if is_train_example:
                train_tracks_observed.append(all_data[offset:offset+16])
                train_tracks_gt.append(x[offset+16:offset+32])
            else:
                test_tracks_observed.append(all_data[offset:offset+16])
                test_tracks_gt.append(x[offset+16:offset+32])

# Create one numpy array from each list
num_samples = len(train_tracks_observed)
train_tracks_observed = np.stack(train_tracks_observed)
train_tracks_gt = np.stack(train_tracks_gt)
test_tracks_observed = np.stack(test_tracks_observed)
test_tracks_gt = np.stack(test_tracks_gt)

# Define a Keras model for the prediction
class PredictionModel(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

    def build(self, input_shapes):
        self.conv1 = tf.keras.layers.Conv1D(8, 3,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(),
            activation=tf.keras.activations.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(),
            activation=tf.keras.activations.relu)
        self.fc2 = tf.keras.layers.Dense(16,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.reshape(x, [-1, 16, 1])
        return x

# Train and evaluate the model
with tf.Graph().as_default():
    # Create tensorflow datasets from numpy arrays
    train_dataset = tf.data.Dataset.from_tensor_slices((train_tracks_observed, train_tracks_gt))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_tracks_observed, test_tracks_gt))

    # Shuffle, repeat and batch the train datset
    train_dataset = train_dataset.shuffle(num_samples)
    train_dataset = train_dataset.repeat(EPOCHS)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = test_dataset.batch(1)

    # Create a generic iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    # Create initializer operations for the iterator.
    # These assign either the test of train dataset.
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)


    session = tf.Session()

    # Instantiate the model we want to train
    net = PredictionModel('net')
    observed_data, labels = iterator.get_next()
    logits = net(observed_data)

    # Define the loss
    squared_diff = tf.pow(labels - logits, 2)
    loss_op = tf.reduce_sum(squared_diff)
    loss_op += tf.reduce_sum(net.losses) # Add regularizer losses

    # Define an OP to calculate the mean error
    mean_error_op = tf.reduce_mean(tf.abs(labels - logits))

    # Define the learning rate schedule and the optimizer OP.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use a polynomial decay
    learning_rate = LEARNING_RATE * (tf.constant(1.0, dtype=tf.float32) - (tf.cast(global_step, dtype=tf.float32) / tf.constant(EPOCHS * num_samples / BATCH_SIZE , dtype=tf.float32))) ** tf.constant(0.9, dtype=tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step)
    train_op = tf.group(train_op, net.updates)

    # Initialize the model variables (randomly).
    session.run(tf.global_variables_initializer())

    print('Train the model. This might take a while...')
    # Initialize the dataset iterator for training.
    session.run(train_init_op)
    i = 0
    while True:
        try:
            _, mean_error, loss = session.run([train_op, mean_error_op, loss_op])
            if i % 20 == 0:
                print('Step %4i - Mean error: %.4f, loss: %.4f' % (i, mean_error, loss))
            i += 1
        except tf.errors.OutOfRangeError:
            break # We finished!

    print('Evaluate the model. This might take a while...')
    # Initialize the dataset iterator for training.
    session.run(test_init_op)
    i = 0
    summed_error = 0.0
    while True:
        try:
            input_val, labels_val, logits_val, error = session.run([observed_data, labels, logits, mean_error_op])
            summed_error += error
            if DRAW_PLOTS:
                plt.scatter(range(16), input_val[0, :, 0], c='blue')
                plt.scatter(range(16, 32), labels_val[0, :, 0], c='red')
                plt.scatter(range(16, 32), logits_val[0, :, 0], c='green', marker='x')
                plt.savefig(PLOT_DIR + '/' + str(i) + '.png')
                plt.gcf().clear()
            i += 1
        except tf.errors.OutOfRangeError:
            break # We finished!
    print('Mean error on the test dataset: %.4f' % (summed_error / i))
