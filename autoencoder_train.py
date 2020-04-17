"""tensorboard --logdir=./tmp"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autoencoder_tf import Autoencoder
import tensorflow as tf
import numpy as np

print(tf.__version__)

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-2
intermediate_dim = 64
original_dim = 784


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)


(training_features, train_label), (test_features, test_label) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2])
training_features = training_features.astype('float32')
training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.shuffle(training_features.shape[0])
training_dataset = training_dataset.prefetch(batch_size * 4)

autoencoder = Autoencoder(intermediate_dim=64, original_dim=784)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

writer = tf.summary.create_file_writer('tmp')

with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            for step, batch_features in enumerate(training_dataset):
                train(loss, autoencoder, opt, batch_features)
                loss_values = loss(autoencoder, batch_features)
                original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                           (batch_features.shape[0], 28, 28, 1))
                tf.summary.scalar('loss', loss_values, step=step)
                tf.summary.image('original', original, max_outputs=10, step=step)
                tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
