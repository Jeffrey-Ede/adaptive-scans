# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn

from dnc import dnc

import numpy as np
import cv2

from scipy import ndimage as nd

from PIL import Image

import os, sys

import time

from utility import alrc, auto_name

from scipy.stats import norm

experiment_number = 280


FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 64, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 0, "Maximum absolute value of controller and dnc outputs.")
tf.flags.DEFINE_bool("use_batch_norm", True, "Use batch normalization in generator.")

tf.flags.DEFINE_string("model", "LSTM", "LSTM or DNC.")

tf.flags.DEFINE_integer("projection_size", 0, "Size of projection layer. Zero for no projection.")
tf.flags.DEFINE_integer("lstm_depth", 2, "Number of LSTM cells deep.")

tf.flags.DEFINE_bool("is_input_embedder", False, "Embed inputs before they are input.")
tf.flags.DEFINE_bool("is_variable_initial_states", True, "Trainable initial states rather than zero states.")

tf.flags.DEFINE_bool("is_layer_norm", False, "Use layer normalization in recurrent networks.")

# Optimizer parameters.
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")

tf.flags.DEFINE_integer("replay_size", 32, "Maximum examples in ring buffer.")
tf.flags.DEFINE_integer("avg_replays", 1, "Mean frequency each experience is used.")
tf.flags.DEFINE_integer("replay_add_frequency", 1, "How often to add expereinces to ring buffer.")
tf.flags.DEFINE_bool("is_cyclic_replay", False, "True to cyclically replace experiences.")

tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("L2_norm", 1.e-5, "Decay rate for L2 regularization. 0 for no regularization.")

tf.flags.DEFINE_float("grad_clip_norm", 0, "Threshold to clip gradient sizes to. Zero for no clipping.")
tf.flags.DEFINE_float("grad_clip_value", 0.000, "Threshold to clip gradient sizes to. Zero for no clipping.")

# Task parameters
tf.flags.DEFINE_integer("img_side", 96, "Number of image pixels for square image")
tf.flags.DEFINE_integer("num_steps", 20, "Number of image pixels for square image")
tf.flags.DEFINE_integer("step_size", 20, "Distance STEM probe moves at each step (in px).")
tf.flags.DEFINE_integer("num_actions", 1, "Number of parameters to describe actions.")
tf.flags.DEFINE_integer("shuffle_size", 2000, "Size of moving buffer to sample data from.")
tf.flags.DEFINE_integer("prefetch_size", 10, "Number of batches to prepare in advance.")

# Training options.
tf.flags.DEFINE_float("actor_lr", 0.0005, "Actor learning rate.")
tf.flags.DEFINE_float("critic_lr", 0.001, "Critic learning rate.")
tf.flags.DEFINE_float("generator_lr", 0.003, "Generator learning rate.")
tf.flags.DEFINE_float("discriminator_lr", 0.003, "Discriminator learning rate.")

tf.flags.DEFINE_float("specificity", 0.0, "Weight specificity loss. Zero for no specificity loss.")
tf.flags.DEFINE_float("loss_norm_decay", 0.997, "Weight for loss normalization. Zero for no normalization.")
tf.flags.DEFINE_float("rnn_norm_decay", 0., "Weight for critic loss normalization. Zero for no normalization.")

tf.flags.DEFINE_bool("is_immediate_critic_loss", False, "True to not backpropagate.")

tf.flags.DEFINE_bool("is_target_actor_feedback", False, "True to use target critic to train actor.")
tf.flags.DEFINE_bool("is_ranked_loss", False, "Loses are indices in sorted arrays.")


tf.flags.DEFINE_float("gamma", 0.97, "Reward/loss decay.")
tf.flags.DEFINE_float("loss_gamma", 1., "Reward/loss decay applied directly to losses.")

tf.flags.DEFINE_float("loss_norm_clip", 0., "Norm to clip losses to.")

tf.flags.DEFINE_bool("is_advantage_actor_critic", False, "Use advantage rather than Q errors for actor.")
tf.flags.DEFINE_bool("is_direct_advantage", False, "Use predicted Q minus exploration Q errors for actor.")

tf.flags.DEFINE_bool("is_cyclic_generator_learning_rate", True, "True for sawtooth oscillations.")
tf.flags.DEFINE_bool("is_decaying_generator_learning_rate", True, "True for decay envelope for sawtooth oscillations.")

tf.flags.DEFINE_integer("supervision_iters", 1, "Starting value for supeversion.")
tf.flags.DEFINE_float("supervision_start", 0., "Starting value for supeversion.")
tf.flags.DEFINE_float("supervision_end", 0., "Starting value for supeversion.")

if FLAGS.supervision_iters:
    #Flag will not be used
    tf.flags.DEFINE_float("supervision", 0.5, "Weighting for known discounted future reward.")
else:
    #Flag will be used
    tf.flags.DEFINE_float("supervision", 0.0, "Weighting for known discounted future reward.")

tf.flags.DEFINE_bool("is_target_actor", False and FLAGS.supervision != 1, "True to use target actor.")
tf.flags.DEFINE_bool("is_target_critic", True and FLAGS.supervision != 1, "True to use target critic.")
tf.flags.DEFINE_bool("is_target_generator", False, "True to use target generator.")

tf.flags.DEFINE_integer("update_frequency", 0, "Frequency of hard target network updates. Zero for soft updates.")
tf.flags.DEFINE_float("target_decay", 0.997, "Decay rate for target network soft updates.")
tf.flags.DEFINE_bool("is_generator_batch_norm_tracked", False, "True to track generator batch normalization.")

tf.flags.DEFINE_bool("is_positive_qs", False, "Whether to clip qs to be positive.")
tf.flags.DEFINE_bool("is_norm_q", False, "Whether to set mean of q values.")

tf.flags.DEFINE_bool("is_infilled", False, "True to use infilling rather than generator.")

tf.flags.DEFINE_bool("is_prev_position_input", True, "True to input previous positions.")

tf.flags.DEFINE_bool("is_ornstein_uhlenbeck", True, "True for O-U exploration noise.")
tf.flags.DEFINE_bool("is_noise_decay", True, "Decay noise if true.")
tf.flags.DEFINE_float("ou_theta", 0.1, "Drift back to mean.")
tf.flags.DEFINE_float("ou_sigma", 0.2, "Size of random process.")

tf.flags.DEFINE_float("exploration_loss", 0., "Exploration loss.")
tf.flags.DEFINE_float("uniform_coverage_loss", 0, "Additive uniform coverage loss.")

tf.flags.DEFINE_bool("is_rel_to_truth", False, "True to normalize losses using expected losses.")

tf.flags.DEFINE_bool("is_clipped_reward", True, "True to clip rewards.")
tf.flags.DEFINE_bool("is_clipped_critic", False, "True to clip critic predictions for actor training.")

tf.flags.DEFINE_float("over_edge_penalty", 0.05, "Penalty for action going over edge of image.")
tf.flags.DEFINE_float("end_edge_penalty", 0.0, "Penalty for action going over edge of image.")

tf.flags.DEFINE_bool("is_prioritized_replay", False, "True to prioritize the replay of difficult experiences.")
tf.flags.DEFINE_bool("is_biased_prioritized_replay", False, "Priority sampling without bias correction.")

tf.flags.DEFINE_bool("is_relative_to_spirals", False, "True to compare generator losses against losses for spirals.")

tf.flags.DEFINE_bool("is_self_competition", True, "Oh it is on. True to compete against past versions of itself.")

tf.flags.DEFINE_float("norm_generator_losses_decay", 0., "Divide generator losses by their running mean. Zero for no normalization.")
tf.flags.DEFINE_float("replay_decay", 0.999, "Decay rates to calculate loss moments.")

tf.flags.DEFINE_bool("is_minmax_reward", False, "True to use highest losses for actor loss.")

tf.flags.DEFINE_integer("start_iter", 0, "Starting iteration")
tf.flags.DEFINE_integer("train_iters", 1_000_000, "Training iterations")
tf.flags.DEFINE_integer("val_examples", 20_000, "Number of validation examples")

tf.flags.DEFINE_float("style_loss", 0., "Weighting of style loss. Zero for no style loss.")
tf.flags.DEFINE_float("spike_loss", 0., "Penalize critic for spikey predictions.")

tf.flags.DEFINE_float("step_incr", np.sqrt(2), "Number of pixels per step.")

tf.flags.DEFINE_string("model_dir", 
                       f"//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/recurrent_conv-1/{experiment_number}/", 
                       "Working directory.")
tf.flags.DEFINE_string("data_file",
                       "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
                       "Datafile containing 19769 96x96 downsampled STEM crops.")

tf.flags.DEFINE_integer("report_freq", 10, "How often to print losses to the console.")


os.chdir(FLAGS.model_dir)
sys.path.insert(0, FLAGS.model_dir)


def l1_batch_norm(x):

    mu = tf.reduce_mean(x, axis=0, keepdims=True)
    d = np.sqrt(np.pi/2)*tf.reduce_mean(tf.abs(x-mu))

    return (x - mu) / d


def sobel_edges(image):
  """Returns a tensor holding Sobel edge maps.
  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.
  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

  kernels_tf = array_ops.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output


def norm_img(img, min=None, max=None, get_min_and_max=False):
    
    if min == None:
        min = np.min(img)
    if max == None:
        max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    if get_min_and_max:
        return img.astype(np.float32), (min, max)
    else:
        return img.astype(np.float32)


def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)

def disp(img):
    #if len(img.shape) == 3:
    #    img = np.sum(img, axis=2)
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return


def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
      "use_layer_norm": FLAGS.is_layer_norm,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence


def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

class RingBuffer(object):

    def __init__(
        self, 
        action_shape,
        observation_shape,
        full_scan_shape,
        batch_size,
        buffer_size=1000,
        num_past_losses=None,
        ):

        self.buffer_size = buffer_size

        self.actions = np.zeros([buffer_size]+list(action_shape)[1:])
        self.observations = np.zeros([buffer_size]+list(observation_shape)[1:])
        self.full_scans = np.zeros([buffer_size]+list(full_scan_shape)[1:])

        self.position = 0

        self._batch_size = batch_size
        self._input_batch_size = batch_size // FLAGS.avg_replays

        if FLAGS.is_prioritized_replay or FLAGS.is_biased_prioritized_replay:
            self.accesses = np.zeros([buffer_size])
            self.priorities = np.ones([buffer_size])
            self.indices = np.arange(buffer_size)

            self._is_accessed = False

        if FLAGS.is_self_competition:
            self.mu1 = 0.5
            self.mu2 = 0.5
            self.std = np.sqrt(self.mu2 - self.mu1**2 + 1.e-3)

            self.d_mu1 = 0.5
            self.d_mu2 = 0.5
            self.d_std = np.sqrt(self.d_mu2 - self.d_mu1**2 + 1.e-3)

            self.past_losses = np.zeros([num_past_losses])
            self.next_losses = np.zeros([num_past_losses])
            self.labels = np.zeros([buffer_size], np.int32)
        
    def add(self, actions, observations, full_scans, labels=None):

        if FLAGS.is_cyclic_replay:
            i0 = self.position % self.buffer_size
        else:
            if self.position < self.buffer_size:
                i0 = self.position
            else:
                i0 = np.random.randint(0, self.buffer_size)

        num_before_cycle = min(self.buffer_size-i0, self._input_batch_size)

        self.actions[i0:i0+num_before_cycle] = actions[:num_before_cycle]
        self.observations[i0:i0+num_before_cycle] = observations[:num_before_cycle]
        self.full_scans[i0:i0+num_before_cycle] = full_scans[:num_before_cycle]

        num_remaining = self._input_batch_size - num_before_cycle
        if num_remaining > 0:
            self.actions[0:num_remaining] = actions[num_before_cycle:]
            self.observations[:num_remaining] = observations[num_before_cycle:]
            self.full_scans[:num_remaining] = full_scans[num_before_cycle:]

        if FLAGS.is_prioritized_replay or FLAGS.is_biased_prioritized_replay:
            if self._is_accessed:
                mean_priority = np.mean(self.priorities[self.accesses > 0])
            else:
                mean_priority = 1.
            self.priorities[i0:i0+num_before_cycle] = mean_priority*np.ones([num_before_cycle])
            if num_before_cycle < self._input_batch_size:
                self.priorities[0:num_remaining] = mean_priority*np.ones([self._input_batch_size - num_before_cycle])

        if FLAGS.is_self_competition:
            self.labels[i0:i0+num_before_cycle] = labels[:num_before_cycle]
            if num_remaining > 0:
                self.labels[0:num_remaining] = labels[num_before_cycle:]

            #self.past_losses[labels] = self.next_losses[labels]
            #self.next_losses[labels] = losses

            #self.mu1 = FLAGS.replay_decay*self.mu1 + (1-FLAGS.replay_decay)*np.mean(losses)
            #self.mu2 = FLAGS.replay_decay*self.mu2 + (1-FLAGS.replay_decay)*np.mean(losses**2)
            #self.std = np.sqrt(self.mu2 - self.mu1**2 + 1.e-3)

            #diffs = losses - self.past_losses[labels]

            #self.d_mu1 = FLAGS.replay_decay*self.d_mu1 + (1-FLAGS.replay_decay)*np.mean(diffs)
            #self.d_mu2 = FLAGS.replay_decay*self.d_mu2 + (1-FLAGS.replay_decay)*np.mean(diffs**2)
            #self.d_std = np.sqrt(self.d_mu2 - self.d_mu1**2 + 1.e-3)

        self.position += self._input_batch_size

    def loss_fn(self, next_loss, prev_loss):
        mse_loss = next_loss#(next_loss - self.mu1) / self.std
        #exp_loss = (next_loss - prev_loss - self.d_mu1) / self.d_std

        loss = mse_loss#sigmoid(mse_loss) # sigmoid(exp_loss)

        return loss

    def get(self):
        
        limit = min(self.position, self.buffer_size)

        if FLAGS.is_prioritized_replay:
            sample_idxs = np.random.choice(
                self.indices, 
                size=self._batch_size, 
                replace=False, 
                p=self.priorities/np.sum(self.priorities)
                ) #alpha=1

            beta = 0.5 + 0.5*(FLAGS.train_iters - self.position)/FLAGS.train_iters

            sampled_priority_weights = self.priorities[sample_idxs]**( -beta )
            sampled_priority_weights /= np.max(sampled_priority_weights)
        elif FLAGS.is_biased_prioritized_replay:
            alpha = 1#(FLAGS.train_iters - self.position)/FLAGS.train_iters
            priorities = self.priorities**alpha
            sample_idxs = np.random.choice(
                self.indices, 
                size=self._batch_size, 
                replace=False, 
                p=self.priorities/np.sum(self.priorities)
                )
        else:
            sample_idxs = np.random.randint(0, limit, size=self._batch_size)

        sampled_actions = np.stack([self.actions[i] for i in sample_idxs])
        sampled_observations = np.stack([self.observations[i] for i in sample_idxs])
        sampled_full_scans = np.stack([self.full_scans[i] for i in sample_idxs])

        if FLAGS.is_prioritized_replay:
            return sampled_actions, sampled_observations, sample_idxs, sampled_priority_weights
        #elif FLAGS.is_biased_prioritized_replay:
            #return sampled_actions, sampled_observations, sample_idxs
        elif FLAGS.is_self_competition:
            sampled_labels = np.stack([self.labels[i] for i in sample_idxs])
            sampled_past_losses = np.stack([self.next_losses[i] for i in sampled_labels])
            return sampled_actions, sampled_observations, sample_idxs, sampled_past_losses, sampled_full_scans
        else:
            return sampled_actions, sampled_observations

    def update_priorities(self, idxs, priorities):
        """For prioritized experience replay"""

        self._is_accessed = True

        self.priorities[idxs] = priorities
        self.accesses[idxs] = 1.
    
class Agent(snt.AbstractModule):

    def __init__(
        self, 
        num_outputs,
        name,
        is_new=False,
        noise_decay=None,
        is_double_critic=False,
        sampled_full_scans=None,
        val_full_scans=None
        ):

        super(Agent, self).__init__(name=name)
    
        access_config = {
            "memory_size": FLAGS.memory_size,
            "word_size": FLAGS.word_size,
            "num_reads": FLAGS.num_read_heads,
            "num_writes": FLAGS.num_write_heads,
        }
        controller_config = {
            "hidden_size": FLAGS.hidden_size,
            #"projection_size": FLAGS.projection_size or None,
        }
        clip_value = FLAGS.clip_value

        self.is_actor = "actor" in name

        with self._enter_variable_scope():

            components = dnc.Components(access_config, controller_config, num_outputs)

            self._dnc_core = dnc.DNC(
                components, 
                num_outputs, 
                clip_value, 
                is_new=False, 
                is_double_critic=is_double_critic,
                is_actor=self.is_actor)

            if is_new:
                self._dnc_core_new = dnc.DNC(
                    components, 
                    num_outputs, 
                    clip_value, 
                    is_new=True,
                    noise_decay=noise_decay,
                    sampled_full_scans=sampled_full_scans,
                    is_noise=True,
                    is_actor=self.is_actor
                    )

                if not val_full_scans is None:
                    self._dnc_core_val = dnc.DNC(
                        components, 
                        num_outputs, 
                        clip_value, 
                        is_new=True, 
                        sampled_full_scans=val_full_scans,
                        is_actor=self.is_actor
                        )

            self._initial_state = self._dnc_core.initial_state(FLAGS.batch_size)
            self._new_start = self._dnc_core.initial_state(FLAGS.batch_size // FLAGS.avg_replays)

            #self._action_embedder = snt.Linear(output_size=64)
            #self._observation_embedder = snt.Linear(output_size=64)

    def _build(self, observations, actions):

        #Tiling here is a hack to make inputs the same size
        num_tiles = 2 // (actions.get_shape().as_list()[-1] // FLAGS.num_actions)
        tiled_actions = tf.tile(actions, [1, 1, num_tiles])
        input_sequence = tf.concat([observations, tiled_actions], axis=-1)

        self._dnc_core.first = True

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=self._dnc_core,
            inputs=input_sequence,
            time_major=False,
            initial_state=self._initial_state
            )

        return output_sequence

    def get_new_experience(self):

        self._dnc_core_new.first = True

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=self._dnc_core_new,
            inputs=tf.zeros([FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, 1]),
            time_major=False,
            initial_state=self._new_start
            )

        if hasattr(tf, 'ensure_shape'):
            output_sequence = tf.ensure_shape(
                output_sequence, 
                [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, FLAGS.step_size+FLAGS.num_actions])
        else:
            output_sequence = tf.reshape(
                output_sequence, 
                [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, FLAGS.step_size+FLAGS.num_actions])

        observations = output_sequence[:,:,:FLAGS.step_size]
        actions = output_sequence[:,:,FLAGS.step_size:]

        return observations, actions

    def get_val_experience(self):

        self._dnc_core_val.first = True

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=self._dnc_core_val,
            inputs=tf.zeros([FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, 1]),
            time_major=False,
            initial_state=self._new_start
            )

        if hasattr(tf, 'ensure_shape'):
            output_sequence = tf.ensure_shape(
                output_sequence, 
                [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, FLAGS.step_size+FLAGS.num_actions])
        else:
            output_sequence = tf.reshape(
                output_sequence, 
                [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.num_steps, FLAGS.step_size+FLAGS.num_actions])

        observations = output_sequence[:,:,:FLAGS.step_size]
        actions = output_sequence[:,:,FLAGS.step_size:]

        return observations, actions

    @property
    def variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope=tf.get_variable_scope().name
                )

    @property
    def trainable_variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=tf.get_variable_scope().name
                )


def spectral_norm(w, iteration=1, in_place_updates=False):
    """Spectral normalization. It imposes Lipschitz continuity by constraining the
    spectral norm (maximum singular value) of weight matrices.

    Inputs:
        w: Weight matrix to spectrally normalize.
        iteration: Number of times to apply the power iteration method to 
        enforce spectral norm.

    Returns:
        Weight matrix with spectral normalization control dependencies.
    """

    w0 = w
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])


    u = tf.get_variable(auto_name("u"), 
                       [1, w_shape[-1]], 
                       initializer=tf.random_normal_initializer(mean=0.,stddev=0.03), 
                       trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    if in_place_updates:
        #In-place control dependencies bottlenect training
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
    else:
        #Execute control dependency in parallel with other update ops
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u.assign(u_hat))

        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def spectral_norm_conv(
    inputs,
    num_outputs, 
    stride=1, 
    kernel_size=3, 
    padding='VALID',
    biases_initializer=tf.zeros_initializer()
    ):
    """Convolutional layer with spectrally normalized weights."""
    
    w = tf.get_variable(auto_name("kernel"), shape=[kernel_size, kernel_size, inputs.get_shape()[-1], num_outputs])

    x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w), 
                        strides=[1, stride, stride, 1], padding=padding)

    if biases_initializer != None:
        b = tf.get_variable(auto_name("bias"), [num_outputs], initializer=biases_initializer)
        x = tf.nn.bias_add(x, b)

    return x


def conv(
    inputs, 
    num_outputs, 
    kernel_size=3, 
    stride=1, 
    padding='SAME',
    data_format="NHWC",
    actv_fn=tf.nn.relu, 
    is_batch_norm=True,
    is_spectral_norm=False,
    is_depthwise_sep=False,
    extra_batch_norm=False,
    biases_initializer=tf.zeros_initializer,
    weights_initializer=initializers.xavier_initializer,
    transpose=False,
    is_training=True
    ):
    """Convenience function for a strided convolutional or transpositional 
    convolutional layer.
    
    Intro: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1.

    The order is: Activation (Optional) -> Batch Normalization (optional) -> Convolutions.

    Inputs: 
        inputs: Tensor of shape `[batch_size, height, width, channels]` to apply
        convolutions to.
        num_outputs: Number of feature channels to output.
        kernel_size: Side lenth of square convolutional kernels.
        stride: Distance between convolutional kernel applications.
        padding: 'SAME' for zero padding where kernels go over the edge.
        'VALID' to discard features where kernels go over the edge.
        activ_fn: non-linearity to apply after summing convolutions. 
        is_batch_norm: If True, add batch normalization after activation.
        is_spectral_norm: If True, spectrally normalize weights.
        is_depthwise_sep: If True, depthwise separate convolutions into depthwise
        spatial convolutions, then 1x1 pointwise convolutions.
        extra_batch_norm: If True and convolutions are depthwise separable, implement
        batch normalization between depthwise and pointwise convolutions.
        biases_initializer: Function to initialize biases with. None for no biases.
        weights_initializer: Function to initialize weights with. None for no weights.
        transpose: If True, apply convolutional layer transpositionally to the
        described convolutional layer.
        is_training: If True, use training specific operations e.g. batch normalization
        update ops.

    Returns:
        Output of convolutional layer.
    """

    x = inputs

    num_spatial_dims = len(x.get_shape().as_list()) - 2

    if biases_initializer == None:
        biases_initializer = lambda: None
    if weights_initializer == None:
        weights_initializer = lambda: None

    if not is_spectral_norm:
        #Convolutional layer without spectral normalization

        if transpose:
            stride0 = 1
            if type(stride) == list or is_depthwise_sep or stride % 1:
                #Apparently there is no implementation of transpositional  
                #depthwise separable convolutions, so bilinearly upsample then 
                #depthwise separably convolute
                if kernel_size != 1:
                    x = tf.image.resize_bilinear(
                        images=x,
                        size=stride if type(stride) == list else \
                        [int(stride*d) for d in x.get_shape().as_list()[1:3]],
                        align_corners=True
                        )
                stride0 = stride      
                stride = 1

            if type(stride0) == list and not is_depthwise_sep:
                layer = tf.contrib.layers.conv2d
            elif is_depthwise_sep:
                layer = tf.contrib.layers.separable_conv2d
            else:
                layer = tf.contrib.layers.conv2d_transpose

            x = layer(
                inputs=x,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format=data_format,
                activation_fn=None,
                weights_initializer=weights_initializer(),
                biases_initializer=biases_initializer())
               
            if type(stride0) != list:
              if (is_depthwise_sep or stride0 % 1) and kernel_size == 1:
                  x = tf.image.resize_bilinear(
                      images=x,
                      size=[int(stride0*d) for d in x.get_shape().as_list()[1:3]],
                      align_corners=True
                      )   
        else:
            if num_spatial_dims == 1:
                layer = tf.contrib.layers.conv1d
            elif num_spatial_dims == 2:
                if is_depthwise_sep:
                    layer = tf.contrib.layers.separable_conv2d
                else:
                    layer = tf.contrib.layers.conv2d
            x = layer(
                inputs=x,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format=data_format,
                activation_fn=None,
                weights_initializer=weights_initializer(),
                biases_initializer=biases_initializer())
    else:
        #Weights are spectrally normalized
        x = spectral_norm_conv(
            inputs=x, 
            num_outputs=num_outputs, 
            stride=stride, 
            kernel_size=kernel_size, 
            padding=padding, 
            biases_initializer=biases_initializer())

    if actv_fn:
        x = actv_fn(x)

    if is_batch_norm and FLAGS.use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        #x = l1_batch_norm(x)

    return x


def residual_block(inputs, skip=3, is_training=True):
    """Residual block whre the input is added to the signal after skipping some
    layers. This architecture is good for learning purturbative transformations. 
    If no layer is provided, it defaults to a convolutional layer.

    Deep residual learning: https://arxiv.org/abs/1512.03385.

    Inputs:
        inputs: Tensor to apply residual block to. Outputs of every layer will 
        have the same shape.
        skip: Number of layers to skip before adding input to layer output.
        layer: Layer to apply in residual block. Defaults to convolutional 
        layer. Custom layers must support `inputs`, `num_outputs` and `is_training`
        arguments.

    Returns:
        Final output of residual block.
    """

    x = x0 = inputs

    def layer(inputs, num_outputs, is_training, is_batch_norm, actv_fn):
        
        x = conv(
            inputs=inputs, 
            num_outputs=num_outputs,
            is_training=is_training,
            actv_fn=actv_fn
            )

        return x

    for i in range(skip):
        x = layer(
            inputs=x, 
            num_outputs=x.get_shape()[-1], 
            is_training=is_training,
            is_batch_norm=i < skip - 1,
            actv_fn=tf.nn.relu
        )

    x += x0

    if FLAGS.use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)
        #x = l1_batch_norm(x)

    return x


class Discriminator(snt.AbstractModule):

    def __init__(self, 
                 name, 
                 is_training
                 ):

        super(Discriminator, self).__init__(name=name)

        self._is_training = is_training


    def _build(self, inputs):

        x = inputs
        std_actv = tf.nn.relu#lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        is_training = self._is_training
        is_depthwise_sep = False

        base_size = 32
    
        for i in range(0, 5):

            x = conv(
                x, 
                num_outputs=base_size*2**i, 
                kernel_size=4,
                stride=2,
                is_depthwise_sep=is_depthwise_sep,
                is_training=is_training,
                actv_fn=std_actv
            )

        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)

        return x

    @property
    def variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope=tf.get_variable_scope().name
                )

    @property
    def trainable_variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=tf.get_variable_scope().name
                )


class Generator(snt.AbstractModule):

    def __init__(self, 
                 name, 
                 is_training
                 ):

        super(Generator, self).__init__(name=name)

        self._is_training = is_training


    def _build(self, inputs):

        x = inputs
        std_actv = tf.nn.relu#lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        is_training = self._is_training
        is_depthwise_sep = False

        base_size = 32

        #x = tf.contrib.layers.batch_norm(x, is_training=is_training)

        x = conv(
            x, 
            num_outputs=32,
            is_training=is_training,
            actv_fn=std_actv
            )
    
        #Encoder
        for i in range(1, 3):

            x = conv(
                x, 
                num_outputs=base_size*2**i, 
                stride=2,
                is_depthwise_sep=is_depthwise_sep,
                is_training=is_training,
                actv_fn=std_actv
            )

            if i == 2:
                low_level = x

        #Residual blocks
        for _ in range(5): #Number of blocks
            x = residual_block(
                x, 
                skip=3,
                is_training=is_training
            )


        #Decoder
        for i in range(1, -1, -1):

            x = conv(
                x, 
                num_outputs=base_size*2**i, 
                stride=2,
                is_depthwise_sep=is_depthwise_sep,
                is_training=is_training,
                transpose=True,
                actv_fn=std_actv
            )

        x = conv(
            x, 
            num_outputs=base_size, 
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training
        )

  
        #Project features onto output image
        x = conv(
            x,
            num_outputs=1,
            biases_initializer=None,
            actv_fn=None,
            is_batch_norm=False,
            is_training=is_training
        )

        return x

    @property
    def variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope=tf.get_variable_scope().name
                )

    @property
    def trainable_variables(self):
        with self._enter_variable_scope():
            return tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 
                scope=tf.get_variable_scope().name
                )


def construct_partial_scans(actions, observations):
    """
    actions: [batch_size, num_steps, 2]
    observations: [batch_size, num_steps, 10]
    """

    if FLAGS.num_actions == 1:
        actions = FLAGS.step_incr*np.concatenate((np.cos(actions), np.sin(actions)), axis=-1)

    #Last action unused and the first action is always the same
    actions = np.concatenate((np.ones([FLAGS.batch_size // FLAGS.avg_replays, 1, 2]), actions[:,:-1,:]), axis=1)

    starts = 0.5*FLAGS.img_side + FLAGS.step_size*(np.cumsum(actions, axis=1) - actions)

    #starts = np.zeros(actions.shape)
    #starts[:,0,:] = actions[:,0,:]
    #for i in range(1, FLAGS.num_steps):
    #    starts[:,i,:] = actions[:,i,:] + starts[:,i-1,:]
    #starts -= actions
    #starts *= FLAGS.step_size
    #starts += 0.5*FLAGS.img_side

    positions = np.stack([starts + i*actions for i in range(FLAGS.step_size)], axis=-2)
    x = np.minimum(np.maximum(positions, 0), FLAGS.img_side-1)

    indices = []
    for j in range(FLAGS.batch_size // FLAGS.avg_replays):
        for k in range(FLAGS.num_steps):
            for i in range(FLAGS.step_size):
                indices.append( [j, int(x[j,k,i,0]), int(x[j,k,i,1])] )
    indices = np.array(indices)
    indices = tuple([indices[:,i] for i in range(3)])

    partial_scans = np.zeros([FLAGS.batch_size // FLAGS.avg_replays, FLAGS.img_side, FLAGS.img_side])
    masks = np.zeros([FLAGS.batch_size // FLAGS.avg_replays, FLAGS.img_side, FLAGS.img_side])
    
    partial_scans[indices] = observations.reshape([-1])
    masks[indices] = 1

    partial_scans /= np.maximum(masks, 1)
    masks = np.minimum(masks, 1)

    partial_scans = np.stack([partial_scans, masks], axis=-1)

    return partial_scans


def target_update_ops(target_network, network, decay=FLAGS.target_decay, l2_norm=False):

    t_vars = target_network.variables
    v_vars = network.variables

    update_ops = []
    for t, v in zip(t_vars, v_vars):
        if FLAGS.is_generator_batch_norm_tracked or not "BatchNorm" in t.name: #Don't track batch normalization
            if l2_norm:
                v_new = (1-FLAGS.L2_norm)*v
                op = v.assign(v_new)
                update_ops.append(op)
            else:
                v_new = v
            op = t.assign(decay*t + (1-decay)*v_new)
            update_ops.append(op)

        #print(t.name.replace("target_", "") == v.name, t.name.replace("target_", ""), v.name)
    return update_ops

def weight_decay_ops(network, weight_decay):

    v_vars = network.variables

    update_ops = []
    for v in v_vars:
        v_new = (1-weight_decay)*v
        op = v.assign(v_new)
        update_ops.append(op)

    return update_ops
    

def augmenter(image):
    image = flip_rotate(image, np.random.randint(0, 8))
    image = cv2.GaussianBlur(image, (5, 5), 2.5, None, 2.5)
    return image


def load_data(shape):

    data_ph = tf.placeholder(tf.float32, shape=list(shape))

    ds = tf.data.Dataset.from_tensor_slices(tuple([data_ph]))

    ds = ds.map(lambda image: tf.reshape(tf.py_func(augmenter, [image], [tf.float32]), [FLAGS.img_side, FLAGS.img_side, 1]))

    if FLAGS.is_self_competition:
        labels = tf.data.Dataset.range(0, list(shape)[0])
        ds = tf.data.Dataset.zip((ds, labels))

    ds = ds.shuffle(buffer_size=FLAGS.shuffle_size)
    ds = ds.repeat()
    ds = ds.batch(FLAGS.batch_size // FLAGS.avg_replays)
    ds = ds.prefetch(FLAGS.prefetch_size)

    iterator = ds.make_initializable_iterator()

    return data_ph, iterator


@tf.custom_gradient
def overwrite_grads(x, y):
    print("OG", x, y)
    def grad(dy):
        return y, None

    return x, grad

def infill(data, mask):
    #b = mask > 0
    #data[b] = cv2.GaussianBlur(data, (5, 5), 2.5, None, 2.5)[b] / \
    #    cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 2.5, None, 2.5)[b]
    return data[tuple(nd.distance_transform_edt(np.equal(mask, 0), return_distances=False, return_indices=True))]

#def infill(data, mask):
#    return data[tuple(nd.distance_transform_edt(np.equal(mask, 0), return_distances=False, return_indices=True))]

#def infill(data, mask):
#    x = np.zeros(data.shape)
#    c = (cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 3.5, None, 3.5) > 0).astype(np.float32)
#    truth = data[tuple(nd.distance_transform_edt(np.equal(mask, 0), return_distances=False, return_indices=True))]
#    x = (truth*c).astype(np.float32)
#    return x

def fill(input):
    return np.expand_dims(np.stack([infill(img, mask) for img, mask in zip(input[:,:,:,0], input[:,:,:,1])]), -1)


def flip_rotate(img, choice):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""
    
    if choice == 0:
        return img
    elif choice == 1:
        return np.rot90(img, 1)
    elif choice == 2:
        return np.rot90(img, 2)
    elif choice == 3:
        return np.rot90(img, 3)
    elif choice == 4:
        return np.flip(img, 0)
    elif choice == 5:
        return np.flip(img, 1)
    elif choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    else:
        return np.flip(np.rot90(img, 1), 1)


def draw_spiral(coverage, side, num_steps=10_000):
    """Duration spent at each location as a particle falls in a magnetic 
    field. Trajectory chosen so that the duration density is (approx.)
    evenly distributed. Trajectory is calculated stepwise.
    
    Args: 
        coverage: Average amount of time spent at a random pixel
        side: Sidelength of square image that the motion is 
        inscribed on.

    Returns:
        A spiral
    """
    #Use size that is larger than the image
    size = int(np.ceil(np.sqrt(2)*side))

    #Maximum radius of motion
    R = size/2

    #Get constant in equation of motion 
    k = 1/ (2*np.pi*coverage)

    #Maximum theta that is in the image
    theta_max = R / k

    #Equispaced steps
    theta = np.arange(0, theta_max, theta_max/num_steps)
    r = k * theta

    #Convert to cartesian, with (0,0) at the center of the image
    x = r*np.cos(theta) + R
    y = r*np.sin(theta) + R

    #Draw spiral
    z = np.empty((x.size + y.size,), dtype=x.dtype)
    z[0::2] = x
    z[1::2] = y

    z = list(z)

    img = Image.new('F', (size,size), "black")
    img_draw = ImageDraw.Draw(img)
    img_draw = img_draw.line(z)
    
    img = np.asarray(img)
    img = img[size//2-side//2:size//2+side//2+side%2, 
              size//2-side//2:size//2+side//2+side%2]

    return img


def average_filter(image):

    kernel = tf.ones([5,5,1,1])
    filtered_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="VALID")

    return filtered_image

def pad(tensor, size):
    d1_pad = size[0]
    d2_pad = size[1]

    paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
    padded = tf.pad(tensor, paddings, mode="REFLECT")
    return padded

def gaussian_kernel(size: int,
                mean: float,
                std: float,
                ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def blur(image, size=2, std_dev=2.5, is_pad=True):
    gauss_kernel = gaussian_kernel( size, 0., std_dev )

    #Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    #Convolve
    if is_pad:
        image = pad(image, (size,size))
    return tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

def calc_generator_losses(img1, img2):

    #if FLAGS.data_file == "//Desktop-sa1evjv/h/96x96_stem_crops.npy":
    #    img2 = blur(img2) #Gaussian blur

    c = 10 #scale factor for historical reasons... Does nothing after first iterations
    generator_losses = c*tf.reduce_mean( (img1 - img2)**2, axis=[1,2,3] )
    losses = generator_losses

    if FLAGS.style_loss:
        edges1 = sobel_edges(img1)
        edges2 = sobel_edges(img2)
        generator_losses += FLAGS.style_loss*c*tf.reduce_mean( (edges1 - edges2)**2, axis=[1,2,3,4] )

    return generator_losses, losses


def construct_scans(sampled_actions, sampled_observations, replay_sampled_full_scans):

    replay_partial_scans = construct_partial_scans(sampled_actions, sampled_observations)

    if not FLAGS.is_infilled:
        sampled_full_scans = []
        partial_scans = []
        spiral_scans = []
        for sampled_full_scan, partial_scan in zip(replay_sampled_full_scans, replay_partial_scans):
            c = np.random.randint(0, 8)
            sampled_full_scans.append( flip_rotate(sampled_full_scan, c) )
            partial_scans.append( flip_rotate(partial_scan, c) )

            if FLAGS.is_relative_to_spirals:
                spiral_scan = spiral * sampled_full_scan
                spiral_scans.append( flip_rotate(spiral_scan, c) )

        sampled_full_scans = np.stack( sampled_full_scans )
        partial_scans = np.stack( partial_scans )
    else:
        sampled_full_scans = replay_sampled_full_scans
        partial_scans = replay_partial_scans

    return partial_scans.astype(np.float32), sampled_full_scans.astype(np.float32)


def avg_bn(x):
    
    if FLAGS.is_infilled:
        print("HELLO")
        y = 3*x
        return y
    else:
        mu = tf.reduce_mean(x)
        avg_mu = tf.get_variable(auto_name("avg_mu"), initializer=tf.constant(0.5, dtype=tf.float32))

        update_op = avg_mu.assign(FLAGS.loss_norm_decay*avg_mu+(1-FLAGS.loss_norm_decay)*mu)

        with tf.control_dependencies([update_op]):
            return x / tf.stop_gradient(avg_mu)

    #mu2 = tf.reduce_mean(x**2)

    #avg_mu = tf.get_variable(auto_name("avg_mu"), initializer=tf.constant(0.5, dtype=tf.float32))
    #avg_mu2 = tf.get_variable(auto_name("avg_mu2"), initializer=tf.constant(1, dtype=tf.float32))

    #update_ops = [avg_mu.assign(FLAGS.loss_norm_decay*avg_mu+(1-FLAGS.loss_norm_decay)*mu), 
    #              avg_mu2.assign(FLAGS.loss_norm_decay*avg_mu2+(1-FLAGS.loss_norm_decay)*mu2)]

    #std = tf.sqrt(avg_mu2 - avg_mu**2 + 1.e-6)

    #with tf.control_dependencies(update_ops):
    #    return tf.sigmoid( (x - avg_mu) / std )


def rnn_loss_norm(loss):

    mean = tf.reduce_mean(loss, axis=0, keepdims=True)

    return loss / tf.stop_gradient(mean)

def norm_clip(x, clip):
    rms = tf.sqrt(tf.reduce_mean(x**2))

    avg_mu = tf.get_variable(auto_name("avg_rms"), initializer=tf.constant(1., dtype=tf.float32))
    update_op = avg_mu.assign(0.997*avg_mu+(1-0.997)*rms)

    with tf.control_dependencies([update_op]):
        return tf.tanh( x / (clip*avg_mu) )

def rnn_action_uniformity_loss(x, sigma, num_points, min_val, max_val):
    #points = tf.lin_space(start=min_val, stop=max_val, num=num_points)
    #points = tf.expand_dims(tf.expand_dims(points, axis=0), axis=0) #Add batch and time dimensions

    np_points = np.linspace(min_val, max_val, num_points)
    np_points = np_points[np.newaxis, np.newaxis, ...]

    np_weights = norm.cdf((np.pi + np_points)/sigma) + norm.cdf((np.pi - np_points)/sigma)
    np_weights = 1/np_weights

    points = tf.convert_to_tensor(np_points.astype(np.float32))
    weights = tf.convert_to_tensor(np_weights.astype(np.float32))
    print("POINTS", points)

    dists = (x-points)**2/(2*sigma**2)
    density = weights*tf.exp(-dists)
    #density = tf.where( dists < 4.5, tf.exp(-dists), tf.zeros(dists.shape) )
    density = tf.reduce_sum(density, axis=0)

    #density = tf.exp(- tf.minimum( , 4.5 ) ) #Broadcast points. 10 for numerical stability

    pdf = density / tf.reduce_sum(density, axis=-1, keepdims=True)

    l2_norm = tf.sqrt(tf.reduce_sum(pdf**2, axis=-1))

    sqrt_d = np.sqrt(num_points)

    #with tf.control_dependencies([tf.print(pdf[0,13:19])]):
    loss = (l2_norm*sqrt_d - 1) / (sqrt_d - 1)

    return loss


def rmsprop_clip_by_value(optimizer, grads_and_vars, limit):

    new_grads_and_vars = []
    for (g, v) in grads_and_vars:
        rms = tf.sqrt(tf.reduce_mean(optimizer.get_slot(v, "v")))

        new_g = tf.clip_by_value(g, -limit*rms, limit*rms)

        new_grads_and_vars.append((new_g, v))

    return new_grads_and_vars


def main(unused_argv):
    """Trains the DNC and periodically reports the loss."""

    graph = tf.get_default_graph()

    action_shape = [FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_actions]
    observation_shape = [FLAGS.batch_size, FLAGS.num_steps, FLAGS.step_size]
    full_scan_shape = [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.img_side, FLAGS.img_side, 1]
    partial_scan_shape = [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.img_side, FLAGS.img_side, 2]

    images = np.load(FLAGS.data_file)
    images[np.logical_not(np.isfinite(images))] = 0
    images = np.stack([norm_img(x) for x in images])

    train_images = images[:int(0.8*len(images))]
    val_images = images[int(0.8*len(images)):]

    train_data_ph, train_iterator = load_data(train_images.shape)
    val_data_ph, val_iterator = load_data(val_images.shape)
    
    if FLAGS.is_self_competition:
        (full_scans, labels) = train_iterator.get_next()
        (val_full_scans, val_labels) = val_iterator.get_next()
    else:
        (full_scans, ) = train_iterator.get_next()
        (val_full_scans, ) = val_iterator.get_next()
    if hasattr(tf, 'ensure_shape'):
        full_scans = tf.ensure_shape(full_scans, full_scan_shape)
        val_full_scans = tf.ensure_shape(val_full_scans, full_scan_shape)
    else:
        full_scans = tf.reshape(full_scans, full_scan_shape)
        val_full_scans = tf.reshape(full_scans, full_scan_shape)

    replay = RingBuffer(
        action_shape=action_shape,
        observation_shape=observation_shape,
        full_scan_shape=full_scan_shape,
        batch_size=FLAGS.batch_size,
        buffer_size=FLAGS.replay_size,
        num_past_losses=train_images.shape[0],
        )

    replay_actions_ph = tf.placeholder(tf.float32, shape=action_shape, name="replay_action")
    replay_observations_ph = tf.placeholder(tf.float32, shape=observation_shape, name="replay_observation")

    replay_full_scans_ph = tf.placeholder(tf.float32, shape=full_scan_shape, name="replay_full_scan")
    partial_scans_ph = tf.placeholder(tf.float32, shape=partial_scan_shape, name="replay_partial_scan")

    is_training_ph = tf.placeholder(tf.bool, name="is_training")

    if FLAGS.over_edge_penalty:
        over_edge_penalty_ph = tf.placeholder(tf.float32, name="over_edge_penalty")

    if FLAGS.is_noise_decay:
        noise_decay_ph = tf.placeholder(tf.float32, shape=(), name="noise_decay")
    else:
        noise_decay_ph = None

    decay_ph = tf.placeholder(tf.float32, name="target_decay")

    if FLAGS.supervision_iters:
        supervision_ph = tf.placeholder(tf.float32, name="supervision")
    else:
        supervision_ph = FLAGS.supervision

    if FLAGS.is_prioritized_replay:
        priority_weights_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size], name="priority_weights")

    batch_size = FLAGS.batch_size

    if FLAGS.is_relative_to_spirals:
        coverage = FLAGS.num_steps*FLAGS.step_size/FLAGS.img_side**2
        spiral = draw_spiral(coverage=coverage, side=FLAGS.img_side)

        ys = [1/i**2 for i in range(9, 2, -1)]
        xs = [np.sum(draw_spiral(coverage=c, side=FLAGS.img_side)) / FLAGS.img_side**2 for c in ys]

        ub_idx = next(i for i, x in xs if x > coverage)
        lb = xs[ub_idx-1]
        ub = xs[ub_idx]

        input_coverage = ( (coverage - lb)*X + (ub - coverage)*Y ) / (lb - ub)

    actor = Agent(
        num_outputs=FLAGS.num_actions, 
        is_new=True,
        noise_decay=noise_decay_ph,
        sampled_full_scans=full_scans, 
        val_full_scans=val_full_scans, 
        name="actor"
        )

    if FLAGS.is_target_actor:
        target_actor = Agent(num_outputs=FLAGS.num_actions, name="target_actor")
    
    critic = Agent(num_outputs=1, is_double_critic=True, name="critic")
    target_critic = Agent(num_outputs=1, is_double_critic=True, name="target_critic")

    new_observations, new_actions = actor.get_new_experience()

    #if hasattr(tf, 'ensure_shape'):
    #    partial_scans_ph = tf.ensure_shape(partial_scans_ph, partial_scan_shape)
    #    replay_full_scans_ph = tf.ensure_shape(replay_full_scans_ph, full_scan_shape)
    #else:
    #    partial_scans_ph = tf.reshape(partial_scans_ph, partial_scan_shape)
    #    replay_full_scans_ph = tf.reshape(replay_full_scans_ph, full_scan_shape)

    #Last actions are unused
    replay_observations = replay_observations_ph[:,:-1,:]
    replay_actions = replay_actions_ph[:,:-1,:]

    #First action must be added for actors (not critics)
    if FLAGS.num_actions == 2:
        start_actions = FLAGS.step_incr*tf.ones([FLAGS.batch_size, 1, FLAGS.num_actions])/np.sqrt(2)
    elif FLAGS.num_actions == 1:
        start_actions = (np.pi/4) * tf.ones([batch_size, 1, FLAGS.num_actions])
        
    started_replay_actions = tf.concat([start_actions, replay_actions[:,:-1,:]], axis=1)

    actions =  actor(replay_observations, started_replay_actions)

    if FLAGS.is_target_actor:
        target_actions = target_actor(replay_observations, started_replay_actions)
    elif FLAGS.supervision != 1:
        target_actions = actions

    #The last action is never used, and the first action is diagonally north-east
    #Shifting because network expect actions from previous steps to be inputted
    #start_actions = tf.ones([FLAGS.batch_size, 1, FLAGS.num_actions])/np.sqrt(2)
    #actions = tf.concat([start_actions, actions[:, :-1, :]], axis=1)
    #target_actions = tf.concat([start_actions, target_actions[:, :-1, :]], axis=1)

    actor_actions = tf.concat([replay_actions, actions], axis=-1)
    qs = critic(replay_observations, actor_actions)
    critic_qs = qs[:,:,:1]

    if FLAGS.is_target_critic and not FLAGS.is_target_actor_feedback:
        actor_qs = qs[:,:,1:]

    if FLAGS.is_target_critic:
        target_actor_actions = tf.concat([replay_actions, target_actions], axis=-1)
        target_qs = target_critic(replay_observations, target_actor_actions)

        target_actor_qs = target_qs[:,:,1:]

        if FLAGS.is_target_actor_feedback:
            actor_qs = target_actor_qs

        target_actor_qs = tf.stop_gradient(target_actor_qs)
    elif FLAGS.supervision != 1:
        target_actor_qs = actor_qs#critic(replay_observations, target_actor_actions)[:,:,1:]
        target_actor_qs = tf.stop_gradient(target_actor_qs)

    #if FLAGS.is_positive_qs and (FLAGS.is_target_critic or FLAGS.supervision != 1):
        #critic_qs = tf.abs(critic_qs)
        #actor_qs = tf.abs(actor_qs)
        #target_actor_qs = tf.abs(target_actor_qs)

        #target_actor_qs = tf.clip_by_value(target_actor_qs, 0, 1)

    #if FLAGS.loss_norm_decay:
    #    processed_replay_losses_ph = avg_bn(replay_losses_ph)
    #else:
    #    processed_replay_losses_ph = replay_losses_ph

    #if FLAGS.is_norm_q:
    #    critic_qs /= tf.reduce_mean(tf.nn.relu(critic_qs), axis=0, keepdims=True)
    #    actor_qs /= tf.reduce_mean(tf.nn.relu(actor_qs), axis=0, keepdims=True)
    #    target_actor_qs /= tf.reduce_mean(tf.nn.relu(target_actor_qs), axis=0, keepdims=True)
        #replay_losses_ph /= tf.reduce_mean(replay_losses_ph)

    if not FLAGS.is_infilled:
        generator = Generator(name="generator", is_training=is_training_ph)
        
        #blurred_partial_scans = tf.concat(
        #    [blur(partial_scans_ph[:,:,:,i:i+1], size=4, std_dev=2.5) for i in range(2)],
        #    axis=-1)

        generation = generator(partial_scans_ph)
    else:
        generation = tf.py_func(fill, [partial_scans_ph], tf.float32)
        if hasattr(tf, 'ensure_shape'):
            generation = tf.ensure_shape(generation, full_scan_shape)
        else:
            generation = tf.reshape(generation, full_scan_shape)

    generator_losses, losses = calc_generator_losses(generation, replay_full_scans_ph)
    unclipped_losses = losses
    #losses = generator_losses

    print("Losses", losses)

    if FLAGS.is_clipped_reward:
        losses = alrc(losses)

    if FLAGS.loss_norm_decay:
        losses = avg_bn(losses)

    #if FLAGS.is_clipped_reward:
    #    losses /= 3
    #    losses = tf.minimum(losses, tf.sqrt(losses))

    if FLAGS.uniform_coverage_loss:
        side = int(np.sqrt(FLAGS.img_side**2 / (FLAGS.num_steps*FLAGS.step_size)))
        blurred_path = blur(partial_scans_ph[:,:,:,1:], size=3*side, std_dev=1.0*side, is_pad=False)

        blurred_path /= FLAGS.num_steps*FLAGS.step_size / FLAGS.img_side**2

        uniformity_loss = tf.reduce_mean( (blurred_path - 1)**2 )

        #blurred_path /= tf.reduce_sum(blurred_path, axis=[1,2,3], keepdims=True)

        #uniformity_loss = tf.sqrt(tf.reduce_sum(blurred_path**2, axis=[1,2,3]))
        #uniformity_loss = (FLAGS.img_side*uniformity_loss - 1) / (FLAGS.img_side - 1)

        losses += noise_decay_ph*FLAGS.uniform_coverage_loss*uniformity_loss

    if FLAGS.is_target_generator and not FLAGS.is_infilled:
        target_generator = Generator(name="target_generator", is_training=is_training_ph)
        target_generation = target_generator(partial_scans_ph)

        if FLAGS.is_minmax_reward:
            errors =  (target_generation - replay_full_scans_ph)**2
            losses = tf.reduce_max( average_filter(errors), reduction_indices=[1,2,3] )
        else:
            target_generator_losses, losses = calc_generator_losses(target_generation, replay_full_scans_ph)
            losses = target_generator_losses #For RL
    else:
        if FLAGS.is_minmax_reward:
            errors =  (generation - replay_full_scans_ph)**2
            losses = tf.reduce_max( average_filter(errors), reduction_indices=[1,2,3] )

    if FLAGS.specificity:
        discriminator = Discriminator(name="discriminator", is_training=is_training_ph)

        true_ps = tf.stack( [replay_full_scans_ph[:,:,:,0], partial_scans_ph[:,:,:,1]], axis=-1 )
        permuted_paths = tf.concat([partial_scans_ph[1:,:,:,1], partial_scans_ph[:1,:,:,1]], axis=0)
        false_ps = tf.stack( [replay_full_scans_ph[:,:,:,0], permuted_paths], axis=-1 )

        discr_inputs = tf.concat([true_ps, false_ps], axis=0)

        discr_outputs = discriminator(discr_inputs)
        discr_outputs, avg_discr_outputs = avg_bn(discr_outputs)

        true_outputs = discr_outputs[:FLAGS.batch_size // FLAGS.avg_replays]
        false_outputs = discr_outputs[FLAGS.batch_size // FLAGS.avg_replays:]

        diffs = true_outputs - false_outputs

        discriminator_losses = tf.reduce_mean( diffs ) + tf.reduce_mean( discr_outputs**2 )

        avg_true_outputs = avg_discr_outputs[:FLAGS.batch_size // FLAGS.avg_replays, 0]
        
        losses += FLAGS.specificity*avg_true_outputs

    if FLAGS.end_edge_penalty:
        if FLAGS.num_actions == 2:
            cartesian_new_actions = replay_actions_ph
        elif FLAGS.num_actions == 1:
            cartesian_new_actions = FLAGS.step_incr*tf.concat([tf.cos(replay_actions_ph), tf.sin(replay_actions_ph)], axis=-1)

        positions = (
            0.5 + #middle of image
            FLAGS.step_incr*FLAGS.step_size/FLAGS.img_side + #First step
            (FLAGS.step_size/FLAGS.img_side)*tf.cumsum(cartesian_new_actions[:,:-1,:], axis=1) # Actions
            )

        is_over_edge = tf.logical_or(tf.greater(positions, 1), tf.less(positions, 0))
        is_over_edge = tf.logical_or(is_over_edge[:,:,0], is_over_edge[:,:,1])
        over_edge_losses = tf.where(
            is_over_edge, 
            over_edge_penalty_ph*tf.ones(is_over_edge.get_shape()), 
            tf.zeros(is_over_edge.get_shape())
            )
        over_edge_losses = tf.reduce_sum(over_edge_losses, axis=-1)

        losses += over_edge_losses


    val_observations, val_actions = actor.get_val_experience()

    #if FLAGS.norm_generator_losses_decay:
    #    mu = tf.get_variable(name="loss_mean", initializer=tf.constant(1., dtype=tf.float32))

    #    mu_op = mu.assign(FLAGS.norm_generator_losses_decay*mu+(1-FLAGS.norm_generator_losses_decay)*tf.reduce_mean(losses))
    #    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mu_op)

    #    losses /= tf.stop_gradient(mu)

    
    #if FLAGS.is_clipped_reward:
    #    losses = alrc(losses)

    #if FLAGS.is_self_competition:
    #    self_competition_losses = tf.where(
    #        past_losses_ph > unclipped_losses, 
    #        tf.ones([FLAGS.batch_size]),
    #        tf.zeros([FLAGS.batch_size])
    #        )

    #    losses += self_competition_losses

    if FLAGS.over_edge_penalty:
        if FLAGS.num_actions == 2:
            cartesian_new_actions = replay_actions_ph
        elif FLAGS.num_actions == 1:
            cartesian_new_actions = FLAGS.step_incr*tf.concat([tf.cos(replay_actions_ph), tf.sin(replay_actions_ph)], axis=-1)

        positions = (
            0.5 + #middle of image
            FLAGS.step_incr*FLAGS.step_size/FLAGS.img_side + #First step
            (FLAGS.step_size/FLAGS.img_side)*tf.cumsum(cartesian_new_actions[:,:-1,:], axis=1) # Actions
            )

        is_over_edge = tf.logical_or(tf.greater(positions, 1), tf.less(positions, 0))
        is_over_edge = tf.logical_or(is_over_edge[:,:,0], is_over_edge[:,:,1])
        over_edge_losses = tf.where(
            is_over_edge, 
            over_edge_penalty_ph*tf.ones(is_over_edge.get_shape()), 
            tf.zeros(is_over_edge.get_shape())
            )
        #over_edge_losses = tf.cumsum(over_edge_losses, axis=1)

    if FLAGS.supervision > 0 or FLAGS.is_advantage_actor_critic:

        supervised_losses = []
        for i in reversed(range(FLAGS.num_steps-1)):
            if i == FLAGS.num_steps-1 - 1: #Extra -1 as idxs start from 0
                step_loss = tf.expand_dims(losses, axis=-1)
            else:
                step_loss = FLAGS.gamma*step_loss

            #if FLAGS.over_edge_penalty:
            #    step_loss += over_edge_losses[:,i:i+1]

            supervised_losses.append(step_loss)
        supervised_losses = tf.concat(supervised_losses, axis=-1)

    if FLAGS.supervision < 1:
        bellman_losses = tf.concat(
            [FLAGS.gamma*target_actor_qs[:,1:,0], tf.expand_dims(losses, axis=-1)], 
            axis=-1
            )

        bellman_losses = supervision_ph * supervised_losses + (1 - supervision_ph) * bellman_losses
    else:
        bellman_losses = supervised_losses

    if FLAGS.over_edge_penalty:
        bellman_losses += over_edge_losses

    if FLAGS.loss_gamma != 1:
        loss_gamma_decays = FLAGS.loss_gamma**tf.expand_dims(tf.lin_space(FLAGS.num_steps-2.0, 0.0, FLAGS.num_steps-1), axis=0)

    if FLAGS.is_prioritized_replay:
        unweighted_critic_losses = tf.reduce_mean( ( critic_qs[:,:,0] - bellman_losses )**2, axis=-1 )
        critic_losses = tf.reduce_mean( priority_weights_ph*unweighted_critic_losses )
    else:
        critic_qs_slice = critic_qs[:,:,0]
        critic_losses = ( critic_qs_slice - bellman_losses )**2

        if FLAGS.is_clipped_critic:
            critic_losses = alrc( critic_losses )
        if FLAGS.rnn_norm_decay:
            critic_losses = rnn_loss_norm(critic_losses)

        if FLAGS.loss_gamma != 1:
            critic_losses *= loss_gamma_decays**2

        if FLAGS.is_biased_prioritized_replay:
            unweighted_critic_losses = critic_losses[:,-1]

        critic_losses = tf.reduce_mean( critic_losses )

        if FLAGS.spike_loss:
            diffs = critic_qs_slice[:,1:] - critic_qs_slice[:,:-1]
            diffs_start = diffs[:,1:]
            diffs_end = diffs[:,:-1]
            spike_losses = tf.where(diffs_start*diffs_end < 0, 
                                    tf.minimum(tf.abs(diffs_start), tf.abs(diffs_end)),
                                    tf.zeros(diffs_start.get_shape())
                                    )
            critic_losses += FLAGS.spike_loss*tf.reduce_mean(spike_losses)

    if FLAGS.is_advantage_actor_critic:
        actor_losses = tf.reduce_mean( supervised_losses - actor_qs[:,:,0] )
    else:
        if FLAGS.is_clipped_critic:
            actor_losses = alrc(actor_qs[:,:,0])
        else:
            actor_losses = actor_qs[:,:,0]

        if FLAGS.is_direct_advantage:
            bases0 = FLAGS.gamma*tf.reduce_mean(critic_qs[:,:1,0], axis=0, keepdims=True)
            bases0 = tf.tile(bases0, [FLAGS.batch_size, 1])

            bases = tf.concat([bases0, critic_qs[:,:-1,0]], axis=1)

            actor_losses = actor_losses - bases


        #if FLAGS.rnn_norm_decay:
        #    actor_losses = rnn_loss_norm(actor_losses, absolute=True)

        
        if FLAGS.loss_gamma != 1:
            actor_losses *= loss_gamma_decays

        actor_losses = tf.reduce_mean( actor_losses )
    
        if FLAGS.exploration_loss:
            exploration_losses = rnn_action_uniformity_loss(actions, np.pi/32, 32, -np.pi, np.pi)

            actor_losses += noise_decay_ph*FLAGS.exploration_loss*tf.reduce_mean(exploration_losses)

    if FLAGS.loss_norm_clip:
       actor_losses = norm_clip(actor_losses, FLAGS.loss_norm_clip)
       critic_losses = norm_clip(critic_losses, FLAGS.loss_norm_clip)

    #critic_losses /= FLAGS.num_steps
    #actor_losses /= FLAGS.num_steps

    #Outputs to provide feedback for the developer
    info = {
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "generator_losses": tf.reduce_mean(unclipped_losses)
        }

    if FLAGS.specificity:
        info.update({"discriminator_output": avg_true_outputs[0]})

    if FLAGS.is_prioritized_replay or FLAGS.is_biased_prioritized_replay:
        info.update( {"priority_weights": unweighted_critic_losses} )

    if FLAGS.is_self_competition:
        info.update( {"unclipped_losses": unclipped_losses} )

    outputs = {
        "generation": generation[0,:,:,0],
        "truth": replay_full_scans_ph[0,:,:,0],
        "input": partial_scans_ph[0,:,:,0]
        }

    history_op = {
        "actions": new_actions,
        "observations": new_observations, 
        "labels": labels,
        "full_scans": full_scans
        }

    if FLAGS.is_self_competition:
        history_op.update( {"labels": labels} )

    ##Modify actor gradients
    #[actor_grads] = tf.gradients(actor_losses, replay_actions_ph)
    #actor_losses = overwrite_grads(actions, actor_grads)

    start_iter = FLAGS.start_iter
    train_iters = FLAGS.train_iters

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #Only use required GPU memory
    #config.gpu_options.force_gpu_compatible = True

    model_dir = FLAGS.model_dir

    log_filepath = model_dir + "log.txt"
    save_period = 1; save_period *= 3600
    log_file = open(log_filepath, "a")
    with tf.Session(config=config) as sess:

        if FLAGS.is_target_actor:
            if FLAGS.update_frequency <= 1:
                update_target_actor_op = target_update_ops(target_actor, actor, decay=decay_ph, l2_norm=FLAGS.L2_norm)
            else:
                update_target_actor_op = []
            initial_update_target_actor_op = target_update_ops(target_actor, actor, decay=0, l2_norm=FLAGS.L2_norm)

        else:
            update_target_actor_op = weight_decay_ops(actor, FLAGS.L2_norm)
            initial_update_target_actor_op = weight_decay_ops(actor, FLAGS.L2_norm)

        if FLAGS.is_target_critic:
            if FLAGS.update_frequency <= 1:
                update_target_critic_op = target_update_ops(target_critic, critic, decay=decay_ph, l2_norm=FLAGS.L2_norm)
            else:
                update_target_critic_op = []
            initial_update_target_critic_op = target_update_ops(target_critic, critic, decay=0, l2_norm=FLAGS.L2_norm)
        else:
            update_target_critic_op = weight_decay_ops(critic, FLAGS.L2_norm)
            initial_update_target_critic_op = weight_decay_ops(critic, FLAGS.L2_norm)

        if FLAGS.is_target_generator and not FLAGS.is_infilled:
            if FLAGS.update_frequency <= 1:
                update_target_generator_op = target_update_ops(target_generator, generator, decay=decay_ph, l2_norm=FLAGS.L2_norm)
            else:
                update_target_generator_op = []
            initial_update_target_generator_op = target_update_ops(target_generator, generator, decay=0, l2_norm=FLAGS.L2_norm)
        elif not FLAGS.is_infilled:
            update_target_generator_op = weight_decay_ops(generator, FLAGS.L2_norm)
            initial_update_target_generator_op = weight_decay_ops(generator, FLAGS.L2_norm)
        else:
            update_target_generator_op = []
            initial_update_target_generator_op = []

        initial_update_target_network_ops = (
            initial_update_target_actor_op +
            initial_update_target_critic_op + 
            initial_update_target_generator_op 
            )

        actor_lr = FLAGS.actor_lr
        critic_lr = FLAGS.critic_lr
        if FLAGS.is_cyclic_generator_learning_rate and not FLAGS.is_infilled:
            generator_lr = tf.placeholder(tf.float32, name="generator_lr")
        else:
            generator_lr = FLAGS.generator_lr

        #critic_rep = (critic_qs[:,:,0] - bellman_losses)**2
        #ps = [critic_qs[0,:,0], target_actor_qs[0,:,0], bellman_losses[0], critic_rep[0]]

        #ps = [critic.trainable_variables[0], target_critic.trainable_variables[0]]
        ps = []
        #p = bellman_losses[0]
        #p = generation[0,:,:,0]

        train_op_dependencies = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if not FLAGS.update_frequency:

            update_target_network_ops = (
                update_target_actor_op + 
                update_target_critic_op + 
                update_target_generator_op 
                )

            if FLAGS.grad_clip_value:
                step = tf.Variable(0., trainable=False, name='step')
                step_op = step.assign_add(1)
                update_target_network_ops += [step_op]

            train_op_dependencies += update_target_network_ops

        train_ops = []
        with tf.control_dependencies(train_op_dependencies):
            actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=actor_lr)
            critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=critic_lr)

            actor_grads = actor_optimizer.compute_gradients(
                loss=actor_losses, var_list=actor.trainable_variables)
            critic_grads = critic_optimizer.compute_gradients(
                loss=critic_losses, var_list=critic.trainable_variables)

            #if FLAGS.grad_clip_value:
            #    actor_optimizer._create_slots(actor.trainable_variables)
            #    critic_optimizer._create_slots(critic.trainable_variables)

            #    actor_optimizer._create_slots = lambda var_list: None
            #    critic_optimizer._create_slots = lambda var_list: None

            #    limit = FLAGS.grad_clip_value*tf.maximum(10_000*(5_000 - step), 1.)

            #    actor_grads = rmsprop_clip_by_value(actor_optimizer, actor_grads, limit)
            #    critic_grads = rmsprop_clip_by_value(critic_optimizer, critic_grads, limit)


            if FLAGS.grad_clip_value:
                #ps = [tf.reduce_max(tf.abs(g)) for (g, v) in actor_grads]
                #with tf.control_dependencies([tf.Print(p, [p]) for p in ps]):
                actor_grads = [(tf.clip_by_value(g, -FLAGS.grad_clip_value, FLAGS.grad_clip_value), v) 
                                for (g, v) in actor_grads]
                #ps = [tf.reduce_mean(tf.abs(g)) for (g, v) in critic_grads]
                #with tf.control_dependencies([tf.Print(p, [p]) for p in ps]):
                critic_grads = [(tf.clip_by_value(g, -FLAGS.grad_clip_value, FLAGS.grad_clip_value), v) 
                                for (g, v) in critic_grads]


            if FLAGS.grad_clip_norm:
                #ps = [tf.reduce_max(tf.abs(g)) for (g, v) in actor_grads]
                #with tf.control_dependencies([tf.Print(p, [p]) for p in ps]):
                actor_grads = [(tf.clip_by_norm(g, FLAGS.grad_clip_norm), v) 
                                for (g, v) in actor_grads]
                #ps = [tf.reduce_mean(tf.abs(g)) for (g, v) in critic_grads]
                #with tf.control_dependencies([tf.Print(p, [p]) for p in ps]):
                critic_grads = [(tf.clip_by_norm(g, FLAGS.grad_clip_norm), v) 
                                for (g, v) in critic_grads]

            actor_train_op = actor_optimizer.apply_gradients(actor_grads)
            critic_train_op = critic_optimizer.apply_gradients(critic_grads)

            train_ops += [actor_train_op, critic_train_op]

            if not FLAGS.is_infilled:
                generator_train_op = tf.train.AdamOptimizer(learning_rate=generator_lr).minimize(
                    loss=generator_losses, var_list=generator.trainable_variables)
                
                train_ops.append(generator_train_op)

            if FLAGS.specificity:
                discriminator_train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_lr).minimize(
                    loss=discriminator_losses, var_list=discriminator.trainable_variables)
                
                train_ops.append(discriminator_train_op)
        
        feed_dict = {is_training_ph: np.bool(True),
                     decay_ph: np.float32(0.99)}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        saver = tf.train.Saver(max_to_keep=1)
        noteable_saver = tf.train.Saver(max_to_keep=2)


        saver.restore(
            sess, 
            tf.train.latest_checkpoint(model_dir+"noteable_ckpt/")
            )

        sess.run(train_iterator.initializer, feed_dict={train_data_ph: train_images})
        sess.run(val_iterator.initializer, feed_dict={val_data_ph: val_images})

        #Add first experiences to the replay
        if FLAGS.is_noise_decay:
            feed_dict.update({noise_decay_ph: np.float32(1)})

        for _ in range(FLAGS.avg_replays):
            history = sess.run(
                history_op,
                feed_dict=feed_dict)

            replay.add(**history) 

        time0 = time.time()
        for iter in range(start_iter, train_iters):

            #Sample experiences from the replay
            if FLAGS.is_prioritized_replay:
                sampled_actions, sampled_observations, replay_sampled_full_scans, sample_idxs, sampled_priority_weights = replay.get()
            #elif FLAGS.is_biased_prioritized_replay:
            #    sampled_actions, sampled_observations, replay_sampled_full_scans, sample_idxs = replay.get()
            elif FLAGS.is_self_competition:
                sampled_actions, sampled_observations, sample_idxs, sampled_past_losses, replay_sampled_full_scans = replay.get()
            else:
                sampled_actions, sampled_observations, replay_sampled_full_scans = replay.get()

            if FLAGS.is_ranked_loss:
                idxs = np.argsort(sampled_past_losses)
                sampled_past_losses[idxs] = np.linspace(0, 1, FLAGS.batch_size)

            target_decay = FLAGS.target_decay**( 1 + max(100_000 - iter, 0)/10_000 )

            augmented_partial_scans, augmented_full_scans = construct_scans(
                sampled_actions, 
                sampled_observations,
                replay_sampled_full_scans
                )

            feed_dict = {
                replay_actions_ph: sampled_actions,
                replay_observations_ph: sampled_observations,
                is_training_ph: np.bool(True),
                decay_ph: np.float32(target_decay),
                partial_scans_ph: augmented_partial_scans,
                replay_full_scans_ph: augmented_full_scans
                }

            if FLAGS.over_edge_penalty:
                penalty = np.float32(FLAGS.over_edge_penalty)
                #penalty = np.float32( FLAGS.over_edge_penalty*max((10_000 - iter)/10_000, 0) )
                feed_dict.update( {over_edge_penalty_ph: penalty} )

            if FLAGS.is_noise_decay:
                noise_decay = np.float32( np.maximum( ((train_iters//2 - iter)/(train_iters//2))**2, 0) )
                feed_dict.update( {noise_decay_ph: noise_decay} )

            if FLAGS.is_prioritized_replay:
                feed_dict.update({priority_weights_ph: sampled_priority_weights})

            if FLAGS.supervision_iters:
                supervision = FLAGS.supervision_start + min(iter, FLAGS.supervision_iters)*(FLAGS.supervision_end-FLAGS.supervision_start) / FLAGS.supervision_iters
                feed_dict.update( {supervision_ph: supervision } )

            if FLAGS.is_cyclic_generator_learning_rate and not FLAGS.is_infilled:
                if FLAGS.is_decaying_generator_learning_rate:
                    envelope = FLAGS.generator_lr * 0.75**(iter/(train_iters//5))
                else:
                    envelope = FLAGS.generator_lr

                cycle_half = train_iters//(10 - 1)
                cycle_full = 2*cycle_half

                cyclic_sawtooth = 1 - (min(iter%cycle_full, cycle_half) - min(iter%cycle_full - cycle_half, 0))/cycle_half

                cyclic_lr = envelope*(0.2 + 0.8*cyclic_sawtooth)

                feed_dict.update( {generator_lr: np.float32(cyclic_lr)} )

            #Train
            if True:
                history, step_info, step_outputs = sess.run([history_op, info, outputs], feed_dict=feed_dict)
          
                for k in step_outputs:
                    save_loc = FLAGS.model_dir + k + str(iter)+".tif"
                    Image.fromarray( (0.5*step_outputs[k]+0.5).astype(np.float32) ).save( save_loc )
            else:
                _, history, step_info = sess.run([train_ops, history_op, info], feed_dict=feed_dict)

            if iter < 100_000 or not np.random.randint(0, int(max(iter/100_000, 1)*FLAGS.replay_add_frequency)):
                replay.add(**history)

            if iter >= 100:
                quit()

            if FLAGS.update_frequency:
                period = max(int(min(iter/100_000, 1)*(FLAGS.update_frequency-1)), 1)
                if not iter % np.random.randint(1, 1 + period):
                    sess.run(initial_update_target_network_ops, feed_dict=feed_dict)

            if FLAGS.is_prioritized_replay or FLAGS.is_biased_prioritized_replay:
                replay.update_priorities(sample_idxs, step_info["priority_weights"])


            output = f"Iter: {iter}"
            for k in step_info:
                if k not in ["priority_weights", "unclipped_losses"]:
                    output += f", {k}: {step_info[k]}"
            if not iter % FLAGS.report_freq:
                print(output)

            #if "nan" in output:
            #    saver.restore(
            #        sess, 
            #        tf.train.latest_checkpoint(model_dir+"model/")
            #        )

            if iter in [train_iters//2-1, train_iters-1]:
                noteable_saver.save(sess, save_path=model_dir+"noteable_ckpt/model", global_step=iter)
                time0 = time.time()
                start_iter = iter
            elif time.time() >= time0 + save_period:
                saver.save(sess, save_path=model_dir+"model/model", global_step=iter)
                time0 = time.time()

        val_losses_list = []
        for iter in range(0, FLAGS.val_examples//FLAGS.batch_size):
            #Add experiences to the replay
            feed_dict = {is_training_ph: np.bool(True)}
            sampled_actions, sampled_observations, sampled_full_scans = sess.run(
                [val_actions, val_observations, val_full_scans],
                feed_dict=feed_dict
                )

            partial_scans = construct_partial_scans(sampled_actions, sampled_observations)

            feed_dict = {
                replay_actions_ph: sampled_actions,
                replay_observations_ph: sampled_observations,
                replay_full_scans_ph: sampled_full_scans,
                partial_scans_ph: partial_scans,
                is_training_ph: np.bool(False)
                }

            val_losses = sess.run( unclipped_losses, feed_dict=feed_dict )
            val_losses_list.append( val_losses )
        val_losses = np.concatenate(tuple(val_losses_list), axis=0)
        np.save(model_dir + "val_losses.npy", val_losses)

if __name__ == "__main__":
  tf.app.run()
