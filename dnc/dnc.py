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
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from dnc import access

from utility import auto_name

FLAGS = tf.flags.FLAGS


DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state', 'output', 
                                               'position', 'noise'))

class Components(snt.AbstractModule):

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               name='components'
               ):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(Components, self).__init__(name=name)


    with self._enter_variable_scope():
      self.controller = snt.DeepRNN([snt.LSTM(**controller_config) for _ in range(FLAGS.lstm_depth)], skip_connections=True)
      #self.controller = snt.LSTM(**controller_config)
      self.access = access.MemoryAccess(**access_config)

      self.output_linear = snt.Linear(output_size=output_size, use_bias=False)

      if FLAGS.is_input_embedder:
          self.input_embedder = snt.Sequential([
              snt.Linear(output_size=64, use_bias=True),
              tf.nn.tanh
              ])

      if FLAGS.is_variable_initial_states:
          def c_fn(x):
              shape = x.get_shape().as_list()
              y = tf.Variable(initial_value=tf.zeros(shape=[1]+shape[1:]), dtype=tf.float32, trainable=True)
              return y
          self.initial_controller_state = tf.contrib.framework.nest.map_structure(
              c_fn, self.controller.initial_state(1, tf.float32))

  def _build(self):
      return

class DNC(snt.RNNCore):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               components,
               output_size,
               clip_value=None,
               is_new=False,
               noise_decay=None,
               is_double_critic=False,
               sampled_full_scans=None,
               is_noise=False,
               is_actor=True,
               name='dnc'
               ):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)

    with self._enter_variable_scope():
      self._controller =  components.controller
      self._access = components.access

      self._batch_flatten = snt.BatchFlatten()

      self._output_linear = components.output_linear

      if FLAGS.is_input_embedder:
          self._input_embedder = components.input_embedder

      if FLAGS.is_variable_initial_states:
          self._initial_controller_state = components.initial_controller_state

    self._access_output_size = np.prod(self._access.output_size.as_list())
    self._output_size_features = output_size
    self._clip_value = clip_value or 0

    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
        controller_state=self._controller.state_size,
        output=tf.TensorShape([output_size]),
        position=tf.TensorShape([output_size]),
        noise=tf.TensorShape([1]))

    if is_new:
        self._output_size = tf.TensorShape([FLAGS.step_size+FLAGS.num_actions])
        self._full_scans = sampled_full_scans
    elif is_double_critic:
        self._output_size = tf.TensorShape([2*output_size])
    else:
        self._output_size = tf.TensorShape([output_size])

    self._is_new = is_new

    self._is_noise = is_noise

    self._noise_decay = noise_decay
    
    self._is_actor = is_actor

    self.first = True


  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def _build(self, inputs, prev_state):
    """Connects the DNC core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state
    prev_output = prev_state.output
    prev_position = prev_state.position
    prev_noise = prev_state.noise

    if FLAGS.is_variable_initial_states and self.first:
        self.first = False
        def c_fn(x):
            shape = x.get_shape().as_list()
            y = tf.tile(x, prev_output.get_shape().as_list()[0:1] + [1 for _ in shape[1:]])
            return y
        prev_controller_state = tf.contrib.framework.nest.map_structure(
            c_fn, self._initial_controller_state)

    if FLAGS.is_prev_position_input:
        prev_position = tf.stop_gradient(prev_position) # Gotta catch 'em all

    if FLAGS.model == "DNC":
        neural_machine = self._neural_machine
    elif FLAGS.model == "LSTM":
        neural_machine = self._neural_machine2

    extra_tiled_actions = []
    if self._is_new:
        actions = prev_output
        observations = self.obs(actions, prev_position)
    else:
        actions = inputs[:,FLAGS.step_size:FLAGS.step_size+FLAGS.num_actions]
        observations = inputs[:,:FLAGS.step_size]

        new_actions = inputs[:,FLAGS.step_size+FLAGS.num_actions:FLAGS.step_size+2*FLAGS.num_actions]

        if FLAGS.is_prev_position_input:
            scaled_prev_position = (FLAGS.img_side/FLAGS.step_size)*prev_position
            if not self._is_actor and not FLAGS.is_advantage_actor_critic: #If critic
                new_actions = tf.concat([new_actions, scaled_prev_position], axis=-1)
            else: #If not critic
                new_actions = scaled_prev_position

        if FLAGS.is_immediate_critic_loss and not self._is_actor:
            hidden_output, _, _, _ = neural_machine(
                observations=observations,
                actions=new_actions,
                prev_controller_state=tf.contrib.framework.nest.map_structure(tf.stop_gradient, prev_controller_state),
                prev_access_output=tf.contrib.framework.nest.map_structure(tf.stop_gradient, prev_access_output),
                prev_access_state=tf.contrib.framework.nest.map_structure(tf.stop_gradient, prev_access_state)
                )
        else:
            hidden_output, _, _, _ = neural_machine(
                observations=observations,
                actions=new_actions,
                prev_controller_state=prev_controller_state,
                prev_access_output=prev_access_output,
                prev_access_state=prev_access_state
                )

    if FLAGS.num_actions == 2:
        position = prev_position + FLAGS.step_size*actions/FLAGS.img_side
    elif FLAGS.num_actions == 1:
        cartesian_actions = FLAGS.step_incr*tf.concat([tf.cos(actions), tf.sin(actions)], axis=-1)
        position = prev_position + FLAGS.step_size*cartesian_actions/FLAGS.img_side

    if FLAGS.is_prev_position_input:
        scaled_prev_position = (FLAGS.img_side/FLAGS.step_size)*prev_position
        if not self._is_actor and not FLAGS.is_advantage_actor_critic: #If critic
            actions = tf.concat([actions, scaled_prev_position], axis=-1)
        else: #If not critic
            actions = scaled_prev_position

    output, access_output, access_state, controller_state = neural_machine(
        observations=observations, 
        actions=actions,
        prev_controller_state=prev_controller_state,
        prev_access_output=prev_access_output,
        prev_access_state=prev_access_state
        )

    if not self._is_actor: #If critic
        hidden_output = tf.concat([output, hidden_output], axis=-1)
    else: #If not critic
        if FLAGS.num_actions == 2:
            output /= tf.sqrt(1.e-8 + tf.reduce_sum(output**2, axis=-1, keepdims=True))

            if FLAGS.step_incr != 1:
                output *= FLAGS.step_incr
        elif FLAGS.num_actions == 1:
            output = np.pi*tf.tanh(output)

    if self._is_new and self._is_noise and self._is_actor and FLAGS.is_ornstein_uhlenbeck:
        noise = self.ornstein_uhlenbeck(prev_noise, FLAGS.ou_theta, FLAGS.ou_sigma)
        
        if not self._noise_decay is None:
            if FLAGS.num_actions == 2:
                applied_noise = tf.scalar_mul(self._noise_decay, noise)

                #Apply noise to action as a rotation
                c = tf.cos(applied_noise)
                s = tf.sin(applied_noise)

                x = c*output[:, :1] - s*output[:, 1:]
                y = s*output[:, :1] + c*output[:, 1:]
        
                output = tf.concat([x, y], axis=-1)
            elif FLAGS.num_actions == 1:
                output += tf.clip_by_value(noise, -np.pi, np.pi)

                output = tf.where(output > np.pi,
                                  output - 2*np.pi,
                                  output)
                output = tf.where(output < -np.pi,
                                  output + 2*np.pi,
                                  output)
    else:
        noise = prev_noise #Unchanged unless exploring

    if self._is_new:
        hidden_output = tf.concat([observations, output], axis=-1)
    elif self._is_actor:#If not critic
        if FLAGS.num_actions == 2:
            hidden_output /= tf.sqrt(1.e-8 + tf.reduce_sum(hidden_output**2, axis=-1, keepdims=True))
            if FLAGS.step_incr != 1:
                hidden_output *= FLAGS.step_incr
        elif FLAGS.num_actions == 1:
            hidden_output = np.pi*tf.tanh(hidden_output)

    return hidden_output, DNCState(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state,
        output=output,
        position=position,
        noise=noise)

  def _neural_machine(
        self,
        observations, 
        actions,
        prev_controller_state,
        prev_access_output,
        prev_access_state
        ):
    
    #step_size = observations.get_shape().as_list()[-1]
    #actions_size = actions.get_shape().as_list()[-1]
    tiled_actions = tf.tile(actions, [1, FLAGS.step_size//FLAGS.num_actions])

    inputs = tf.concat([observations, tiled_actions], axis=-1)

    if self._is_actor: #If not critic
        inputs = tf.stop_gradient(inputs)

    if FLAGS.is_input_embedder:
        inputs = self._input_embedder(inputs)

    controller_input = tf.concat(
        [self._batch_flatten(inputs), self._batch_flatten(prev_access_output)], 1)
    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = tf.contrib.framework.nest.map_structure(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output, prev_access_state)

    output = tf.concat([controller_output, self._batch_flatten(access_output)], 1)
    output = self._output_linear(output)

    if not self._is_actor:
        output = tf.abs(output)
    
    return output, access_output, access_state, controller_state

  def _neural_machine2(
        self,
        observations, 
        actions,
        prev_controller_state,
        prev_access_output,
        prev_access_state
        ):
    """Simple LSTM"""

    #step_size = observations.get_shape().as_list()[-1]
    #actions_size = actions.get_shape().as_list()[-1]
    tiled_actions = tf.tile(actions, [1, FLAGS.step_size//FLAGS.num_actions])

    inputs = tf.concat([observations, tiled_actions], axis=-1)

    if self._is_actor: #If not critic
        inputs = tf.stop_gradient(inputs)

    if FLAGS.is_input_embedder:
        inputs = self._input_embedder(inputs)

    controller_output, controller_state = self._controller(
        self._batch_flatten(inputs), prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = tf.contrib.framework.nest.map_structure(self._clip_if_enabled, controller_state)

    output = self._output_linear(controller_output)

    if not self._is_actor:
        output = tf.abs(output)

    return output, prev_access_output, prev_access_state, controller_state


  def initial_state(self, batch_size, dtype=tf.float32):
    
    controller_state = self._controller.initial_state(batch_size, dtype)

    if FLAGS.num_actions == 2:
        output = FLAGS.step_incr*tf.ones([batch_size, self._output_size_features])/np.sqrt(2)
    else:
        output = (np.pi/4) * tf.ones([batch_size, self._output_size_features])

    return DNCState(
        controller_state=controller_state,
        access_state=self._access.initial_state(batch_size, dtype),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype),
        output=output,
        position=tf.ones([batch_size, 2])/2,
        noise=tf.zeros([batch_size, 1])
        )

  def obs(self, actions, starts):
      x = tf.py_func(self.make_observations, [actions, starts, self._full_scans], tf.float32)
      if hasattr(tf, 'ensure_shape'):
          x = tf.ensure_shape(x, [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.step_size])
      else:
          x = tf.reshape(x, [FLAGS.batch_size // FLAGS.avg_replays, FLAGS.step_size])
      return x

  def ornstein_uhlenbeck(self, input, theta, sigma):
    """Ornstein-Uhlembeck perturbation. Using Gaussian Wiener process."""
    noise_perturb = -theta*input + sigma*tf.random_normal(shape=input.get_shape())
    return input + noise_perturb

  @staticmethod
  def make_observations(actions, starts, full_scans):

    if FLAGS.num_actions == 1:
        actions = FLAGS.step_incr*np.concatenate((np.cos(actions), np.sin(actions)), axis=-1)

    starts *= FLAGS.img_side
    x = np.minimum(np.maximum(np.stack([starts + i*actions for i in range(FLAGS.step_size)]), 0), FLAGS.img_side-1)

    indices = []
    for j in range(FLAGS.batch_size // FLAGS.avg_replays):
        for i in range(FLAGS.step_size):
            indices.append( [j, int(x[i][j][0]), int(x[i][j][1]), 0] )
    indices = tuple([np.array(indices)[:,i] for i in range(4)])

    observations = full_scans[indices].reshape([-1, FLAGS.step_size])

    return observations

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
