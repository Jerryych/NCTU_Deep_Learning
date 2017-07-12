import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple
import csv


# Hyper Parameters:
GAMMA = 0.99				# decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0			# 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1			# 0.001 # final value of epsilon
EXPLORE_STPES = 500000			# frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 300000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000		# Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]

class ObservationProcessor():
	"""
	Processes a raw Atari image. Resizes it and converts it to grayscale.
	"""
	def __init__(self):
		with tf.variable_scope("state_processor"):
			self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)			# input image
			self.output = tf.image.rgb_to_grayscale(self.input_state)				# rgb to grayscale
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)		# crop image
			self.output = tf.image.resize_images(							# resize image
				self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)							# remove rgb dimension

	def process(self, sess, state):
		"""
		Args:
			sess: A Tensorflow session object
			state: A [210, 160, 3] Atari RGB State

		Returns:
			A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run(self.output, { self.input_state: state })

class DQN():
	# Define the following things about Deep Q Network here:
	#   1. Network Structure (Check lab spec for details)
	#	   * tf.contrib.layers.conv2d()
	#	   * tf.contrib.layers.flatten()
	#	   * tf.contrib.layers.fully_connected()
	#	   * You may need to use tf.variable_scope in order to set different variable names for 2 Q-networks
	#   2. Target value & loss
	#   3. Network optimizer: tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
	#   4. Training operation for tensorflow

	def __init__(self, scope):
		self.scope = scope
		with tf.variable_scope(scope):
			self._build_network()

	# You may need 3 placeholders for input: 4 input images, target Q value, action index
	def _build_network(self):
		# Placeholders for our input
		# Our input are 4 grayscale frames of shape 84, 84 each
		self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
		# The TD target value
		self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
		# Integer id of which action was selected
		self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

		X = tf.to_float(self.X_pl) / 255.0
	
		# Network Structure
		conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, padding='SAME', activation_fn=tf.nn.relu)
		conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, padding='SAME', activation_fn=tf.nn.relu)
		conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
		flatten = tf.contrib.layers.flatten(conv3)
		fc1 = tf.contrib.layers.fully_connected(flatten, 512)
		self.Q = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=None)

		# actions
		idx = tf.range(BATCH_SIZE) * tf.shape(self.Q)[1] + self.actions_pl
		self.actionsQ = tf.gather(tf.reshape(self.Q, [-1]), idx)

		# Target value & loss
		self.sdiff = tf.squared_difference(self.y_pl, self.actionsQ)
		self.loss = tf.reduce_mean(self.sdiff)

		# Optimizer
		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op = self.optimizer.minimize(self.loss)
	
	def get_Q(self, sess, x):
		return sess.run(self.Q, feed_dict={self.X_pl: x})

	def update(self, sess, x, y, a):
		sess.run(self.train_op, feed_dict={self.X_pl: x, self.y_pl: y, self.actions_pl: a})

def update_target_network(sess, behavior_Q, target_Q):
	# copy weights from behavior Q-network to target Q-network
	# Hint:
	#   * tf.trainable_variables()				https://www.tensorflow.org/api_docs/python/tf/trainable_variables
	#   * variable.name.startswith(scope_name)		https://docs.python.org/3/library/stdtypes.html#str.startswith
	#   * assign						https://www.tensorflow.org/api_docs/python/tf/assign
	behavior_params = [t for t in tf.trainable_variables() if t.name.startswith(behavior_Q.scope)]
	behavior_params = sorted(behavior_params, key=lambda v: v.name)
	target_params = [t for t in tf.trainable_variables() if t.name.startswith(target_Q.scope)]
	target_params = sorted(target_params, key=lambda v: v.name)

	update = []
	for b_v, t_v in zip(behavior_params, target_params):
		op = tf.assign(t_v, b_v)
		update.append(op)

	sess.run(update)


def epsilon_policy(sess, behavior_Q, observations, epsilon):
	if random.random() <= epsilon:
		return random.randint(0, 3), 'r'
	else:
		# qs = sess.run(behavior_Q.Q, feed_dict={behavior_Q.X_pl: np.expand_dims(observations, axis=0)})[0]
		qs = behavior_Q.get_Q(sess, np.expand_dims(observations, axis=0))[0]
		return np.argmax(qs), 'q'

def main(_):
	tf.reset_default_graph()
	# make game eviornment
	env = gym.envs.make("Breakout-v0")

	# Define Transition tuple
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

	# The replay memory
	replay_memory = []

	# create a observation processor
	ob_proc = ObservationProcessor()

	# Behavior Network & Target Network
	behavior_Q = DQN('behavior')
	target_Q = DQN('target')

	# tensorflow session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	# Populate the replay buffer
	observation = env.reset()					# retrive first env image
	observation = ob_proc.process(sess, observation)		# process the image
	state = np.stack([observation] * 4, axis=2)			# stack the image 4 times
	action = None
	while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
		'''
		*** This part is just pseudo code ***

		action = None
		if random.random() <= epsilon
			action = random_action
		else
			action = DQN_action
		'''
		action, _ = epsilon_policy(sess, behavior_Q, state, INITIAL_EPSILON)
		next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
		next_observation = ob_proc.process(sess, next_observation)
		next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
		replay_memory.append(Transition(state, action, reward, next_state, done))
		sys.stdout.write('\rPopulating replay memory...%d' % len(replay_memory))
		sys.stdout.flush()

		# Current game episode is over
		if done:
			observation = env.reset()
			observation = ob_proc.process(sess, observation)
			state = np.stack([observation] * 4, axis=2)

		# Not over yet
		else:
			state = next_state
	print

	# record videos
	record_video_every = 50
	env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count % record_video_every == 0, resume=True)

	# total steps
	total_t = 0

	epsilon_schedule = np.linspace(INITIAL_EPSILON, FINAL_EPSILON, EXPLORE_STPES)
	samples = None
	maxreward = -32767
	epr = []

	for episode in range(TRAINING_EPISODES):

		# Reset the environment
		observation = env.reset()
		observation = ob_proc.process(sess, observation)
		state = np.stack([observation] * 4, axis=2)
		episode_reward = 0							  # store the episode reward
		'''
		How to update episode reward:
		next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
		episode_reward += reward
		'''

		for t in itertools.count():

			# choose a action
			action, _ = epsilon_policy(sess, behavior_Q, state, epsilon_schedule[min(total_t, EXPLORE_STPES - 1)])
			# execute the action
			next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
			episode_reward += reward
			# if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
			if len(replay_memory) == REPLAY_MEMORY_SIZE:
				replay_memory.pop(0)
			# save the transition to replay buffer
			next_observation = ob_proc.process(sess, next_observation)
			next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
			replay_memory.append(Transition(state, action, reward, next_state, done))
			# sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
			samples = random.sample(replay_memory, BATCH_SIZE)
			# calculate target Q values by target network
			states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
			# qs = sess.run(target_Q.Q, feed_dict={target_Q.X_pl: next_states_batch})
			qs = target_Q.get_Q(sess, next_states_batch)
			yi_batch = reward_batch + np.invert(done_batch).astype(np.float32) * GAMMA * np.amax(qs, axis=1)
			# Update network
			states_batch = np.array(states_batch)
			# sess.run(behavior_Q.train_op, feed_dict={behavior_Q.X_pl: states_batch, behavior_Q.y_pl: yi_batch, behavior_Q.actions_pl: action_batch})
			behavior_Q.update(sess, states_batch, yi_batch, action_batch)
			# Update target network every FREQ_UPDATE_TARGET_Q steps
			if total_t % FREQ_UPDATE_TARGET_Q == 0:
				update_target_network(sess, behavior_Q, target_Q)
				print 'Copy to target Q.'

			if done:
				print("Episode %5d reward: %3d best reward: %3d epsilon: %.8f" % (episode + 1, episode_reward, maxreward, epsilon_schedule[min(total_t, EXPLORE_STPES - 1)]))
				break

			state = next_state
			total_t += 1

		epr.append(episode_reward)
		if maxreward < episode_reward:
			maxreward = episode_reward

	out = []
	out.append(['reward'])
	for i in epr:
		out.append([i])

	with open('reward.csv', 'w') as f:
		w = csv.writer(f)
		w.writerows(out)

	if not os.path.exists('checkpoint_dir'):
		os.makedirs('checkpoint_dir')
	
	saver.save(tf.get_default_session(), 'checkpoint_dir/last')


if __name__ == '__main__':
	tf.app.run()
