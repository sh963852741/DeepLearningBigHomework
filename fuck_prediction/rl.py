"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import tensorflow.compat.v1 as tf
# import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 2000
ACTION_LEARNING_RATE = 0.001    # learning rate for actor
CRITIC_LEARNING_RATE = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v1'


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, action_dim, state_dim, action_bound,):
        # memory shape:[有多少组数据(MEMORY_CAPACITY), 之前的state + 后来的state + 采取了何种action + 1] 2维
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.action_dim, self.state_dim, self.action_bound = action_dim, state_dim, action_bound,
        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.action = self._build_actor(self.S,)
        q_value = self._build_critic(self.S, self.action, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        # 滑动平均
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_actor(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q_value)  # maximize the q
        self.action_optimizer = tf.train.AdamOptimizer(ACTION_LEARNING_RATE).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_value)
            self.critic_optimizer = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        return self.sess.run(self.action, {self.S: state[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
        br = bt[:, -self.state_dim - 1: -self.state_dim]
        bs_ = bt[:, -self.state_dim:]

        self.sess.run(self.action_optimizer, {self.S: bs})
        self.sess.run(self.critic_optimizer, {self.S: bs, self.action: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor(self, state, reuse=None, custom_getter=None):
        ''' 
        通过状态构建要采取的行动。

        Arguments:
        --------
        - self：
        - state：向量。
        '''
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(state, 128, activation=tf.nn.relu, name='l1', trainable=trainable)
            action = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(action, self.action_bound, name='scaled_a')

    def _build_critic(self, state, action, reuse=None, custom_getter=None):
        '''
        返回一个网络结构，具体为Q(state,action)
        Arguments:
        --------
        - self：
        - state：向量。
        - action：向量。
        '''
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_layer1 = 128
            weight1_state = tf.get_variable('w1_s', [self.state_dim, n_layer1], trainable=trainable)
            weight1_action = tf.get_variable('w1_a', [self.action_dim, n_layer1], trainable=trainable)
            bais1 = tf.get_variable('b1', [1, n_layer1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(state, weight1_state) + tf.matmul(action, weight1_action) + bais1)
            return tf.layers.dense(net, 1, trainable=trainable)  


###############################  training  ####################################


from reinforcement.environment import PUEEnviroment
# env = gym.make(ENV_NAME)
# env = env.unwrapped
# env.seed(1)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# action_bound = env.action_space.high

env = PUEEnviroment(".\\dist\\machine_gbr.pkl")

state_dim = 256 -64
action_dim = 64
action_bound = [30 for x in range(0,64)]

ddpg = DDPG(action_dim, state_dim, action_bound)

var = 10  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    state = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        action = ddpg.choose_action(state)
        action = np.clip(np.random.normal(action, var), 20, 25)    # add randomness to action selection for exploration
        s_, reward, done, info = env.step(action)

        # ddpg.store_transition(state, action, reward / 10, s_)
        ddpg.store_transition(state, action, reward, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        state = s_
        ep_reward += reward
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break

print('Running time: ', time.time() - t1)