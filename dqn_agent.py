from __future__ import division
from __future__ import print_function

import argparse
from random import sample, randint, random
from time import time
from tqdm import trange
import sys

import numpy as np
import cv2
import tensorflow as tf

from env_lab import EnvLab
from env_vizdoom import EnvVizDoom

def MakeDir(path):
    try:
        import os
        os.makedirs(path)
    except:
        pass

lab = False
load_model = False
train = True
test_display = True
test_write_video = False
path_work_dir = "~/lab/python/"
path_vizdoom = "~/ViZDoom/"

# Lab parameters.
if (lab):
    learning_rate = 0.00025  # 0.001
    discount_factor = 0.99
    iteration_num = int(5e5)  # int(1e6)
    replay_memory_size = int(1e6)
    replay_memory_batch_size = 64

    # Exploration rate.
    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * iteration_num

    frame_repeat = 10  # 4
    channels = 3
    resolution = (40, 40) + (channels,)  # Original: 240x320

    model_save_path = path_work_dir + "model_lab/"
    save_each = 0.01 * iteration_num
    iter_to_load = 100

# Vizdoom parameters.
if (not lab):
    learning_rate = 0.00025
    discount_factor = 0.99
    iteration_num = int(5e4)
    replay_memory_size = int(1e5)
    replay_memory_batch_size = 64

    frame_repeat = 10
    channels = 3
    resolution = (40, 40) + (channels,) # Original: 480x640

    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * iteration_num

    model_save_path = path_work_dir + "model_vizdoom/"
    save_each = 0.1 * iteration_num
    iter_to_load = 10

MakeDir(model_save_path)
model_save_name = model_save_path + "dqn"

# Global variables.
env = None
agent = None

def Preprocess(img):
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(20)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (resolution[1], resolution[0]))
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(200)
    return img

class ReplayMemory(object):
    def __init__(self, capacity):

        self.s = np.zeros((capacity,) + resolution, dtype=np.uint8)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def Add(self, s, action, isterminal, reward):

        #self.s[self.pos, :, :, 0] = s # gray
        self.s[self.pos, ...] = s
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        idx = sample(xrange(0, self.size-2), sample_size)
        idx2 = []
        for i in idx:
            idx2.append(i + 1)
        return self.s[idx], self.a[idx], self.s[idx2], self.isterminal[idx], self.r[idx]

class Model(object):
    def __init__(self, session, actions_count):

        self.session = session

        # Create the input.
        self.s_ = tf.placeholder(tf.float32, [None] + list(resolution))
        self.q_ = tf.placeholder(tf.float32, [None, actions_count])

        # Create the network.
        self.conv1 = tf.contrib.layers.conv2d(self.s_, num_outputs=8, kernel_size=[3, 3], stride=[2, 2])
        self.conv2 = tf.contrib.layers.conv2d(self.conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        self.conv2_flat = tf.contrib.layers.flatten(self.conv2)
        self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat, num_outputs=128)

        self.q = tf.contrib.layers.fully_connected(self.fc1, num_outputs=actions_count, activation_fn=None)
        self.action = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q, self.q_)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def Learn(self, s, q):

        s = s.astype(np.float32)
        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.s_: s, self.q_: q})
        return l

    def GetQ(self, state):

        state = state.astype(np.float32)
        return self.session.run(self.q, feed_dict={self.s_: state})

    def GetAction(self, state):

        state = state.astype(np.float32)
        state = state.reshape([1] + list(resolution))
        return self.session.run(self.action, feed_dict={self.s_: state})[0]

class Agent(object):

    def __init__(self, num_actions):

        self.session = tf.Session()

        self.model = Model(self.session, num_actions)
        self.memory = ReplayMemory(replay_memory_size)

        self.rewards = 0

        self.saver = tf.train.Saver(max_to_keep=1000)
        if (load_model):
            model_name_curr = model_save_name + "_{:04}".format(iter_to_load)
            print("Loading model from: ", model_name_curr)
            self.saver.restore(self.session, model_name_curr)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

        self.num_actions = num_actions

    def LearnFromMemory(self):

        if (self.memory.size > 2*replay_memory_batch_size):
            s1, a, s2, isterminal, r = self.memory.Get(replay_memory_batch_size)

            q = self.model.GetQ(s1)
            q2 = np.max(self.model.GetQ(s2), axis=1)
            q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * discount_factor * q2
            self.model.Learn(s1, q)

    def GetAction(self, state):

        if (random() <= 0.05):
            a = randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(state)

        return env.MapActions(a)

    def Step(self, iteration):

        s = Preprocess(env.Observation())

        # Epsilon-greedy.
        if (iteration < eps_decay_iter):
            eps = start_eps - iteration / eps_decay_iter * (start_eps - end_eps)
        else:
            eps = end_eps

        if (random() <= eps):
            a = randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(s)

        reward = env.Act(env.MapActions(a), frame_repeat)
        self.rewards += reward

        isterminal = not env.IsRunning()
        self.memory.Add(s, a, isterminal, reward)
        self.LearnFromMemory()

    def Train(self):

        print("Starting training.")
        time_start = time()
        train_episodes_finished = 0
        train_scores = []
        env.Reset()
        for iter in trange(1, iteration_num+1):
            self.Step(iter)
            if (not env.IsRunning()):
                train_episodes_finished += 1
                train_scores.append(self.rewards)
                self.rewards = 0
                env.Reset()

            if (iter % save_each == 0):
                model_name_curr = model_save_name + "_{:04}".format(int(iter / save_each))
                print("\nSaving the network weigths to:", model_name_curr, file=sys.stderr)
                self.saver.save(self.session, model_name_curr)

                print("Episodes: {}".format(train_episodes_finished), file=sys.stderr)

                mean_train = 0
                std_train = 0
                min_train = 0
                max_train = 0
                if (len(train_scores) > 0):
                    train_scores = np.array(train_scores)
                    mean_train = train_scores.mean()
                    std_train = train_scores.std()
                    min_train = train_scores.min()
                    max_train = train_scores.max()
                print("Results: mean: {}, std: {}, min: {}, max: {}".format(mean_train, std_train, min_train, max_train), file=sys.stderr)

                train_episodes_finished = 0
                train_scores = []

        print("Training time: {} hours".format((time() - time_start) / 3600))
        env.Reset()

def Test():
    if (test_write_video):
        size = (640, 480)
        fps = 30.0 #/ frame_repeat
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 30
    while (True):
        if (not env.IsRunning()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1
            if (num_episodes==0):
                break

        state_raw = env.Observation()
        state = Preprocess(state_raw)
        action = agent.GetAction(state)

        for _ in xrange(frame_repeat):
            # Display.
            if (test_display):
                cv2.imshow("frame-test", state_raw)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw)

            reward = env.Act(action, 1)  #frame_repeat)
            reward_total += reward

            state_raw = env.Observation()
            if (not env.IsRunning()):
                break

if __name__ == '__main__':

    if (lab):
        env = EnvLab(40, 40, 60, "seekavoid_arena_01")
    else:
        env = EnvVizDoom(path_vizdoom + "scenarios/simpler_basic.cfg")

    agent = Agent(env.NumActions())

    if (train):
        agent.Train()

    Test()

