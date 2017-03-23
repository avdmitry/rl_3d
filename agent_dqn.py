#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import sys
import os

import numpy as np
import cv2
import tensorflow as tf

def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

lab = False
load_model = False
train = True
test_display = True
test_write_video = False
path_work_dir = "~/rl_3d/"
vizdoom_path = "~/ViZDoom/"
vizdoom_scenario = vizdoom_path + "scenarios/simpler_basic.wad"

# Lab parameters.
if (lab):
    from env_lab import EnvLab

    learning_rate = 0.00025  # 0.001
    discount_factor = 0.99
    step_num = int(5e5)  # int(1e6)
    replay_memory_size = int(1e6)
    replay_memory_batch_size = 64

    # Exploration rate.
    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * step_num

    frame_repeat = 10  # 4
    channels = 3
    resolution = (40, 40) + (channels,)  # Original: 240x320

    model_path = path_work_dir + "model_lab_dqn/"
    save_each = 0.01 * step_num
    step_load = 100

# Vizdoom parameters.
if (not lab):
    from env_vizdoom import EnvVizDoom

    learning_rate = 0.00025
    discount_factor = 0.99
    step_num = int(5e4)
    replay_memory_size = int(1e5)
    replay_memory_batch_size = 64

    frame_repeat = 10
    channels = 3
    resolution = (40, 40) + (channels,) # Original: 480x640

    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * step_num

    model_path = path_work_dir + "model_vizdoom_dqn/"
    save_each = 0.01 * step_num
    step_load = 100

MakeDir(model_path)
model_name = model_path + "dqn"

# Global variables.
env = None

def PrintStat(elapsed_time, step, step_num, train_scores):
    steps_per_s = 1.0 * step / elapsed_time
    steps_per_m = 60.0 * step / elapsed_time
    steps_per_h = 3600.0 * step / elapsed_time
    steps_remain = step_num - step
    remain_h = int(steps_remain / steps_per_h)
    remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
    remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
    elapsed_h = int(elapsed_time / 3600)
    elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
    elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
    print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s), file=sys.stderr)

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
    print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)

def Preprocess(img):
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(20)
    if (channels == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (resolution[1], resolution[0]))
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(200)
    return np.reshape(img, resolution)

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

        self.s[self.pos, ...] = s
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        idx = random.sample(xrange(0, self.size-2), sample_size)
        idx2 = []
        for i in idx:
            idx2.append(i + 1)
        return self.s[idx], self.a[idx], self.s[idx2], self.isterminal[idx], self.r[idx]

class Model(object):
    def __init__(self, session, actions_count):

        self.session = session

        # Create the input.
        self.s_ = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32)
        self.q_ = tf.placeholder(shape=[None, actions_count], dtype=tf.float32)

        # Create the network.
        conv1 = tf.contrib.layers.conv2d(self.s_, num_outputs=8, kernel_size=[3, 3], stride=[2, 2])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128)

        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=actions_count, activation_fn=None)
        self.action = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q_, self.q)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def Learn(self, state, q):

        state = state.astype(np.float32)
        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.s_: state, self.q_: q})
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        self.model = Model(self.session, num_actions)
        self.memory = ReplayMemory(replay_memory_size)

        self.rewards = 0

        self.saver = tf.train.Saver(max_to_keep=1000)
        if (load_model):
            model_name_curr = model_name + "_{:04}".format(step_load)
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

        if (random.random() <= 0.05):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(state)

        return a

    def Step(self, iteration):

        s = Preprocess(env.Observation())

        # Epsilon-greedy.
        if (iteration < eps_decay_iter):
            eps = start_eps - iteration / eps_decay_iter * (start_eps - end_eps)
        else:
            eps = end_eps

        if (random.random() <= eps):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(s)

        reward = env.Act(a, frame_repeat)
        self.rewards += reward

        isterminal = not env.IsRunning()
        self.memory.Add(s, a, isterminal, reward)
        self.LearnFromMemory()

    def Train(self):

        print("Starting training.")
        start_time = time.time()
        train_scores = []
        env.Reset()
        for step in xrange(1, step_num+1):
            self.Step(step)
            if (not env.IsRunning()):
                train_scores.append(self.rewards)
                self.rewards = 0
                env.Reset()

            if (step % save_each == 0):
                model_name_curr = model_name + "_{:04}".format(int(step / save_each))
                print("\nSaving the network weigths to:", model_name_curr, file=sys.stderr)
                self.saver.save(self.session, model_name_curr)

                PrintStat(time.time() - start_time, step, step_num, train_scores)

                train_scores = []

        env.Reset()

def Test(agent):
    if (test_write_video):
        size = (640, 480)
        fps = 30.0 #/ frame_repeat
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 30
    while (num_episodes != 0):
        if (not env.IsRunning()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1

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

            reward = env.Act(action, 1)
            reward_total += reward

            if (not env.IsRunning()):
                break

            state_raw = env.Observation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="the GPU to use")
    args = parser.parse_args()

    if (args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if (lab):
        env = EnvLab(80, 80, 60, "seekavoid_arena_01")
    else:
        env = EnvVizDoom(vizdoom_scenario)

    agent = Agent(env.NumActions())

    if (train):
        agent.Train()

    Test(agent)
