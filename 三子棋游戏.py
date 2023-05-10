import collections
from pygame import *
import sys
import time
import pygame
import numpy as np
from 模型列表 import *
import matplotlib.pyplot as plt
from itertools import combinations
import tensorflow as tf
import tqdm
import statistics
from typing import Tuple, List
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()
pygame.init()
width, height = 600, 600
size_temp = [width // 5, width - (width // 5)]
left_top = (size_temp[0], size_temp[0])
right_top = (size_temp[1], size_temp[0])
left_down = (size_temp[0], size_temp[1])
right_down = (size_temp[1], size_temp[1])


def get_coord(l_t, lent):
    coord_l = []
    for _ in range(3):
        y = ((_ + 1) * lent) - lent // 2
        for _1 in range(3):
            x = ((_1 + 1) * lent) - lent // 2
            coord = (l_t[0] + x, l_t[1] + y)
            coord_l.append(coord)
    return coord_l


class ThreeChess(object):
    def __init__(self, w, h, rate=0.5):
        self.human = None
        self.lines_all = []
        self.rate = rate
        self.model = model
        self.screen = display.set_mode((w, h))
        self.over = False
        self.km = right_down[0] - left_down[0]
        self.coord_list = get_coord(left_top, self.km // 3)
        self.grid_to_state = {self.coord_list[a]: a for a in range(9)}
        self.state_to_grid = {a: self.coord_list[a]for a in range(9)}
        self.font = pygame.font.SysFont('SimHei', 30)
        self.reward = 0
        a = np.arange(9)
        self.grid_table = []
        for J in range(1, 10, 1):
            for index, value in enumerate(combinations(a, J)):
                self.grid_table.append(list(value))
        self.grid_table = self.grid_table[::-1]
        self.state_fun = lambda list1: tf.constant([self.grid_table.index(list1)], dtype=tf.float32)

    def print_font(self, text, c, position):
        textimage = self.font.render(text, True, c)
        self.screen.blit(textimage, position)

    def judge(self, coord_list):
        coord_list.sort()
        w_or_d = 5
        jxz = 0
        hq = 0
        if (0 in coord_list and 1 in coord_list and 2 in coord_list)\
                or (3 in coord_list and 4 in coord_list and 5 in coord_list)\
                or (6 in coord_list and 7 in coord_list and 8 in coord_list):
            return w_or_d, True
        elif (0 in coord_list and 3 in coord_list and 6 in coord_list)\
                or (1 in coord_list and 4 in coord_list and 7 in coord_list)\
                or (2 in coord_list and 5 in coord_list and 8 in coord_list):
            return w_or_d, True
        elif (0 in coord_list and 4 in coord_list and 8 in coord_list) or \
                (2 in coord_list and 4 in coord_list and 6 in coord_list):
            return w_or_d, True
        elif len(coord_list) == 5:
            return hq, True
        else:
            return jxz, False

    def computer_step(self):
        action_per = np.random.permutation(self.nA)[0]
        self.computer.append(action_per)
        self._display(action_per, human=False)
        self.nA.remove(action_per)
        r, done = self.judge(self.computer)
        self.reward -= r
        all_step = self.human + self.computer
        all_step = sorted(all_step)
        state = self.state_fun(all_step)
        return state, -r, done

    def step(self, action_step):
        action_step = self.nA[int(action_step)]
        self.human.append(action_step)
        self.nA.remove(action_step)  # 可能发生错误
        self._display(action_step)
        r, done = self.judge(self.human)
        self.reward += r
        all_step = self.human + self.computer
        all_step = sorted(all_step)
        state = self.state_fun(all_step)
        if done:
            return state, r, done
        else:
            return self.computer_step()

    def running_episode(self, epoch, initialize_state):
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        self.actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.initialize_state_shape = initialize_state.shape
        state = initialize_state
        self.screen.fill((255, 255, 255))
        self.lines_all.append([left_top, left_down])
        self.human = []
        self.nA = [n for n in range(9)]
        self.computer = []
        t = 0
        state = tf.expand_dims(state, 0)
        state, reward, done = self.step_func(state, t)  # 模型下棋
        while True:
            t += 1
            for e in event.get():
                if e.type == QUIT:
                    sys.exit()
                elif e.type == KEYUP:
                    if e.key == K_RETURN:
                        sys.exit()
            for a, b in self.lines_all:
                for _ in range(4):
                    x = self.km - _ * (self.km // 3) + left_top[0]
                    draw.line(self.screen, (0, 0, 0), start_pos=(x, a[1]), end_pos=(x, b[1]), width=3)
                    draw.line(self.screen, (0, 0, 0), start_pos=(a[1], x), end_pos=(b[1], x), width=3)
            self.print_font('第{}回合'.format(epoch + 1), (255, 0, 0), (50, 50))
            self.print_font('奖励:{}'.format(self.reward),
                            (255, 0, 0), (250, 80))
            if done:
                break
            state = tf.expand_dims(state, 0)
            state, reward, done = self.step_func(state, t)  # 模型下棋
        actions = self.actions.stack()
        values = self.values.stack()
        rewards = self.rewards.stack()
        return actions, values, rewards

    def step_func(self, s, t):
        a1, a2, a3, a4, value = model(s)
        if len(self.nA) == 9:
            action_logit = a1
        elif len(self.nA) == 7:
            action_logit = a2
        elif len(self.nA) == 5:
            action_logit = a3
        elif len(self.nA) == 3:
            action_logit = a4
        elif len(self.nA) == 1:
            return self.step(0)
        action = tf.random.categorical(action_logit, 1)[0, 0]
        action_prob_t = tf.nn.softmax(action_logit)
        self.actions = self.actions.write(t, action_prob_t[0, action])
        self.values = self.values.write(t, tf.squeeze(value))
        display.update()
        time.sleep(self.rate)
        state, reward, done = self.step(action)
        self.rewards = self.rewards.write(t, reward)
        state.set_shape(self.initialize_state_shape)
        time.sleep(self.rate)
        return state, reward, done

    def _display(self, action, human=True):
        center = self.state_to_grid[action]
        if human:
            c = (255, 0, 0)
        else:
            c = (0, 255, 0)
        draw.circle(self.screen, c, center, radius=20)
        display.update()


def get_expected_return(rewards: tf.Tensor,
                        gamma: float,
                        standardization: bool = True) -> tf.Tensor:
    n = tf.shape(rewards)[0]
    return_value = tf.constant(0.0)
    return_value_shape = return_value.shape
    returns = tf.TensorArray(tf.float32, size=n, dynamic_size=True)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    for i in tf.range(n):
        reward = rewards[i]
        return_value = return_value * gamma + reward
        return_value.set_shape(return_value_shape)
        returns = returns.write(i, return_value)
    returns = returns.stack()
    returns = returns[::-1]
    if standardization:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns


def computer_loss(returns, values, actions):
    f1_1 = returns - values
    f1_2 = tf.math.log(actions)
    f1 = -tf.math.reduce_sum(f1_2 * f1_1)
    f2 = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    critic = f2(values, returns)
    return f1 + critic


def train_episode(optimizers, initialize_state, gamma, max_episode):
    with tf.GradientTape() as tape:
        actions, values, rewards = env.running_episode(i, initialize_state)
        returns = get_expected_return(rewards, gamma)
        actions, values, returns = [tf.expand_dims(x, 1) for x in [actions, values, returns]]
        loss = computer_loss(returns, values, actions)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizers.apply_gradients(zip(grad, model.trainable_variables))
    reward = tf.math.reduce_sum(rewards)
    return reward


plt.rcParams['font.sans-serif'] = ['SimHei']
echos = 100
hidden_num = 128
gamma = 0.99
max_episode_criterion = 100
device_name = tf.test.gpu_device_name()
with tf.device(device_name):
    model = SYH(units_hidden=hidden_num)
optimizer = tf.keras.optimizers.Adam(0.02)
env = ThreeChess(600, 600, rate=0)
episode_deque = collections.deque(maxlen=max_episode_criterion)
with tqdm.trange(echos) as t:
    for tt in t:
        for i in tf.range(50):
            r = int(train_episode(optimizer, tf.constant([5], dtype=tf.float32), gamma, 50))
            episode_deque.append(r)
            deque_mean = statistics.mean(episode_deque)
            t.set_description(f'Epochs:{tt}')
            t.set_postfix(report=r, running_report=deque_mean)


