import numpy as np
import pygame
import random

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def saveData(self):
        self.memory.rotate(7000)
        print('data saved')

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


BATCH_SIZE = 256
GAMMA = 0.99
EPS_DECAY = 0.99
MIN_EPSILON = 0.001
TAU = 0.005
LR = 0.005
epsilon = 1
policy_net = DQN(13, 8).to(device)
target_net = DQN(13, 8).to(device)
policy_net.load_state_dict(torch.load('model sa enemy kutijama/novimodel3enpol___300.00.pth'))
target_net.load_state_dict(torch.load('model sa enemy kutijama/novimodel3enpol___300.00.pth'))
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

memory = ReplayMemory(50000)

action = 0
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    global epsilon
    if sample > 0:
        with torch.no_grad():

            return policy_net(state).max(dim=1)[1].view(1, 1)
    else:
        rand_action = np.random.randint(0, 8)
        return torch.tensor([[rand_action]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


class Robot(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)

        self.image = pygame.Surface((10, 10))
        self.image.fill((255,0,0))
        self.rect = self.image.get_rect(topleft = (90, 130))
        self.old_rect = self.rect.copy()
        self.pos = pygame.math.Vector2(self.rect.center)
        self.direction = pygame.math.Vector2()
        self.speed = 0.8
        self.remembered_actions = deque([-1, -1, -1, -1], maxlen=4)
        self.target_left = 0
        self.target_right = 0
        self.target_bottom = 0
        self.target_up = 0

    def update_target_pos(self, x, y):
        if x-5 < self.rect.center[0] < x+5:
            self.target_left = 0
            self.target_right = 0
        elif (self.rect.center[0] < x):
            self.target_left = 0
            self.target_right = 1
        else:
            self.target_left = 1
            self.target_right = 0

        if y - 5 < self.rect.center[1] < y + 5:
            self.target_bottom = 0
            self.target_up = 0
        elif (self.rect.center[1] < y):
            self.target_bottom = 1
            self.target_up = 0
        else:
            self.target_bottom = 0
            self.target_up = 1

    def is_repeating(self):
        if self.remembered_actions[0] == self.remembered_actions[1] == self.remembered_actions[2] == \
                self.remembered_actions[3]:
            return False
        elif (self.remembered_actions[0] == self.remembered_actions[2]) and (
                self.remembered_actions[1] == self.remembered_actions[3]):
            return True
        else:
            return False

    def move(self, dir):

        if dir == 0:
            self.direction.y = -1
            self.direction.x = 0
        elif dir == 1:
            self.direction.y = 1
            self.direction.x = 0
        elif dir == 2:
            self.direction.x = -1
            self.direction.y = 0
        elif dir == 3:
            self.direction.x = 1
            self.direction.y = 0
        elif dir == 4:
            self.direction.y = -1
            self.direction.x = -1
        elif dir == 5:
            self.direction.y = 1
            self.direction.x = 1
        elif dir == 6:
            self.direction.x = -1
            self.direction.y = 1
        elif dir == 7:
            self.direction.x = 1
            self.direction.y = -1
        else:
            self.direction.x = 0
            self.direction.y = 0

    def window_collision(self):
        if self.rect.left < 0:
            self.rect.left = 0
            self.pos.x = self.rect.x
            return True
        if self.rect.right > 200:
            self.rect.right = 200
            self.pos.x = self.rect.x
            return True
        if self.rect.bottom > 200:
            self.rect.bottom = 200
            self.pos.y = self.rect.y
            return True
        return False

    def update(self, dir):
        self.old_rect = self.rect.copy()
        self.move(dir)

        self.pos.x += self.direction.x * self.speed
        self.rect.x = round(self.pos.x)
        self.pos.y += self.direction.y * self.speed
        self.rect.y = round(self.pos.y)


class EnemyBox(pygame.sprite.Sprite):
    def __init__(self, groups, obstacles, x, y):
        super().__init__(groups)
        self.image = pygame.Surface((20, 20))
        self.image.fill('gray')
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = pygame.math.Vector2(self.rect.center)
        self.old_rect = self.rect.copy()
        self.obstacles = obstacles

    # pomoc za koliziju nadjena na youtube tutorijalu
    def collision(self, direction):
        collision_sprites = pygame.sprite.spritecollide(self, self.obstacles, False)
        if collision_sprites:
            if direction == 'horizontal':
                for sprite in collision_sprites:
                    if self.rect.right >= sprite.rect.left and self.old_rect.right <= sprite.old_rect.left:
                        self.rect.right = sprite.rect.left
                        self.pos.x = self.rect.x
                    if self.rect.left <= sprite.rect.right and self.old_rect.left >= sprite.old_rect.right:
                        self.rect.left = sprite.rect.right
                        self.pos.x = self.rect.x

            if direction == 'vertical':
                for sprite in collision_sprites:
                    if self.rect.bottom >= sprite.rect.top and self.old_rect.bottom <= sprite.old_rect.top:
                        self.rect.bottom = sprite.rect.top
                        self.pos.y = self.rect.y
                    if self.rect.top <= sprite.rect.bottom and self.old_rect.top >= sprite.old_rect.bottom:
                        self.rect.top = sprite.rect.bottom
                        self.pos.y = self.rect.y

    def hitted(self):
        old_dist = abs(self.old_rect.center[1])
        dist = abs(self.rect.center[1])
        return dist - old_dist
    def update(self, dir):
        self.old_rect = self.rect.copy()
        self.collision('horizontal')
        self.collision('vertical')

class Box(pygame.sprite.Sprite):
    def __init__(self, groups, obstacles, x, y):
        super().__init__(groups)
        self.image = pygame.Surface((10, 10))
        self.image.fill('blue')
        self.rect = self.image.get_rect(topleft=(x, y))
        self.pos = pygame.math.Vector2(self.rect.center)
        self.old_rect = self.rect.copy()
        self.attached_right = False
        self.attached_left = False
        self.attached_bottom = False
        self.attached_up = False
        self.attachable = True
        self.attached = 0
        self.difference = 0
        self.done = False
    def is_end(self):
        if self.rect.bottom <= 20:
            self.attachable = False
            self.attached_left = False
            self.attached_up = False
            self.attached_bottom = False
            self.attached_right = False
            self.done = True
            return True
        else:
            self.attachable = True
            return False

    def is_closer_to_target(self):
        old_dist = abs(self.old_rect.center[1])
        dist = abs(self.rect.center[1])
        if dist < old_dist:
            return True
        else:
            return False

    def is_further_from_target(self):
        old_dist = abs(self.old_rect.center[1])
        dist = abs(self.rect.center[1])
        if dist > old_dist:
            return True
        else:
            return False

    def check_if_robot_attached(self, x, y):
        if self.attachable:
            if y - 13 <= self.rect.bottom <= y - 7 and self.rect.center[0] - 11 <= x <= self.rect.center[0] + 11:
                self.difference = self.rect.center[0] - x
                self.attached = 1
                self.attached_bottom = True
            if self.rect.center[1] - 11 <= y <= self.rect.center[1] + 11 and x + 7 <= self.rect.left <= x + 13:
                self.difference = self.rect.center[1] - y
                self.attached = 1
                self.attached_left = True
            if y + 7 <= self.rect.top <= y + 13 and self.rect.center[0] - 11 <= x <= self.rect.center[0] + 11:
                self.difference = self.rect.center[0] - x
                self.attached = 1
                self.attached_up = True
            if self.rect.center[1] - 11 <= y <= self.rect.center[1] + 11 and x - 13 <= self.rect.right <= x - 7:
                self.difference = self.rect.center[1] - y
                self.attached = 1
                self.attached_right = True

    def move_with_robot(self, x, y):
        if self.attached_bottom:
            self.rect.right = self.difference + x + 5
            self.rect.bottom = y - 5
            self.pos.x = self.rect.x
            self.pos.y = self.rect.y
        if self.attached_left:
            self.rect.top = self.difference + y - 5
            self.rect.left = x + 5
            self.pos.x = self.rect.x
            self.pos.y = self.rect.y
        if self.attached_up:
            self.rect.right = self.difference + x + 5
            self.rect.top = y + 5
            self.pos.x = self.rect.x
            self.pos.y = self.rect.y
        if self.attached_right:
            self.rect.top = self.difference + y - 5
            self.rect.right = x - 5
            self.pos.x = self.rect.x
            self.pos.y = self.rect.y

    def update(self, dir):
        self.old_rect = self.rect.copy()

class warehouse():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((200, 200), pygame.SCALED)
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()
        self.boxes = []
        self.robot = Robot([self.all_sprites, self.collision_sprites])
        self.box1 = Box(self.all_sprites, self.collision_sprites, 45, 40)
        self.box2 = Box(self.all_sprites, self.collision_sprites, 85, 40)
        self.boxes.append(self.box1)
        self.boxes.append(self.box2)
        self.current_box = self.box1
        self.enemyBox = EnemyBox(self.all_sprites, self.collision_sprites, 75, 140)
        self.enemyBox1 = EnemyBox(self.all_sprites, self.collision_sprites, 75, 140)
        self.enemyBox2 = EnemyBox(self.all_sprites, self.collision_sprites, 75, 140)
        self.enemyBox.obstacles.add(self.box1)
        self.enemyBox.obstacles.add(self.box2)
        self.enemyBox1.obstacles.add(self.box1)
        self.enemyBox1.obstacles.add(self.box2)
        self.enemyBox.obstacles.add(self.enemyBox1)
        self.enemyBox1.obstacles.add(self.enemyBox)
        self.enemies = []
        self.enemies.append(self.enemyBox)
        self.enemies.append(self.enemyBox1)
        self.enemies.append(self.enemyBox2)
        self.done = False
        self.reward = 0
        self.state = [0,0,0,0,0]

    def reset(self):
        self.current_box = self.box1
        for box in self.boxes:
            box.done = False
            box.attached_right = False
            box.attached_up = False
            box.attached_bottom = False
            box.attached_left = False
            box.attachable = True
            box.attached = 0

        x, y = self.get_robot_random_pos()
        self.robot.rect = self.robot.image.get_rect(center=(x, y))
        self.robot.old_rect = self.robot.rect.copy()
        self.robot.pos = pygame.math.Vector2(self.robot.rect.topleft)

        for box in self.boxes:
            x, y = self.get_box_random_pos()
            while not self.position_possible(x, y):
                x, y = self.get_box_random_pos()
            box.rect = box.image.get_rect(center=(x, y))
            box.old_rect = box.rect.copy()
            box.pos = pygame.math.Vector2(box.rect.topleft)

        self.set_enemy_box_pos()

        self.robot.update_target_pos(self.current_box.rect.center[0], self.current_box.rect.center[1])
        self.done = False
        self.reward = 0
        self.state = [self.robot.target_up, self.robot.target_bottom, self.robot.target_left,
                      self.robot.target_right,
                      self.current_box.attached,
                      self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.enemyBox.rect.center[0]),
                      self.normalize(self.enemyBox.rect.center[1]),
                      self.normalize(self.enemyBox1.rect.center[0]),
                      self.normalize(self.enemyBox1.rect.center[1]),
                      self.normalize(self.enemyBox2.rect.center[0]),
                      self.normalize(self.enemyBox2.rect.center[1])
                      ]
        return self.state

    def set_enemy_box_pos(self):
        for enemy in self.enemies:
            x = np.random.randint(32, 128)
            y = np.random.randint(32, 128)
            enemy.rect = enemy.image.get_rect(center=(x, y))
            enemy.old_rect = enemy.rect.copy()
            enemy.pos = pygame.math.Vector2(enemy.rect.topleft)
    def get_robot_random_pos(self):
        x = np.random.randint(60, 178)
        y = np.random.randint(60, 178)
        return x, y

    def get_box_random_pos(self):
        x = np.random.randint(32, 168)
        y = np.random.randint(42, 178)
        return x, y

    def position_possible(self, x, y):
        if x - 15 < self.robot.rect.center[0] < x + 15 and y - 15 < self.robot.rect.center[1] < y + 15:
            return False
        return True

    def normalize(self, x):
        return (2 * x / 200) - 1

    def is_robot_closer_to_box(self, box):
        old_dist = abs(self.robot.old_rect.center[0] - box.old_rect.center[0]) + abs(
            self.robot.old_rect.center[1] - box.old_rect.center[1])
        dist = abs(self.robot.rect.center[0] - box.rect.center[0]) + abs(
            self.robot.rect.center[1] - box.rect.center[1])
        if dist < old_dist:
            return True
        else:
            return False

    def update_reward(self):
        if self.enemyBox.hitted() != 0:
            self.reward = self.enemyBox.hitted()/8
        elif self.enemyBox1.hitted() != 0:
            self.reward = self.enemyBox1.hitted()/8
        elif self.enemyBox2.hitted() != 0:
            self.reward = self.enemyBox2.hitted() / 8
        elif self.current_box.is_closer_to_target():
            self.reward = 0.25
        elif self.current_box.is_further_from_target():
            self.reward = -0.325
        elif self.is_robot_closer_to_box(self.current_box):
            self.reward = 0.215
        else:
            self.reward = -0.55

    def change_target(self):
        for box in self.boxes:
            if not box.done:
                self.current_box.attachable = True
                self.current_box = box

    def step(self, dir):
        self.screen.fill('white')

        pygame.draw.rect(self.screen, (231, 255, 182), pygame.Rect(0, 0, 200, 20))
        self.all_sprites.update(dir)

        if self.current_box.attached:
            self.current_box.move_with_robot(self.robot.rect.center[0], self.robot.rect.center[1])
        else:
            self.current_box.check_if_robot_attached(self.robot.rect.center[0], self.robot.rect.center[1])
        self.all_sprites.draw(self.screen)

        self.update_reward()

        if self.current_box.is_end():
            self.change_target()
        if self.box1.is_end() and self.box2.is_end():
            self.done = True
            self.reward = 20
        if self.robot.window_collision():
            self.reward = -2
            self.done = True
        self.robot.update_target_pos(self.current_box.rect.center[0], self.current_box.rect.center[1])
        pygame.display.update()
        self.state = [self.robot.target_up, self.robot.target_bottom, self.robot.target_left,
                      self.robot.target_right,
                      self.current_box.attached,
                      self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.enemyBox.rect.center[0]),
                      self.normalize(self.enemyBox.rect.center[1]),
                      self.normalize(self.enemyBox1.rect.center[0]),
                      self.normalize(self.enemyBox1.rect.center[1]),
                      self.normalize(self.enemyBox2.rect.center[0]),
                      self.normalize(self.enemyBox2.rect.center[1])
                      ]
        return self.state, self.reward, self.done

env = warehouse()

global_step = 0
for i_episode in range(3000):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward=0
    step = 0
    done = False
    while not done:
        action = select_action(state)
        observation, reward, terminated = env.step(action.item())
        ep_reward += reward
        #print(reward)
        reward = torch.tensor([reward], device=device)
        done = terminated
        env.robot.remembered_actions.append(action)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        #memory.push(state, action, next_state, reward)

        state = next_state

        #optimize_model()
        #target_net_state_dict = target_net.state_dict()
        #policy_net_state_dict = policy_net.state_dict()
        #for key in policy_net_state_dict:
        #    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        #target_net.load_state_dict(target_net_state_dict)
        #global_step += 1
        step+=1
        if(step > 3500):
            done=True
'''
    if(global_step > 50100):
        global_step = 0
        memory.saveData()
    if (i_episode % 10) == 0:
        scheduler.step()
        print('epsilon: ', epsilon)
    if (i_episode % 50) == 0:
        torch.save(policy_net.state_dict(), f'model sa enemy kutijama/model3ennpol__{i_episode:_>7.2f}.pth')
        torch.save(target_net.state_dict(), f'model sa enemy kutijama/model3enntarget__{i_episode:_>7.2f}.pth')
    if epsilon > MIN_EPSILON:
        epsilon *= EPS_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
'''
