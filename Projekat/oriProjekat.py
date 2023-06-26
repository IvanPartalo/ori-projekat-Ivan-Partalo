# pomoc za implementaciju DQN-a nadjena na pytorch tutorijalu za DQN
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import pygame
import random

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# TAU is the update rate of the target network
BATCH_SIZE = 128
GAMMA = 0.99
EPS_DECAY = 0.99
MIN_EPSILON = 0.001
TAU = 0.005
LR = 1e-4
epsilon = 1
policy_net = DQN(7, 4).to(device)
target_net = DQN(7, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(30000)

action = 0


def select_action(state):
    sample = random.random()
    global epsilon

    if sample > epsilon:
        with torch.no_grad():
            #max(dim=1) vrati najveci broj u izlazu, [1] vrati indeks gde je nadjen (a to je akcija koja ce se izvrsiti)
            return policy_net(state).max(dim=1)[1].view(1, 1)
    else:
        rand_action = np.random.randint(0, 4)
        return torch.tensor([[rand_action]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch)


    # Qnew = reward + gamma*maximumFutureQ
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = reward_batch + next_state_values * GAMMA

    # Srednja kvadratna greska izmedju predvidjenih Q(s,a) neuronske mreze i izracunatih Q(s,a) iz formule, na osnovu ovoga se uci neuronska mreza.
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
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2()
        self.speed = 2
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
        else:
            self.direction.x = 0
            self.direction.y = 0

    def collided_with_window(self):
        if self.rect.left < 0:
            self.rect.left = 0
            self.pos.x = self.rect.x
            return True
        if self.rect.right > 200:
            self.rect.right = 200
            self.pos.x = self.rect.x
            return True
        if self.rect.top < 0:
            self.rect.top = 0
            self.pos.y = self.rect.y
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


class Box(pygame.sprite.Sprite):
    def __init__(self, groups, x, y):
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
        self.attached = 0
        self.difference = 0
        self.done = False

    def is_end(self):
        if self.rect.bottom <= 120 and self.rect.top >= 60 and self.rect.left >= 60 and self.rect.right <= 120:
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

    def check_if_robot_attached(self, x, y):
        if y - 13 <= self.rect.bottom <= y - 7 and self.rect.center[0] - 10 <= x <= self.rect.center[0] + 10:
            self.difference = self.rect.center[0] - x
            self.attached = 1
            self.attached_bottom = True
        if self.rect.center[1] - 10 <= y <= self.rect.center[1] + 10 and x + 7 <= self.rect.left <= x + 13:
            self.difference = self.rect.center[1] - y
            self.attached = 1
            self.attached_left = True
        if y + 7 <= self.rect.top <= y + 13 and self.rect.center[0] - 10 <= x <= self.rect.center[0] + 10:
            self.difference = self.rect.center[0] - x
            self.attached = 1
            self.attached_up = True
        if self.rect.center[1] - 10 <= y <= self.rect.center[1] + 10 and x - 13 <= self.rect.right <= x - 7:
            self.difference = self.rect.center[1] - y
            self.attached = 1
            self.attached_right = True

    

    def move_with_robot(self, x, y):
        # cuva se pozicija gde je bila kutija kada ju je robot prikljucio
        # posto se salje centar robota, sabira/oduzima se sa 5 
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

    def is_closer_to_target(self):
        old_dist = 0
        dist = 0
        # znaci umesto ovako zakucanih vrednosti bolje je da promenim na neke "konstante" koje cu menjati
        old_dist = abs(self.old_rect.center[0] - 90) + abs(self.old_rect.center[1] - 90)
        dist = abs(self.rect.center[0] - 90) + abs(self.rect.center[1] - 90)
        if dist < old_dist:
            return True
        else:
            return False

    def is_further_from_target(self):
        old_dist = 0
        dist = 0
        # znaci umesto ovako zakucanih vrednosti bolje je da promenim na neke "konstante" koje cu menjati
        old_dist = abs(self.old_rect.center[0] - 90) + abs(self.old_rect.center[1] - 90)
        dist = abs(self.rect.center[0] - 90) + abs(self.rect.center[1] - 90)
        if dist > old_dist:
            return True
        else:
            return False

    def update(self, dir):
        self.old_rect = self.rect.copy()

class Warehouse():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((200,200), pygame.SCALED)
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()
        self.robot = Robot([self.all_sprites, self.collision_sprites])
        self.box1 = Box(self.all_sprites, 45, 40)
        self.box2 = Box(self.all_sprites, 45, 60)
        self.boxes=[]
        self.boxes.append(self.box1)
        self.boxes.append(self.box2)
        self.current_box = self.box1
        self.done = False
        self.reward = 0
        self.state = [0, 0, 0, 0, 0, 0, 0]

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

        self.robot.rect = self.robot.image.get_rect(topleft=(20, 20))
        self.robot.old_rect = self.robot.rect.copy()
        self.robot.pos = pygame.math.Vector2(self.robot.rect.topleft)
        z=0
        for box in self.boxes:
            box.rect = box.image.get_rect(topleft=(30, 100+z))
            box.old_rect = box.rect.copy()
            box.pos = pygame.math.Vector2(box.rect.topleft)
            z+=20

        self.done = False
        self.reward = 0
        self.state = [self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.current_box.rect.center[0]),
                      self.normalize(self.current_box.rect.center[1]),
                      self.normalize(90), self.normalize(90),
                      self.current_box.attached]
        return self.state

    def normalize(self, x):
        return (2 * x / 200) - 1

    def change_target(self):
        if not self.box1.done and self.box1.is_end():
            if not self.box2.is_end():
                print('target changed')
                self.box2.attachable = True
                self.current_box = self.box2
        elif not self.box2.done and self.box2.is_end():
            if not self.box1.is_end():
                print('target changed')
                self.box1.attachable = True
                self.current_box = self.box1

    def is_robot_closer_to_box(self, box):
        old_dist = 0
        dist = 0
        old_dist = abs(self.robot.old_rect.center[0] - box.old_rect.center[0]) + abs(
            self.robot.old_rect.center[1] - box.old_rect.center[1])
        dist = abs(self.robot.rect.center[0] - box.rect.center[0]) + abs(
            self.robot.rect.center[1] - box.rect.center[1])
        if dist < old_dist:
            return True
        else:
            return False

    def update_reward(self):
        if self.box1.is_closer_to_target():
            self.reward = 0.35
        elif self.box1.is_further_from_target():
            self.reward = -0.125
        elif self.is_robot_closer_to_box(self.box1):
            self.reward = 0.215
        else:
            self.reward = -0.025
       
    def step(self, dir):
        self.screen.fill('white')
        pygame.draw.rect(self.screen, (231, 255, 182), pygame.Rect(60, 60, 60, 60))
        self.all_sprites.update(dir)
        if self.current_box.attached:
            self.current_box.move_with_robot(self.robot.rect.center[0], self.robot.rect.center[1])
        else:
            self.current_box.check_if_robot_attached(self.robot.rect.center[0], self.robot.rect.center[1])
        self.all_sprites.draw(self.screen)

        self.update_reward()

        self.change_target()
        if self.box1.is_end() and self.box2.is_end():
            self.done = True
            self.reward = 20

        if self.robot.collided_with_window():
            self.reward = -2
            self.done = True

        pygame.display.update()
        self.state = [self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.current_box.rect.center[0]),
                      self.normalize(self.current_box.rect.center[1]),
                      self.normalize(90), self.normalize(90),
                      self.current_box.attached]
        return self.state, self.reward, self.done
    
pygame.init()
env = Warehouse()
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
        reward = torch.tensor([reward], device=device)
        done = terminated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        step+=1
        if(step > 2000):
            done=True
    #postepeno smanjivanje epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPS_DECAY
        epsilon = max(MIN_EPSILON, epsilon)