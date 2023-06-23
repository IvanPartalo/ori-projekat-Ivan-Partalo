import pygame

class Robot(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)
        self.image = pygame.Surface((10, 10))
        self.image.fill((255,0,0))
        self.rect = self.image.get_rect(topleft = (90, 130))
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2()
        self.speed = 0.2
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
        self.done = False
        self.reward = 0
        self.state = [0, 0, 0, 0, 0, 0, 0]

    def reset(self):
        self.robot.rect = self.robot.image.get_rect(topleft = (90, 130))
        self.robot.pos = pygame.math.Vector2(self.robot.rect.topleft)
        self.done = False
        self.reward = 0
        self.state = [self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.box1.rect.center[0]), self.normalize(self.box1.rect.center[1]),
                      self.normalize(90), self.normalize(90),
                      self.box1.attached]

    def normalize(self, x):
        return (2 * x / 200) - 1

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
        self.all_sprites.draw(self.screen)
        for box in self.boxes:
            if box.attached:
                box.move_with_robot(self.robot.rect.center[0], self.robot.rect.center[1])
            else:
                box.check_if_robot_attached(self.robot.rect.center[0], self.robot.rect.center[1])
        self.update_reward()
        if self.robot.collided_with_window():
            self.done = True
        pygame.display.update()
        self.state = [self.normalize(self.robot.rect.center[0]), self.normalize(self.robot.rect.center[1]),
                      self.normalize(self.box1.rect.center[0]), self.normalize(self.box1.rect.center[1]),
                      self.normalize(90), self.normalize(90),
                      self.box1.attached]
        return self.done
    
pygame.init()
env = Warehouse()
done = env.step(3)
while not done:
    pygame.event.get()
    keys=pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
       done = env.step(1)
    if keys[pygame.K_UP]:
        done = env.step(0)
    if keys[pygame.K_LEFT]:
        done = env.step(2)
    if keys[pygame.K_RIGHT]:
        done = env.step(3)