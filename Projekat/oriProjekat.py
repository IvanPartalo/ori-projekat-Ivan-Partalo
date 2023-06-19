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

class warehouse():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((200,200), pygame.SCALED)
        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()
        self.robot = Robot([self.all_sprites, self.collision_sprites])
        self.done = False
        self.reward = 0

    def reset(self):
        self.robot.rect = self.robot.image.get_rect(topleft = (90, 130))
        self.robot.pos = pygame.math.Vector2(self.robot.rect.topleft)
        
        self.done = False
        self.reward = 0
            
    def step(self, dir):
        self.screen.fill('white')
        self.all_sprites.update(dir)
        self.all_sprites.draw(self.screen)
        
        if self.robot.collided_with_window():
            self.done = True
        pygame.display.update()
        return self.done
    
pygame.init()
env = warehouse()
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