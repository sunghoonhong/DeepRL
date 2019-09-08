'''
Author: Sunghoon Hong
Title: snake_v2.py
Version: 0.1.1
Description: Snake 2 game Environment
Detail:
    Continuous Action Space

Update:
    add traps

'''

import os
import time
import random
import numpy as np
from gym.spaces import Box
import pygame as pg
from pygame import gfxdraw as gdraw


DENSE_REWARD = {
    'goal': 1,
    'boundary' : -1,
    'body': -1,
    'trap': -1,
    'closer': 0.01,
    'farther': -0.01
}

SPARSE_REWARD = {
    'goal': 1,
    'boundary': -1,
    'body': -1,
    'trap': -1,
    'normal': 0
}

# size
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

GRID_LEN = 20
GRID_SIZE = (GRID_LEN, GRID_LEN)
RADIUS = GRID_LEN // 2 - 1
RADIUS_SIZE = np.array([RADIUS, RADIUS])

TRAP_SIZE = (GRID_LEN * 6, GRID_LEN * 6)
TRAP_RADIUS_HIGH = GRID_LEN * 3 - 1
TRAP_RADIUS_LOW = GRID_LEN - 1
TRAP_RADIUS_DIFF = TRAP_RADIUS_HIGH - TRAP_RADIUS_LOW

# time
FPS = 30
DELAY = 0.02

# color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GREEN50 = (50, 200, 50)
RED50 = (200, 50, 50)

BG_COLOR = WHITE

SPEED = 10
SMOOTH = 0.3
EYE_RADIUS = 0.1
EYE_SIZE = (EYE_RADIUS*2, EYE_RADIUS*2)
TRAP_IMGS = [
    pg.transform.scale(pg.image.load(os.path.join('imgs/snake', 'trap_5.png')), TRAP_SIZE),
    pg.transform.scale(pg.image.load(os.path.join('imgs/snake', 'trap_4.png')), TRAP_SIZE),
    pg.transform.scale(pg.image.load(os.path.join('imgs/snake', 'trap_3.png')), TRAP_SIZE),
    pg.transform.scale(pg.image.load(os.path.join('imgs/snake', 'trap_2.png')), TRAP_SIZE),
    pg.transform.scale(pg.image.load(os.path.join('imgs/snake', 'trap_1.png')), TRAP_SIZE)
]

class Head(pg.sprite.Sprite):
    
    def __init__(self,
            init_x=WINDOW_WIDTH//2//GRID_LEN*GRID_LEN,
            init_y=WINDOW_HEIGHT//2//GRID_LEN*GRID_LEN):
        pg.sprite.Sprite.__init__(self)        
        self.eye = pg.sprite.Sprite()
        self.eye.rect = pg.Rect((0,0), EYE_SIZE)
        self.eye.radius = EYE_RADIUS

        self.rect = pg.Rect((0,0), GRID_SIZE)
        self.radius = RADIUS
        self.init_x = init_x
        self.init_y = init_y
        self.init_pos = np.array((init_x, init_y))
        self.image = pg.Surface(GRID_SIZE, pg.SRCALPHA)
        gdraw.aacircle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN50)
        gdraw.filled_circle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN50)

        self.reset()
        

    def reset(self):
        self.speed = SPEED * 0.2
        self.direction = np.array((0, 0))
        self.theta = 0.
        self.pos = self.init_pos
        self.rect.center = self.init_pos
        self.eye.rect.center = self.rect.center
        self.trace = self.rect

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update(self):
        self.move()

    def move(self):
        self.trace = self.rect
        self.rect = self.rect.move(self.speed  * self.direction)
        self.eye.rect.center = self.rect.center + self.radius * self.direction
        self.pos = self.rect.center

    def set_direction(self, direc):
        self.direction = direc

    def set_speed(self, speed):
        self.speed = SPEED * (speed + 0.2)

    def chain(self, tail):
        self.tail = tail


class Body(pg.sprite.Sprite):

    def __init__(self, head):
        pg.sprite.Sprite.__init__(self)
        self.radius = RADIUS
        self.head = head
        self.tail = None
        self.trace = None
        self.rect = pg.Rect((0, 0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE, pg.SRCALPHA)
        gdraw.aacircle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN)
        gdraw.filled_circle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), GREEN)
        self.rect.center = head.trace.center

    def move(self):
        self.trace = self.rect
        self.rect = self.head.trace

    def update(self):
        self.move()

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def chain(self, tail):
        self.tail = tail

class Snake:

    def __init__(self, head):
        self.head = head
        self.bodys = pg.sprite.Group()
        self.reset()

    def reset(self):
        self.head.reset()
        self.bodys.empty()
        self.tail = self.head
        self.life = 1
        self.len = 1

    def push_back(self):
        new_tail = Body(self.tail)
        self.tail.chain(new_tail)
        self.bodys.add(new_tail)
        self.tail = new_tail
        self.len += 1

    def pop_back(self):
        if not self.bodys:
            return
        new_tail = self.tail.head
        self.tail.kill()
        self.tail = new_tail
        self.len -= 1

    def update(self):
        self.head.update()
        self.bodys.update()

    def draw(self, screen):
        self.bodys.draw(screen)
        self.head.draw(screen)
    
    def boundary_collision(self):
        if (self.head.rect.center[1] < 0 or self.head.rect.center[1] > WINDOW_HEIGHT
            or self.head.rect.center[0] < 0 or self.head.rect.center[0] > WINDOW_WIDTH):
            return True
        return False
    
    def set_direction(self, direc):
        self.head.set_direction(direc)

    def set_speed(self, speed):
        self.head.set_speed(speed)

class Goal(pg.sprite.Sprite):

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.rect = pg.Rect((0, 0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE, pg.SRCALPHA)
        self.radius = RADIUS
        gdraw.aacircle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), RED50)
        gdraw.filled_circle(self.image, self.rect.center[0], self.rect.center[1], int(self.radius), RED50)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

class Trap(pg.sprite.Sprite):

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        
        self.rect = pg.Rect((0, 0), TRAP_SIZE)
        self.pos = None
        self.radius = TRAP_RADIUS_HIGH
        self.speed = SPEED * 0.1
        self.img_idx = 0
        self.img_dir = 1
        self.image = TRAP_IMGS[self.img_idx]
        
    def update(self, head_pos):
        self.move(head_pos)
        self.radius = (self.img_idx + (TRAP_RADIUS_LOW / TRAP_RADIUS_DIFF)) / 5 * TRAP_RADIUS_HIGH
        self.image = TRAP_IMGS[int(self.img_idx)]
        self.img_idx += 0.1 * self.img_dir
        if self.img_idx >= len(TRAP_IMGS):
            self.img_idx = len(TRAP_IMGS) - 1
            self.img_dir *= -1
        elif self.img_idx <= 0:
            self.img_idx = 0
            self.img_dir *= -1

    def move(self, head_pos):
        head_pos = np.array(head_pos, dtype=np.float64)
        heading = normalize(head_pos - self.pos)
        noise = np.random.normal(0, 0.5, size=2).astype(np.float64)
        self.pos += (heading + noise) * self.speed
        self.rect.center = self.pos
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)

        
class Game:

    def __init__(self):
        self.snake = Snake(Head())
        self.goals = pg.sprite.Group()
        self.traps = pg.sprite.Group()
        self.screen = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display = False

    def reset(self):
        self.snake.reset()
        self.goals.empty()
        self.traps.empty()
        self.create_goal()
        self.create_trap()
        self.score = 0
        self.direc = np.array([0., 0.])
        self.speed = 0.1

    def create_goal(self):
        if not self.goals:
            goal = Goal()
            while True:
                goal.rect.topleft = (
                    random.randrange(0, WINDOW_WIDTH-GRID_LEN, GRID_LEN),
                    random.randrange(0, WINDOW_HEIGHT-GRID_LEN, GRID_LEN)
                )
                if goal.rect.topleft != self.snake.head.rect.topleft:
                    break
            self.goals.add(goal)
        
    def create_trap(self):
        trap = Trap()
        while True:
            trap.rect.topleft = random.sample(
                [(0, random.randrange(0, WINDOW_HEIGHT - GRID_LEN*2, GRID_LEN*2)),
                (WINDOW_WIDTH - GRID_LEN * 2, random.randrange(0, WINDOW_HEIGHT - GRID_LEN*2, GRID_LEN*2)),
                (random.randrange(0, WINDOW_WIDTH - GRID_LEN*2, GRID_LEN*2), 0),
                (random.randrange(0, WINDOW_WIDTH - GRID_LEN*2, GRID_LEN*2), WINDOW_HEIGHT - GRID_LEN*2)], 1
            )[0]
            if trap.rect.topleft != self.snake.head.rect.topleft:
                break
        trap.pos = np.array(trap.rect.topleft, dtype=np.float64)
        self.traps.add(trap)


    def collision(self):
        '''
        return:
            info
        '''
        info = 'normal'
        if pg.sprite.spritecollide(self.snake.head, self.traps, False, pg.sprite.collide_circle):
            self.snake.life -= 1
            info = 'trap'
        elif pg.sprite.spritecollide(self.snake.head.eye, self.snake.bodys, False, pg.sprite.collide_circle):
            self.snake.life -= 1
            info = 'body'
        if self.snake.boundary_collision():
            self.snake.life -= 1
            info = 'boundary'
        elif pg.sprite.spritecollide(self.snake.head, self.goals, True, pg.sprite.collide_circle):
            self.score += 1
            info = 'goal'
            self.snake.push_back()
            
        return info
        
    def update(self):
        '''
        return:
            info
        '''
        self.snake.set_direction(self.direc)
        self.snake.set_speed(self.speed)
        self.snake.update()
        self.traps.update(self.snake.head.pos)
        info = self.collision()
        self.create_goal()
            
        return info

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.goals.draw(self.screen)
        self.snake.draw(self.screen)
        self.traps.draw(self.screen)
    
    def input(self, speed, delta_theta):
        delta_theta *= np.pi
        temp_theta = self.snake.head.theta
        theta = temp_theta + delta_theta
        direction = np.array([np.cos(theta), np.sin(theta)])
        self.direc += SMOOTH * (direction - self.direc)
        self.direc = normalize(self.direc)
        self.snake.head.theta = theta
        self.speed = speed

    def init_render(self):
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption('Snake v2.0.0')
        self.display = True

    def render(self):
        if not self.display:
            self.init_render()
        pg.display.flip()


class Env:
    def __init__(self, sparse_reward=True, use_feature=False):
        pg.init()
        self.action_space = Box(low=np.array([0.1, -1.]), high=np.array([1., 1.]), dtype='float32')
        self.game = Game()
        if not sparse_reward:
            raise NotImplementedError
        self.scheme = SPARSE_REWARD if sparse_reward else DENSE_REWARD
        self.use_feature = use_feature
        if self.use_feature:
            raise NotImplementedError
        self.observation_size = [4] if use_feature else [400, 400, 3]

    def reset(self):
        self.game.reset()
        self.game.draw()
        if self.use_feature:
            raise NotImplementedError
        else:
            observe = pg.surfarray.array3d(self.game.screen)
        return observe

    def step(self, action):
        if self.game.display:
            pg.event.pump()
        self.game.input(action[0], action[1])
        info = self.game.update()
        done = (self.game.snake.life <= 0)
        reward = self.scheme[info]
            
        self.game.draw()
        if self.use_feature:
            raise NotImplementedError
        else:
            observe = pg.surfarray.array3d(self.game.screen)
        return observe, reward, done, info

    def init_render(self):
        self.game.init_render()

    def render(self):
        if not self.game.display:
            self.init_render()
        pg.display.flip()

    def snapshot(self):
        pg.image.save(self.game.screen, 'snapshots/'+str(int(time.time()*10000))+'.png')


def normalize(vec):
    return vec / np.linalg.norm(vec)

if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (64, 64)
    clock = pg.time.Clock()
    render = True
    game = Game()
    game.reset()
    if render:
        game.init_render()
    step = 0
    while game.snake.life > 0:
        time.sleep(DELAY)
        step += 1
        clock.tick(FPS)
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                quit()
            elif evt.type == pg.KEYDOWN:
                if evt.key == pg.K_ESCAPE:
                    quit()
        pos = np.array(pg.mouse.get_pos())
        temp_theta = game.snake.head.theta
        head = game.snake.head.rect.center
        heading = (pos - head)
        speed = np.linalg.norm(heading)
        speed = np.clip(speed, 0, (WINDOW_WIDTH // 4))
        speed /= WINDOW_WIDTH // 4
        
        if heading[0]==0 and heading[1]==0:
            continue
        pos = normalize(heading)
        theta = np.arctan2(pos[1], pos[0])

        delta_theta = theta - temp_theta
        delta_theta = np.clip(delta_theta, -np.pi, np.pi)
        delta_theta /= np.pi
        # print(delta_theta)
        game.input(speed, delta_theta)
        game.update()
        game.draw()
        if render:
            game.render()            

    print('Score:', game.score, 'Step:', step)
