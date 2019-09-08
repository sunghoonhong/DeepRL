'''
Author: Sunghoon Hong
Title: snake_v1.py
Version: 0.0.1
Description: Snake game Environment
Fix:
    3 action (straight / right / left)
'''

import os
import time
import random
import numpy as np
import pygame as pg
import gym


DENSE_REWARD = {
    'goal': 1,
    'boundary' : -1,
    'body': -1,
    'closer': 0.01,
    'farther': -0.01
}

SPARSE_REWARD = {
    'goal': 1,
    'boundary': -1,
    'body': -1,
    'closer': 0,
    'farther': 0
}

# size
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

GRID_LEN = 20
GRID_SIZE = (GRID_LEN, GRID_LEN)

GRID_WIDTH = WINDOW_WIDTH // GRID_LEN
GRID_HEIGHT = WINDOW_HEIGHT // GRID_LEN

# time
FPS = 20  # This variable will define how many frames we update per second.
DELAY = 0.1

# color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN50 = (50, 200, 50)
RED50 = (200, 50, 50)

BG_COLOR = WHITE

# action
STAY = 0
LEFT = 1
RIGHT = 2
ACTION = ['STAY', 'LEFT', 'RIGHT']

class Head(pg.sprite.Sprite):
    
    def __init__(self,
            init_x=WINDOW_WIDTH//2//GRID_LEN*GRID_LEN,
            init_y=WINDOW_HEIGHT//2//GRID_LEN*GRID_LEN,
            init_direction=None):
        pg.sprite.Sprite.__init__(self)
        self.speed = GRID_LEN
        self.direction = None
        self.rect = pg.Rect((0,0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(GREEN50)
        self.init_pos = np.array([init_x, init_y])
        self.rect.topleft = self.init_pos
        self.trace = self.rect

    def reset(self):
        self.speed = GRID_LEN
        self.direction = None
        self.rect.topleft = self.init_pos
        self.trace = self.rect

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update(self):
        self.move()

    def move(self):
        self.trace = self.rect
        if self.direction is not None:
            self.rect = self.rect.move(self.direction * self.speed)

    def set_direction(self, dir):
        if self.direction is None:
            if dir == STAY:
                self.direction = np.array([0, -1])
            elif dir == LEFT:
                self.direction = np.array([-1, 0])
            elif dir == RIGHT:
                self.direction = np.array([1, 0])
        else:                
            if dir == LEFT:
                self.direction = np.array([self.direction[1], -self.direction[0]])
            elif dir == RIGHT:
                self.direction = np.array([-self.direction[1], self.direction[0]])
            
    def chain(self, tail):
        self.tail = tail


class Body(pg.sprite.Sprite):

    def __init__(self, head):
        pg.sprite.Sprite.__init__(self)
        self.speed = GRID_LEN
        self.head = head
        self.tail = None
        self.trace = None
        self.rect = pg.Rect(head.trace.topleft, GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(GREEN50)

    def move(self):
        self.trace = self.rect
        self.rect = self.head.trace

    def update(self):
        self.move()

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def chain(self, tail):
        self.tail = tail

class Snake():

    def __init__(self, head):
        self.head = head
        self.bodys = pg.sprite.Group()
        self.tail = self.head
        self.life = 1
        self.len = 1

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
        # self.life -= 4 / float(GRID_WIDTH * GRID_HEIGHT)

    def draw(self, screen):
        self.bodys.draw(screen)
        self.head.draw(screen)
    
    def boundary_collision(self):
        if (self.head.rect.top < 0 or self.head.rect.bottom > WINDOW_HEIGHT
            or self.head.rect.left < 0 or self.head.rect.right > WINDOW_WIDTH):
            return True
        return False
    
    def set_direction(self, key):
        self.head.set_direction(key)


class Goal(pg.sprite.Sprite):

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.rect = pg.Rect((0, 0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(RED50)
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)


class Game:

    def __init__(self):
        self.snake = Snake(Head())
        self.goals = pg.sprite.Group()
        self.screen = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display = False

    def reset(self):
        self.snake.reset()
        self.goals.empty()
        self.create_goal()
        self.distance = self.calculate_distance()
        self.key = None
        self.score = 0

    def calculate_distance(self):
        head = self.snake.head.rect.topleft
        goal = self.goals.sprites()[0].rect.topleft
        return abs(head[0]-goal[0]) + abs(head[1]-goal[1])
    
    def collision_distance(self):
        '''
            Calculate distances between head and obstacle in 3 possible directions
            return:
                (dist_straight, dist_left, dist_right)
        '''
        head = self.snake.head
        head_pos = head.rect.topleft
        
        distances = {
            -2 : head_pos[0],   # west
            -1 : head_pos[1],   # north
            1 : WINDOW_HEIGHT - head_pos[1] - GRID_LEN,    # south
            2 : WINDOW_WIDTH - head_pos[0] - GRID_LEN    # east
        }

        if head.direction is None:
            return distances[-1], distances[-2], distances[2]
        
        straight_dir = head.direction
        left_dir = np.array([straight_dir[1], -straight_dir[0]])
        right_dir = np.array([-straight_dir[1], straight_dir[0]])

        straight_dist = distances[self.dir_to_idx(straight_dir)]
        left_dist = distances[self.dir_to_idx(left_dir)]
        right_dist = distances[self.dir_to_idx(right_dir)]

        for body in self.snake.bodys.sprites():
            pos = body.rect.topleft
            if pos[0] == head_pos[0]:
                temp_dist = abs(pos[1] - head_pos[1]) - GRID_LEN
                if straight_dir[0] == 0:
                    if straight_dir[1] * (pos[1] - head_pos[1]) > 0:
                        if temp_dist < straight_dist:
                            straight_dist = temp_dist
                else:
                    if straight_dir[0] * (pos[1] - head_pos[1]) > 0:
                        if temp_dist < right_dist:
                            right_dist = temp_dist
                    else:
                        if temp_dist < left_dist:
                            left_dist = temp_dist
            elif pos[1] == head_pos[1]:
                temp_dist = abs(pos[0] - head_pos[0]) - GRID_LEN
                if straight_dir[0] == 0:
                    if straight_dir[1] * (pos[0] - head_pos[0]) > 0:
                        if temp_dist < left_dist:
                            left_dist = temp_dist
                    else:
                        if temp_dist < right_dist:
                            right_dist = temp_dist
                else:
                    if straight_dir[0] * (pos[0] - head_pos[0]) > 0:
                        if temp_dist < straight_dist:
                            straight_dist = temp_dist
        
        return straight_dist, left_dist, right_dist

    def dir_to_idx(self, dir):
        return dir[0] * 2 + dir[1]

    def create_goal(self):
        if not self.goals:
            goal = Goal()
            avail = [(i*GRID_LEN, j*GRID_LEN) for i in range(GRID_WIDTH) for j in range(GRID_HEIGHT)]
            exists = []
            exists.append(self.snake.head.rect.topleft)
            for body in self.snake.bodys.sprites():
                exists.append(body.rect.topleft)
            avail = list(set(avail) - set(exists))
            goal.rect.topleft = random.choice(avail)
            self.goals.add(goal)
        
    def collision(self):
        '''
        return:
            info
        '''
        info = 'ok'
        body_collision = pg.sprite.spritecollide(self.snake.head, self.snake.bodys, False)
        if body_collision:
            self.snake.life -= 1
            info = 'body'
        elif self.snake.boundary_collision():
            self.snake.life -= 1
            info = 'boundary'
        elif pg.sprite.spritecollide(self.snake.head, self.goals, True):
            # self.snake.life = 1
            self.score += 1
            info = 'goal'
            self.snake.push_back()
            
        return info
        
    def update(self):
        '''
        return:
            info
        '''
        if self.key is not None:
            self.snake.set_direction(self.key)
            self.key = None
        self.snake.update()
        info = self.collision()
        self.create_goal()
        if info == 'ok':
            dist = self.calculate_distance()
            info = 'closer' if dist < self.distance else 'farther'
            self.distance = dist
            
        return info

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.goals.draw(self.screen)
        self.snake.draw(self.screen)
    
    def keydown(self, key):
        self.key = key

    def init_render(self):
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption('Snake v1.0.0')
        self.display = True
        self.draw()

    def render(self):
        if not self.display:
            self.init_render()
        pg.display.flip()


class Env:
    def __init__(self, **kwargs):
        '''
        params:
            sparse_reward (bool) default=True
            use_feature (bool) default=False
            difficulty (int) default=0
        '''
        
        self.difficulty = kwargs.get('difficulty', 0)
        self.sparse_reward = kwargs.get('sparse_reward', True)
        self.use_feature = kwargs.get('use_feature', False)

        self.action_space = gym.spaces.Discrete(3)
        self.game = Game()
        self.scheme = SPARSE_REWARD if self.sparse_reward else DENSE_REWARD
        self.observation_size = [8] if self.use_feature else [400, 400, 3]

    def reset(self):
        '''
        observe: 
            - feature: shape=(8,) [direction vector(2d), head-goal vector(2d), straight_dist, left_dist, right_dist, snake_len]
            - rgb: WINDOW_WIDTH * WINDOW_HEIGHT * 3, rgb_array
        '''
        self.game.reset()
        self.game.draw()
        head_pos = self.game.snake.head.rect.topleft
        goal_pos = self.game.goals.sprites()[0].rect.topleft
        dist0, dist1, dist2 = self.game.collision_distance()
        observe = np.array([0, 0,
                        head_pos[0] - goal_pos[0], head_pos[1] - goal_pos[1],
                        dist0, dist1, dist2, self.game.snake.len * GRID_LEN]) / GRID_LEN
        if self.use_feature:
            observe = observe.reshape(self.observation_size)
        else:
            observe = pg.surfarray.array3d(self.game.screen)
        return observe

    def step(self, action):
        if self.game.display:
            pg.event.pump()
        self.game.keydown(action)
        info = self.game.update()
        done = (self.game.snake.life <= 0)
        reward = self.scheme[info]
            
        self.game.draw()
        
        if self.use_feature:
            # define state by feature extraction
            head_pos = self.game.snake.head.rect.topleft
            goal_pos = self.game.goals.sprites()[0].rect.topleft
            dist0, dist1, dist2 = self.game.collision_distance()
            direc = self.game.snake.head.direction * GRID_LEN * GRID_LEN
            observe = np.array([direc[0], direc[1], head_pos[0] - goal_pos[0], head_pos[1] - goal_pos[1],
                            dist0, dist1, dist2, self.game.snake.len * GRID_LEN]) / GRID_LEN
            observe = observe.reshape(self.observation_size)
        else:
            observe = pg.surfarray.array3d(self.game.screen)
        return observe, reward, done, info

    def render(self):
        self.game.render()

    def snapshot(self):
        pg.image.save(self.game.screen, 'snapshots/'+str(int(time.time()*10000))+'.png')

if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (64, 64)
    clock = pg.time.Clock()
    render = True
    pg.init()
    game = Game()
    game.reset()
    step = 0
    while game.snake.life > 0:
        time.sleep(DELAY)
        step+=1
        clock.tick(FPS)
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                quit()
            elif evt.type == pg.KEYDOWN:
                if evt.key == pg.K_UP:
                    game.keydown(STAY)
                elif evt.key == pg.K_LEFT:
                    game.keydown(LEFT)
                elif evt.key == pg.K_RIGHT:
                    game.keydown(RIGHT)
                elif evt.key == pg.K_ESCAPE:
                    quit()

        print(game.update())
        game.draw()
        if render:
            game.render()            

    print('Score:', game.score, 'Step:', step)
