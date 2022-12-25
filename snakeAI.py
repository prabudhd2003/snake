import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision

class Direction(Enum):   # so that we are sure we use the correct direction
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 40

font = pygame.font.Font('arial.ttf', 25) 
#font = pygame.font.SysFont('arial', 25) 

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, width=640, height=480):
        
        self.width = width
        self.height = height
        
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game made by Prabudhd K Kandpal')
        self.clock = pygame.time.Clock()  # to control the speed of the game
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT   # intial direction is moving to the right 

        self.head = Point(self.width/2, self.height/2)    # intial postion of the snake 
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y),
                                 Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):   # this will randomly  place the food
        x = random.randint(0, (self.width-BLOCK_SIZE) //BLOCK_SIZE)*BLOCK_SIZE  # will generate random positions on the screen 
        y = random.randint(0, (self.height-BLOCK_SIZE) //BLOCK_SIZE)*BLOCK_SIZE # that are multiples of this block size
        self.food = Point(x, y)
        if self.food in self.snake:  # to insure snake and food dont overlap 
            self._place_food()
    
    def play_step(self, action):
        # collect the user input
        self.frame_iteration += 1
        self.frame_iteration += 1
        for event in pygame.event.get():    # this gets all the user events that happen inside one play_step
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()        

        # move our snake
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # check if game over
        reward = 0
        game_over = False
        if self.is_collison() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # place new food 
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # update the pygame ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game over and score
        return reward, game_over, self.score

    def is_collison(self, pt=None):
        if pt is None:
            pt = self.head
        # hits the boundary
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
           return True
        # hits its self
        if pt in self.snake[1:]:  # we are doing slicing because head is alwayas in the snake 
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score:" + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0]) 
        pygame.display.flip()
    
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)   # current direction

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)
