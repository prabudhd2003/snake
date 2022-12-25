import torch
import random
import numpy as np
from collections import deque
from snakeAI import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        self.no_games = 0
        self.epsilon = 0  # parameter to control the randomness
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # if we excced this memory it will remove elements from the left i.e. call popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # model, trainer

    def get_state(self, snakeAI):
        head = snakeAI.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = snakeAI.direction == Direction.LEFT
        dir_r = snakeAI.direction == Direction.RIGHT
        dir_u = snakeAI.direction == Direction.UP
        dir_d = snakeAI.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snakeAI.is_collision(point_r)) or 
            (dir_l and snakeAI.is_collision(point_l)) or 
            (dir_u and snakeAI.is_collision(point_u)) or 
            (dir_d and snakeAI.is_collision(point_d)),

            # Danger right
            (dir_u and snakeAI.is_collision(point_r)) or 
            (dir_d and snakeAI.is_collision(point_l)) or 
            (dir_l and snakeAI.is_collision(point_u)) or 
            (dir_r and snakeAI.is_collision(point_d)),

            # Danger left
            (dir_d and snakeAI.is_collision(point_r)) or 
            (dir_u and snakeAI.is_collision(point_l)) or 
            (dir_r and snakeAI.is_collision(point_u)) or 
            (dir_l and snakeAI.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            snakeAI.food.x < snakeAI.head.x,  # food left
            snakeAI.food.x > snakeAI.head.x,  # food right
            snakeAI.food.y < snakeAI.head.y,  # food up
            snakeAI.food.y > snakeAI.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self, state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # intial random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.no_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:   # smaller the epsilion less this will happen
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # model based move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []  # used for plotting
    plot_mean_scores = []
    total_score = 0
    record = 0  # best score
    agent = Agent()
    game = SnakeGameAI()
    
    while True:    # run forever until we quit the script
        # get the old/current state
        state_old = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory, plot the result
            game.reset()
            agent.no_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.no_games, "Score: ", score, "Record: ", record)

            # plot 
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.no_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
