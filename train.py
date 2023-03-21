import random
from collections import deque
import numpy as np
import torch
from dataclasses import dataclass
from env import _2048Env
from mcts import MCTS_Evaluator
import time

class ReplayMemory:
    def __init__(self, max_size=10000) -> None:
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)
        pass

    def sample(self, num_samples): 
        return random.sample(self.memory, num_samples)
    
    def insert(self, sample):
        self.memory.append(sample)

    def extend(self, samples):
        self.memory.extend(samples)

    def size(self):
        return len(self.memory)


@dataclass()
class MCTS_HYPERPARAMETERS:
    lr: float = 5e-4
    weight_decay: float = 1e-2
    minibatch_size: int = 1000
    replay_memory_size: int = 30000 
    num_mcts_train_evals: int = 500
    num_mcts_test_evals: int = 1000
    num_episodes: int = 1000
    checkpoint_every: int = 100

import matplotlib.pyplot as plt
import IPython.display as display

class MetricsHistory:
    def __init__(self) -> None:
        self.rewards = []
        self.game_moves = []
        self.prob_losses = []
        self.value_losses = []
        self.total_losses = []
        self.high_squares = []
        self.episodes = 0
        self.best_result = float('-inf')
    
    def add_history(self, info):
        self.rewards.append(info['reward'])
        self.game_moves.append(info['game_moves'])
        self.prob_losses.append(info['prob_loss'])
        self.value_losses.append(info['value_loss'])
        self.total_losses.append(info['total_loss'])
        self.high_squares.append(info['high_square'])
        self.episodes += 1
        if info['reward'] > self.best_result:
            self.best_result = info['reward']
            return True
        return False
    
    def plot_history(self, plots = None):
        if plots is None:
            plots = ['rewards', 'game_moves', 'prob_losses', 'value_losses', 'total_losses', 'high_squares']
        display.clear_output(wait=True)
        # this could probably be implemented a tad more elegantly
        for pl in plots:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if pl == 'rewards':
                ax.plot(self.rewards)
                ax.set_title('Rewards (log2)')
            elif pl == 'game_moves':
                ax.plot(self.game_moves)
                ax.set_title('Game Moves')
            elif pl == 'prob_losses':
                ax.plot(self.prob_losses)
                ax.set_title('Prob. Loss')
            elif pl == 'value_losses':
                ax.plot(self.value_losses)
                ax.set_title('Value Loss')
            elif pl == 'total_losses':
                ax.plot(self.total_losses)
                ax.set_title('Total Loss')
            elif pl == 'high_squares':
                ax.plot(self.high_squares)
                ax.set_title('High Square (log2)')
            else:
                print(f'Unknown metric: {pl}')
            display.display(fig)
        

def load_from_checkpoint(filename, model_class, load_replay_memory=True):
    run_tag = filename.split('_')[0]
    checkpoint = torch.load(filename)
    hyperparameters = checkpoint['hyperparameters']
    episode = checkpoint['episode']
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.lr, weight_decay=hyperparameters.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    env = _2048Env(keep_history=True)
    mcts = MCTS_Evaluator(model, env, training=True)
    metrics_history = checkpoint['metrics_history']
    memory = None
    if load_replay_memory:
        memory = checkpoint.get('replay_memory')
    elif memory is None:
        memory = ReplayMemory(hyperparameters.replay_memory_size)
    
    return env, mcts, episode, model, optimizer, hyperparameters, metrics_history, memory, run_tag
    

def save_checkpoint(episodes, model, optimizer, hyperparameters, metrics_history, replay_memory, run_tag='', save_replay_memory=True):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': hyperparameters,
        'metrics_history': metrics_history,
        'replay_memory': replay_memory if save_replay_memory else None,
        'episode': episodes,
        'model_type': str(type(model))
    }, f'{run_tag}_ep{episodes}.pt')



def train(samples, model, optimizer, tensor_conversion_fn, c_prob=5):
    model.train()
    obs, mcts_probs, rewards = zip(*samples)
    obs = tensor_conversion_fn(obs)
    mcts_probs = torch.from_numpy(np.array(mcts_probs))
    rewards = torch.from_numpy(np.array(rewards)).unsqueeze(1).float()

    optimizer.zero_grad()

    exp_probs, exp_rewards = model(obs)
    value_loss = torch.mean((rewards - exp_rewards) ** 2)
    prob_loss = -1 * c_prob * torch.mean(torch.sum(mcts_probs*torch.log(exp_probs), dim=1))

    loss = value_loss + prob_loss
    loss.backward()
    optimizer.step()

    return value_loss.item(), prob_loss.item(), loss.item()


def train_from_episode(cur_episode, model, optimizer, env, mcts, replay_memory, metrics_history: MetricsHistory, hyperparameters: MCTS_HYPERPARAMETERS, run_tag='', save_replay_memory=True):
    best_result = max(metrics_history.rewards)
    while cur_episode < hyperparameters.num_episodes:
        directional_moves = {0:0, 1:0, 2:0, 3:0}
        with torch.no_grad():
            env.reset()
            mcts.reset()
            mcts.clear_cache()
            training_examples = []
            moves = 0
            while True:
                # get inputs, reward, mcts probs, run n_iterations of MCTS
                terminated, inputs, reward, mcts_probs, move = mcts.choose_progression(hyperparameters.num_mcts_train_evals)
                directional_moves[move] += 1
                moves += 1
                training_examples.append([inputs, mcts_probs])
                if terminated:
                    break
                
            for example in training_examples:
                example.append(reward)
                rotated_examples = rotate_examples(example)
                for r_example in rotated_examples:
                    replay_memory.insert(r_example)

        right_percent, up_percent, left_percent, down_percent = directional_moves[0]/moves, directional_moves[1]/moves, directional_moves[2]/moves, directional_moves[3]/moves

        if len(replay_memory.memory) > hyperparameters.minibatch_size:
            value_loss, prob_loss, total_loss = train(replay_memory.sample(hyperparameters.minibatch_size), model, optimizer)
            metrics_history.add_history({
                'reward': reward,
                'game_moves': moves,
                'prob_loss': prob_loss,
                'value_loss': value_loss,
                'total_loss': total_loss,
                'high_square': env.get_highest_square()
            })
            metrics_history.plot_history()

        print(f'[EPISODE {cur_episode}] Total Loss: {total_loss}, Prob Loss {prob_loss}, Value Loss {value_loss}, Reward {reward}, Moves: {moves}, Highest Square: {env.get_highest_square()} \
                                        Left% {left_percent}, Right% {right_percent}, Up% {up_percent}, Down% {down_percent}')
        if reward > best_result:
            print('*** NEW BEST REWARD ***')
            print(f'prev: {best_result}, new: {reward}')
            best_result = reward

        if cur_episode % hyperparameters.checkpoint_every == 0:
            save_checkpoint(cur_episode + 1, model, optimizer, hyperparameters, metrics_history, replay_memory, run_tag, save_replay_memory)
        
        cur_episode += 1

MOVE_MAP = {0: 'right', 1: 'up', 2: 'left', 3: 'down'}
def test_network(model, hyperparameters, tensor_conversion_fn, debug_print=False):
    env = _2048Env(keep_history=True)
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn, training=False)
    env.reset()
    model.eval()
    with torch.no_grad():
        moves = 0
        while True:
            start_time = time.time()
            probs, value = model(tensor_conversion_fn([env.board]))
            if debug_print:
                print(env.board)
            terminated, _, reward, mcts_probs, move = mcts.choose_progression(hyperparameters.num_mcts_test_evals)
            moves += 1
            if debug_print:
                print(f'Time elapsed: {time.time() - start_time}s')
                print(f'Move #: {moves}')
                print(f'Move: {MOVE_MAP[move]}')
                print(f'Network Probs: {probs.detach().numpy()}')
                print(f'MCTS Probs: {mcts_probs}')
                print(f'Network value: {value.item()}')
                print(f'Q Value: {mcts.puct_node.w / mcts.puct_node.n}')
            
            
            if terminated:
                print(f'Terminated, final reward = {reward}')
                break

def collect_episode(model, hyperparameters, tensor_conversion_fn):
    training_examples = []
    env = _2048Env(keep_history=True)
    env.reset()
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn=tensor_conversion_fn, training=True)
    moves = 0
    with torch.no_grad():
        while True:
            # get inputs, reward, mcts probs, run n_iterations of MCTS
            terminated, inputs, reward, mcts_probs, move = mcts.choose_progression(hyperparameters.num_mcts_train_evals)
            moves += 1
            training_examples.append([inputs, mcts_probs])
            if terminated:
                break

        for example in training_examples:
            example.append(reward)


    return training_examples, reward, moves, env.get_highest_square()

def rotate_training_examples(training_examples):
    inputs, probs, rewards = zip(*training_examples)
    rotated_inputs = []
    for i in inputs:
        for k in range(4):
            rotated_inputs.append(np.rot90(i, k=k))
    rotated_probs = []
    for p in probs:
        # left -> down
        # down -> right
        # right -> up
        # up -> left
        for k in range(4):
            rotated_probs.append(np.roll(p, k))
    rotated_rewards = []
    for _ in range(4):
        rotated_rewards.extend(rewards)
    
    return zip(rotated_inputs, rotated_probs, rotated_rewards)