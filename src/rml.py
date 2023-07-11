import os
import datetime
import copy
import torch
from torch.nn import Module
import tqdm
import numpy as np 
import gym
from gym import Env
from gym.spaces import Box
from gym .utils.env_checker import check_env
from typing import (
    OrderedDict,
    List,
    Tuple,
    Callable,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env


class Layer:
    '''Metadata object to hold layer and track some useful information about the layer.
    '''
    def __init__(self, 
                 layer: torch.nn.Linear):
        self.layer = layer
        self.used = False
        self.times_used = 0

class LayerPool:
    '''Pool of dense layers, each with a specified number of nodes to be composed by the RL agent.
    '''
    def __init__(self, 
                 size: int, 
                 input_dim: int,
                 output_dim: int,
                 num_nodes_per_layer: int=5):
        self.size = size
        self.num_nodes_per_layer = num_nodes_per_layer

        # these layers comprise the layer pool
        # each layer has a metadata object associated with it that stores useful information
        # each layer that is used gets updated (i.e., their parameters change and the copy in 
        # this layer pool is updated)
        # the one exception to this update rule is the first and last layers in the network,
        # or 'initial' and 'final', below
        self.layers = { 
                       id : Layer(layer) 
                       for id, layer in enumerate([torch.nn.Linear(in_features=num_nodes_per_layer, out_features=num_nodes_per_layer) for _ in range(size)])
                       }
        [torch.nn.init.xavier_uniform_(layer_meta_obj.layer.weight) for layer_meta_obj in self.layers.values()]

        # these 'initial' and 'final' layers are the first layers in each target network
        # they are also uniquely updated for each target network (i.e., they aren't included in the self.layers pool
        # for other future target networks to have access to)
        # we don't have meta objects for these layers because they are used by every target network
        self.initial_layer = torch.nn.Linear(input_dim, self.num_nodes_per_layer)
        self.final_layer = torch.nn.Linear(self.num_nodes_per_layer, output_dim)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)
        torch.nn.init.xavier_uniform_(self.initial_layer.weight)


class TargetNetwork(gym.Env, Module):
    # TODO(ms): ensure that rml can't choose the initial or final layer (i.e., these are not part of action space)
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 layer_pool: LayerPool,
                 depth: int=5,
                 num_nodes: int=32):
        super(TargetNetwork, self).__init__()
        self.observation_space = Box(low=float('-inf'), high=float('inf'), shape=(num_nodes,)) # vector with latent space of network
        self.action_space = Box(low=0., high=1., shape=(depth,)) # vector of probabilities for each layer in pool

        # data 
        self.X = X
        self.y = y
        self.curr_i = 0
        self.curr_x = torch.Tensor( [X[0]] ) # these parameterize the parameterized state (i.e., layer has parameters, and x passed through has value/parameter)
        self.curr_y = torch.Tensor( [y[0]] ) # these parameterize the parameterized state (i.e., layer has parameters, and x passed through has value/parameter)
        self.layer_pool = layer_pool

        # nn module
        # https://discuss.pytorch.org/t/how-to-add-a-layer-to-an-existing-neural-network/30129/3
        self.depth = depth
        self.fcs = torch.nn.ModuleList([self.layer_pool.initial_layer, self.layer_pool.final_layer]) # every target network begins with the same initial and final layers
        self.state = self.reset() 
        self.layerpool_indices = [] # doesn't include the indices of 'initial' and 'final' layers because they don't exist
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.fcs.parameters(), lr=0.001)

    def run_inner_loop(self) -> Tuple[np.ndarray, float]: 
        '''
        Performs inference using "inner loop" network that we are composing. Returns next state and its reward.

        NOTE Design choice whether to train with new layer first. Currently training first.
        '''
        self.train()
        self.curr_i  = (self.curr_i + 1) % len(self.X) # go through data multiple times / dataset does not dictate truncation condition
        self.curr_x = torch.Tensor( [self.X[self.curr_i]] ) 
        self.curr_y = torch.Tensor( [self.y[self.curr_i]] )

        def predict(x):
            for i in range(len(self.fcs) - 1): # -1 because we don't want the last output layer yet
                x = torch.nn.functional.relu(self.fcs[i](x))
            s_prime = x
            y_hat = self.fcs[-1](x)
        
            return self.loss_fn(self.curr_y, y_hat), s_prime, self.loss_fn(self.curr_y, y_hat)

        # update params
        loss, s_prime, reward = predict(self.curr_x)
        self.opt = torch.optim.Adam(self.fcs.parameters(), lr=0.001) # re-create because parameters added
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # push updated params to pool (NOTE  handled by the rml class going forward.)
        for i in range(len(self.layerpool_indices)):
            pool_index = self.layerpool_indices[i]
            updated_layer_copy = self.fcs[i+1]
            self.layer_pool.layers[pool_index].layer = updated_layer_copy
            self.layer_pool.layers[pool_index].used = True
            self.layer_pool.layers[pool_index].times_used += 1
        
        # return updated state and reward
        _, s_prime, reward = predict(self.curr_x)

        return s_prime.detach().numpy().T.squeeze(), float(reward.detach().numpy())

    def reset(self) -> np.ndarray:
        # BUG  Stablebaselines3 seems to call the reset() method even when we're not terminating.
        # self.fcs = torch.nn.ModuleList([self.layer_pool.initial_layer, self.layer_pool.final_layer])

        initial_x = torch.Tensor( [self.X[0]] )
        initial_state = torch.nn.functional.relu(self.fcs[0](initial_x))

        return initial_state.detach().numpy().T.squeeze() # TODO(ms): is there a better way to massage this data?
    
    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, int, bool, dict]: # "given some new layer"
        # action is probability for each layer in pool
        if len(self.fcs) < self.depth:
            pool_index = np.argmax(action)
            self.layerpool_indices.append(pool_index) # now I know what layer in pool I am composing

            # compose layer to model and get s' and r from the model
            next_layer = self.layer_pool.layers[pool_index].layer

            final_layer = self.fcs.pop(-1)
            self.fcs.append(next_layer)  # compose new layer 
            self.fcs.append(final_layer) # re-insert final layer
        s_prime, reward = self.run_inner_loop()

        # temporary terminate logic
        done = False # TODO(ms): update to make episodic? is this continuous learning?

        return (
            s_prime,
            reward, 
            done, 
            {}
        )
    
    def render(self, mode='human', close=False):
        print(f'[INFO] # of layers {len(self.fcs)}')

    def close(self):
        pass


class Rlmetalearn:
    '''
    Meta-policy. The parameters are learned by training on multiple different tasks, where
    each task is represented by a different dataset (X,y) and is an instance of the 
    TargetNetwork class.

    Uses multiple environmental variables, whose default values are specified with print_env().

    Example usage.
        $ pool = LayerPool(in_features, out_features)  # this assumes the same architecture for every layer in the pool
        $ tasks = { str(round(f))+str("_task") : ((np.linspace(-2, 2, 100), (xvalues/round(f)) + np.sin(4*xvalues) + np.random.normal(0, 0.2, 100) for f in np.linspace(1, 1000, NUM_TASKS)) }
        $ rml = ReinforcementMetaLearning(pool, tasks)  # "you have this pool of layers, now learn these tasks"
        $ rml.train()
        $ rml.introspect(X, y) # TODO(ms): decide what this will do, then how.

    https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html#example

    '''
    def __init__(self,
                layer_pool,
                tasks,
                timesteps=5000,
                device='cpu',
                overwrite=True, 
                intra_update=True,
                network_depth_limit=5,
                network_num_nodes=32):
        self.layer_pool = layer_pool
        self.tasks = tasks
        self.out_features = np.array(self.tasks[list(self.tasks.keys())[0]][1]).shape # output shape; assumes uniformity across tasks
        self.timesteps = timesteps
        self.device = device
        self.overwrite = overwrite  # TODO(ms): to implement this, need some filter on the actions returned
        self.intra_update = intra_update
        self.network_depth_limit = network_depth_limit
        self.network_num_nodes = network_num_nodes

    def train(self):
        # train on all tasks
        task_to_yhats = {}
        task_to_fcs = {}
        for i, (n, (x, y)) in enumerate(self.tasks.items()):
            print(f'[INFO] Training {i+1}/{len(self.tasks)}...')
            env = TargetNetwork(x, 
                                y, 
                                self.layer_pool,
                                self.network_depth_limit,
                                self.network_num_nodes)
            if i==0: # create new rl agent
                model  = A2C('MlpPolicy', env)
            else:    # just update the env
                model.set_env(env)

            # use stablebaselines3 api to learn
            model.learn(total_timesteps=self.timesteps)          

            # predict and store
            y_hats = predict_from_composed_layers(env.fcs)
            task_to_yhats[n] = y_hats
            task_to_fcs[n] = env.fcs

            # update
            self.update_params(env.layerpool_indices, env.fcs)

        self.task_to_yhats = task_to_yhats
        self.task_to_fcs = task_to_fcs
        

    def update_params(self, pool_indices, layers):
        # push updated params to pool
        # NOTE  number of layerpool_indices and layers should be 1:1
        for i in range(len(pool_indices)):
            pool_index = pool_indices[i]
            updated_layer_copy = layers[i+1]
            self.layer_pool.layers[pool_index].layer = updated_layer_copy
            self.layer_pool.layers[pool_index].used = True
            self.layer_pool.layers[pool_index].times_used += 1


NUM_LAYERS=1000
NUM_NODES=32
NETWORK_DEPTH=5
SEED=123
LOSS_FN=torch.nn.MSELoss()
EPOCHS=1000
NUM_TASKS=20
TIMESTEPS=20000
DEVICE='cpu'

def print_env():
    print(f'POOL_NUM_LAYERS  {NUM_LAYERS}')
    print(f'NETWORK_NUM_NODES  {NUM_NODES}')
    print(f'NETWORK_DEPTH  {NETWORK_DEPTH}')
    print(f'NUM_TASKS  {NUM_TASKS}')
    print(f'SEED  {SEED}')
    print(f'LOSS_FN  {LOSS_FN}')
    print(f'EPOCHS  {EPOCHS}')
    print(f'TIMESTEPS  {TIMESTEPS}')
    print(f'DEVICE  {DEVICE}')

print_env()