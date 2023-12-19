# sequence is adding layers from pool with negative loss learning signal
import os
import argparse
import math
import copy 
import random
import datetime
from collections import defaultdict
from enum import Enum
import json
import numpy as np 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gymnasium
from typing import (
    Type,
    List,
    Tuple,
)
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import wandb
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee', 'notebook', 'bright'])
plt.rcParams.update({'figure.dpi': '75'})

# configuration
default_config = {
    'seed' : 41,
    'device' : 'cuda',
    'wandb_run' : '',
    'n_runs' : 5,
    'epochs' : 5,
    'timesteps' : 100,
    'n_x' : 100,
    'n_tasks' : 6,
    'task_min_loss' : defaultdict(lambda: float('inf')),
    'task_max_loss' : defaultdict(lambda: -float('inf')),
    'in_features' : 1,
    'out_features' : 1,
    'n_pool_hidden_layers' : 10,
    'n_hidden_layers_per_network' : 3,
    'n_layers_per_network' : 5,
    'n_nodes_per_layer' : 40,
    'pool_layer_type' : torch.nn.Linear,
    'batch_size' : 64,
    'learning_rate' : 0.005,
    'discount_factor' : 0.95,
    'loss_fn' : torch.nn.MSELoss(),
    'sb3_model' : 'PPO',
    'sb3_policy' : 'MlpPolicy',
    'data_dir' : 'data',
    }
config = default_config
config['n_pool_hidden_layers'] = config['n_tasks'] * config['n_hidden_layers_per_network']
parser = argparse.ArgumentParser(description="REML command line")
parser.add_argument('--device', type=str, default=default_config['device'], help='Device to run computations', required=False)
parser.add_argument('--n_runs', type=int, default=default_config['n_runs'], help='Number of runs', required=False)
parser.add_argument('--n_tasks', type=int, default=default_config['n_tasks'], help='Number of tasks to generate', required=False)
parser.add_argument('--epochs', '-e', type=int, default=default_config['epochs'], help='Epochs', required=False)
parser.add_argument('--timesteps', '-t', type=int, default=default_config['timesteps'], help='Timesteps', required=False)
parser.add_argument('--sb3_model',  type=str, default=default_config['sb3_model'], help='SB3 model to use', required=False)
parser.add_argument('--sb3_policy', type=str, default=default_config['sb3_policy'], help='SB3 policy to use', required=False)
parser.add_argument('--data_dir', '-o', type=str, default=default_config['data_dir'], help='Directory to save tensorboard logs', required=False)
parser.add_argument('--n_layers_per_network', type=int, default=default_config['n_layers_per_network'], help='Number of layers per network', required=False)
args = parser.parse_args()
config = { key : getattr(args, key, default_value) for key, default_value in default_config.items() }

# initialize wandb
wandb.init(
    project='reinforcement-meta-learning',
    config=config,
    name=config['wandb_run']
)
print(f'[INFO] Config={config}')

# create tasks
lower_bound = torch.tensor(-5).float()
upper_bound = torch.tensor(5).float()
X = np.linspace(lower_bound, upper_bound, config['n_x'])
amplitude_range = torch.tensor([0.1, 5.0]).float()
phase_range = torch.tensor([0, math.pi]).float()
amps = torch.from_numpy(np.linspace(amplitude_range[0], amplitude_range[1], config['n_tasks'])).float()
phases = torch.from_numpy(np.linspace(phase_range[0], phase_range[1], config['n_tasks'])).float()
tasks_data = torch.tensor(np.array([ 
        X
        for _ in range(config['n_tasks'])
        ])).float()
tasks_targets = torch.tensor(np.array([
        [((a * np.sin(x)) + p).float()
        for x in X] 
        for a, p in zip(amps, phases)
        ])).float()
tasks_info = [
        {'i' : i, 
         'amp' : a, 
         'phase_shift' : p, 
         'lower_bound' : lower_bound, 
         'upper_bound' : upper_bound, 
         'amplitude_range_lower_bound' : amplitude_range[0], 
         'amplitude_range_upper_bound' : amplitude_range[1], 
         'phase_range_lower_bound' : phase_range[0],
         'phase_range_lower_bound' : phase_range[1]}
        for i, (a, p) in enumerate(zip(amps, phases))
]
print(f'[INFO] Tasks created.')

class LayerPool:
    def __init__(self, 
                layer_constructor: Type[torch.nn.Linear]=config['pool_layer_type'],
                in_features: int=config['in_features'],
                out_features: int=config['out_features'],
                num_nodes_per_layer: int=config['n_nodes_per_layer'],
                layers: List[torch.nn.Linear]=None):
        self.layer_constructor = layer_constructor
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes_per_layer = num_nodes_per_layer

        if layers is None:
            self.layers = [self.layer_constructor(in_features=num_nodes_per_layer, out_features=num_nodes_per_layer)for _ in range(config['n_pool_hidden_layers'])]
            for _ in range(config['n_tasks']):
                self.layers.append(self.layer_constructor(in_features=in_features, out_features=num_nodes_per_layer))
                self.layers.append(self.layer_constructor(in_features=num_nodes_per_layer, out_features=out_features))
            [torch.nn.init.xavier_uniform_(layer.weight) for layer in self.layers]
        else:
            self.layers = layers
            config['n_pool_hidden_layers'] = len(self.layers)

        self.initial_input_layer = copy.deepcopy(random.choice([layer for layer in self.layers if layer.in_features==1]))
        self.initial_output_layer = copy.deepcopy(random.choice([layer for layer in self.layers if layer.out_features==1]))
        
    def __str__(self) -> str:
        return f"LayerPool(size={self.size}, layer_type={config['pool_layer_type']}, num_nodes_per_layer={config['n_nodes_per_layer']}"

class InnerNetworkAction(Enum):
    # no action is 0 to prevent multiplication
    # by 0 in building s prime with the past n
    # action enum values
    ADD = 1
    ERROR = 2

class InnerNetworkTask(Dataset):
    def __init__(self, data, targets, info):
        self.data = data 
        self.targets = targets
        self.info = info

    def __len__(self):
        assert len(self.data) == config['n_x'], '[ERROR] Length should be the same as n_x.'
        return len(self.data)

    def __getitem__(self, index):
        assert self.data[index].dtype == torch.float32, f'[ERROR] Expected type torch.float32, got type: {self.data[index].dtype}'
        assert self.targets[index].dtype == torch.float32, f'[ERROR] Expected type torch.float32, got type: {self.targets[index].dtype}'
        sample = {
            'x' : self.data[index],
            'y' : self.targets[index],
            'info' : self.info
        }
        return sample
    
    def __str__(self):
        return f'[INFO] InnerNetworkTask(data={self.data}, targets={self.targets}, info={self.info})'

class InnerNetwork(gymnasium.Env, torch.nn.Module):
    def __init__(self, 
                task: InnerNetworkTask,
                layer_pool: LayerPool,
                calibration: bool=False,
                epoch: int=0,
                in_features: int=config['in_features'],
                out_features: int=config['out_features'],
                learning_rate: float=config['learning_rate'],
                batch_size: int=config['batch_size'],
                shuffle: bool=True,
                ):
        super(InnerNetwork, self).__init__()
        self.epoch = epoch
        self.task = task
        self.layer_pool = layer_pool
        self.calibration = calibration
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task_max_loss = config['task_max_loss'][self.task.info['i']]
        self.task_min_loss = config['task_min_loss'][self.task.info['i']]
        self.local_max_loss = -float('inf')
        self.local_min_loss = float('inf')

        self.prev = defaultdict(lambda: None)
        self.curr = defaultdict(lambda: None)
        self.data_loader = DataLoader(task, batch_size=batch_size, shuffle=shuffle)
        self.data_iter = iter(self.data_loader)
       
        # initial input and output layers to allow state calculation to get actions. these layers
        # are then replaced by the outer network. the objective of the outer network is to find
        # the best layers for a neural network (that means not just selecting the hidden layers).
        # the same hidden layers produce drastically different results with different input and output
        # layers 
        self.initial_input_layer = layer_pool.initial_input_layer
        self.initial_output_layer = layer_pool.initial_output_layer
        self.layers = torch.nn.ModuleList([self.initial_input_layer, self.initial_output_layer]) 
        self.layers_pool_indices = [] 
        self.actions_taken = []
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)

        self.timestep = 0
        self.loss_vals = []
        self.reward_vals = []
        self.errors = 0

        self.train()
        self.next_batch()
        self.train_inner_network()
        self.observation_space = gymnasium.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=self.build_state().shape)
        self.action_space = gymnasium.spaces.discrete.Discrete(len(self.layer_pool.layers))

    def step(self, action: np.int64) -> Tuple[torch.Tensor, float, bool, dict]: 
        assert action.shape == (), f'[ERROR] Expected action shape () for scalar {self.action_space.n}, got: {action.shape}'
        assert action.dtype == np.int64, f'[ERROR] Expected np.int64 dtype, got: {action.dtype}'

        self.timestep += 1
        self.prev = self.curr
        self.curr = defaultdict(lambda: None)
        self.next_batch()
        self.update(action)
        termination = False if len(self.layers)<config['n_layers_per_network'] else True
        if termination:
            self.train_inner_network()
        else:
            self.forward()

        # calibration is finding the min and max loss values for the task to
        # scale the loss (and the reward) between 0 and 1 across tasks
        if self.calibration==True:
            self.task_max_loss = max(self.curr['loss'], self.task_max_loss)
            self.task_min_loss = min(self.curr['loss'], self.task_min_loss) 
        self.local_max_loss = max(self.curr['loss'], self.local_max_loss)
        self.local_min_loss = min(self.curr['loss'], self.local_min_loss) 
        
        s_prime = self.build_state()
        reward = self.reward()
        self.log()

        # update pool
        for index, layer in zip(self.layers_pool_indices, self.layers[1:-1]):
            self.layer_pool.layers[index] = layer

        return (
            s_prime,
            reward, 
            termination,
            False,
            {}
        )

    def next_batch(self, throw_exception=False) -> None:
        if (throw_exception):
            batch = next(self.data_iter)
            self.curr['x'] = batch['x'].view(-1, 1)
            self.curr['y'] = batch['y'].view(-1, 1)
            self.curr['info'] = batch['info']
        else: 
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_loader = DataLoader(self.task, batch_size=self.batch_size, shuffle=self.shuffle)
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)
            finally:
                self.curr['x'] = batch['x'].view(-1, 1)
                self.curr['y'] = batch['y'].view(-1, 1)
                self.curr['info'] = batch['info']
    
    def update(self, action: np.int64) -> None:
        new_layer = self.layer_pool.layers[action]
        
        # first step and input layer and not already in network
        if self.timestep==1 \
            and new_layer.in_features==1 \
            and new_layer not in self.layers:
            
            self.layers[0] = new_layer
            self.curr['action_type'] = InnerNetworkAction.ADD
        # last step and output layer and not already in network
        elif self.timestep==2 \
            and new_layer.out_features==1 \
            and new_layer not in self.layers:

            self.layers[-1] = new_layer
            self.curr['action_type'] = InnerNetworkAction.ADD
        # not first or last step and hidden layer and not already in network
        elif self.timestep!=1 \
            and self.timestep!=2 \
            and new_layer not in self.layers \
            and new_layer.in_features!=1 \
            and new_layer.out_features!=1 \
            and len(self.layers) < config['n_layers_per_network']: 

            final_layer = self.layers.pop(-1) 
            self.layers.append(new_layer)
            self.layers.append(final_layer) 
            self.layers_pool_indices.append(torch.tensor(action))
            self.curr['action_type'] = InnerNetworkAction.ADD
        else: 
            self.curr['action_type'] = InnerNetworkAction.ERROR
            
    def forward(self) -> None:
        x = copy.deepcopy(self.curr['x'])
        for i in range(len(self.layers) - 1): 
            x = torch.nn.functional.relu(self.layers[i](x))
        self.curr['latent_space'] = x
        self.curr['y_hat'] = self.layers[-1](x) 
        self.curr['loss'] = self.loss_fn(self.curr['y'], self.curr['y_hat'])
        self.curr['loss'].backward()
    
    def train_inner_network(self) -> None: 
        for _ in range(10):
            self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate) 
            self.opt.zero_grad()
            self.forward()
            self.opt.step()
            self.next_batch()
    
    def build_state(self) -> np.ndarray:
        task_info = torch.tensor([self.task.info['amp'], self.task.info['phase_shift']]).squeeze()
        loss = torch.Tensor([self.curr['loss']])
        yhat_scale = torch.Tensor([torch.Tensor(torch.max(torch.abs(self.curr['y_hat']))).detach().item()])
        one_hot_layers = torch.tensor(np.array([1 if self.layer_pool.layers[i] in self.layers else 0 for i in range(len(self.layer_pool.layers))]))
        layer_indices = [index + 1 for index in self.layers_pool_indices.copy()] # 0 bump by 1 to avoid 0th index layer being 0 since padding 0s too
        while len(layer_indices) < config['n_layers_per_network']:
            layer_indices.insert(0, 0)
        layer_indices = torch.tensor(layer_indices)
        return torch.concat((
            task_info,
            yhat_scale,
            layer_indices,
            one_hot_layers,
            loss,
        ), dim=0).detach().numpy()
    
    def reward(self) -> torch.Tensor:
        # min-max scaled reward is negative loss of inner network multiplied 
        # by a scale factor that is "how bad" initial layers chosen are to 
        # credit those early actions more in the return

        if self.calibration:
            scale_factor = 1
        else:
            # "how bad" the initial layers are is a function of their loss 
            # versus the min and max loss seen for task to ensure that credit 
            # assignment skews towards ADD rather than TRAIN actions (because 
            # Adam optimizer can train any set of layers to good performance 
            # in few steps, but that's not the learning objective) 
            # e.g., with max loss for task=14, 
            # max loss for task=12, reduce each reward with a factor of 
            # 0.14 <- 14-12/14 = 2/14 = 0.14
            scale_factor = ((self.task_max_loss - self.local_max_loss) / self.task_max_loss) 

        if (self.curr['action_type'] == InnerNetworkAction.ERROR):
            reward = torch.tensor(-1)
        else:
            epsilon = 1e-8 # prevent division by zero
            reward = - (((self.curr['loss'] - self.task_min_loss + epsilon) / (self.task_max_loss - self.task_min_loss + epsilon)))
            reward = scale_factor * reward
        
        self.curr['reward'] = reward
        return reward

    def log(self):
        task_num = str(self.task.info['i'])
        self.loss_vals.append(copy.copy(self.curr['loss'].item()))
        self.reward_vals.append(copy.copy(self.curr['reward'].item()))
        self.errors = self.errors + 1 if self.curr['action_type']==InnerNetworkAction.ERROR else self.errors
        wandb.log({ f'loss_task{task_num}_per_step' : self.curr['loss']})
        wandb.log({ f'reward_task{task_num}_per_step' : self.curr['reward']})
        wandb.log({ f'action_types_task{task_num}_per_step' : wandb.Histogram(torch.tensor([self.actions_taken]))})
        wandb.log({ f'pool_indices_task{task_num}_per_step' : wandb.Histogram(torch.tensor(self.layers_pool_indices))})

    def reset(self, seed=None) -> np.ndarray:
        # print(f"[INFO] Reset at timestep={self.timestep}, layers={self.layers_pool_indices}")
        self.timestep = 0
        self.prev = defaultdict(lambda: None)
        self.curr = defaultdict(lambda: None)
        self.initial_input_layer = copy.deepcopy(random.choice([layer for layer in self.layer_pool.layers if layer.in_features==1]))
        self.initial_output_layer = copy.deepcopy(random.choice([layer for layer in self.layer_pool.layers if layer.out_features==1]))
        self.layers = torch.nn.ModuleList([self.initial_input_layer, self.initial_output_layer]) 
        self.layers_pool_indices = [] 
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        self.actions_taken = []
        self.loss_vals = []
        self.reward_vals = []

        self.train()
        self.next_batch()
        self.train_inner_network()
        return self.build_state(), None

class REML:
    def __init__(
        self,
        layer_pool: LayerPool,
        tasks: List[InnerNetworkTask],
        path: str,
        run: int=1,
        model=config['sb3_model'],
        policy=config['sb3_policy'],
        epochs: int=config['epochs'],
        timesteps: int=config['timesteps'],
        ):
        self.layer_pool = layer_pool
        self.tasks = tasks
        self.path=path
        if config['sb3_model']=='PPO':
            model = PPO
        elif config['sb3_model']=='RecurrentPPO':
            model = RecurrentPPO
        dummy_env = self.make_env(tasks[0], layer_pool) #TODO: why tasks[0]?
        self.run = run
        self.model = model(policy, dummy_env)
        self.policy = policy
        self.epochs = epochs
        self.timesteps = timesteps
        self.return_epochs = defaultdict(lambda: [])
        self.cumuloss_epochs = defaultdict(lambda: [])
        self.errors_epochs = defaultdict(lambda: [])

    def __str__(self) -> str:
        return f'REML(model={self.model}, policy={self.policy})'
    
    def make_env(self, task, epoch=None, calibration=False) -> gymnasium.Env:
        return gymnasium.wrappers.NormalizeObservation(InnerNetwork(task, self.layer_pool, epoch=epoch, calibration=calibration))
    
    def calibrate(self):
        # get the min and max loss per task to min-max
        # scale across tasks so no one task dominates learning
        # based on the magnitude of the loss (and reward) signal
        for i, task in enumerate(self.tasks): 
            print(f'[INFO] Calculating min and max loss for task {i+1}.')
            self.env = self.make_env(task, calibration=True)
            self.model.set_env(self.env)
            self.model.learn(total_timesteps=self.timesteps)
            config['task_min_loss'][task.info['i']] = min(self.env.task_min_loss, config['task_min_loss'][task.info['i']])
            config['task_max_loss'][task.info['i']] = max(self.env.task_max_loss, config['task_max_loss'][task.info['i']])
    
    def train(self):
        # wraps stablebaselines3 learn() so we call it n * m times
        # n is the number of epochs where we run all m tasks
        # we use the same policy, swapping out envs for the n tasks, m times. 

        # get min and max loss for task to calculate learning signals
        self.calibrate()

        # to calculate variance
        # e.g., task: [ n: [epoch: [100 values]] ] / array with n rows, epoch columns 
        # where cell @ [nth run][mth epoch] is cumulative loss/reward
        for epoch in range(self.epochs):
            print(f'[INFO] Epoch={epoch+1}/{self.epochs}')
            for i, task in enumerate(self.tasks): 
                print(f'[INFO] Task={i+1}/{len(self.tasks)}')

                # each task gets its own network
                self.task = task 
                self.env = self.make_env(task, epoch=epoch)
                self.model.set_env(self.env)
                self.model.learn(total_timesteps=self.timesteps)
                
                # update min and max loss for task
                local_min_loss = self.env.local_min_loss
                local_max_loss = self.env.local_max_loss
                config['task_min_loss'][self.task.info['i']] = min(local_min_loss, config['task_min_loss'][self.task.info['i']])
                config['task_max_loss'][self.task.info['i']] = max(local_max_loss, config['task_max_loss'][self.task.info['i']])

                # track reward and loss for plots
                self.return_epochs[str(self.task.info['i'])].append(sum(self.env.reward_vals))
                self.cumuloss_epochs[str(self.task.info['i'])].append(sum(self.env.loss_vals))
                self.errors_epochs[str(self.task.info['i'])].append(self.env.errors)

                # log to wandb
                wandb.log({ f'errors_run{self.run}_task{i}_per_epoch' : self.env.errors })
                wandb.log({ f'cumulative_reward_run{self.run}_task{i}_per_epoch' : sum(self.env.reward_vals) })
                wandb.log({ f'cumulative_loss_run{self.run}_task{i}_per_epoch' : sum(self.env.loss_vals) })
                wandb.log({ f'pool_indices_run{self.run}_task{i}_per_epoch' : wandb.Histogram(torch.tensor(self.env.layers_pool_indices))})

                plot_sine_curves(self, tasks=[self.task], path=self.path, run=self.run, epoch=epoch, image=True)

# plotting functions require loading reml parameters to new reml object
# e.g.,
#   reml = REML(layer_pool=pool, tasks=tasks)
#   reml.model.load('RecurrentPPO_12-37')

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(config['in_features'], config['n_nodes_per_layer']),  
            torch.nn.Linear(config['n_nodes_per_layer'], config['n_nodes_per_layer']), 
            torch.nn.Linear(config['n_nodes_per_layer'], config['out_features'])  
        ])

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

def plot_loss_vs_step_curves(reml,
                      tasks,
                      path,
                      steps=100,
                      image=False,
                      ) -> Tuple[dict]:

    meta_task_lossbystep = defaultdict(lambda: [])
    scratch_task_lossbystep = defaultdict(lambda: [])
    for task in tasks: 
        # get loss per step from meta-learned model
        env = reml.make_env(task)
        reml.model.set_env(env, force_reset=False)
        obs, _ = env.reset()
        while len(env.layers) < config['n_layers_per_network']:
            action, _ = reml.model.predict(obs)
            obs, _, done, _, _ = env.step(action)
            if done: # network built
                layers = copy.deepcopy(env.layers)
                network = RegressionModel()
                network.layers = layers
                optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
                criterion = torch.nn.MSELoss()
                for _ in range(steps):
                    optimizer.zero_grad() 
                    outputs = network(task.data.view(-1, 1))
                    loss = criterion(outputs, task.targets.view(-1,1))
                    loss.backward()
                    optimizer.step()
                    meta_task_lossbystep[task].append(loss)

        # get loss per step from model trained from scratch 
        network = RegressionModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
        network.train()
        for _ in range(steps):
            optimizer.zero_grad() 
            outputs = network(task.data.view(-1, 1))
            loss = criterion(outputs, task.targets.view(-1,1))
            loss.backward()
            optimizer.step()
            scratch_task_lossbystep[task].append(loss)

        # plot loss vs step curve for each task
        plt.figure()
        plot_title = f"Loss versus step (task={task.info['i']+1})"
        plot_path = f'{path}/{plot_title}.png'  
        meta_label = "Meta-trained network"
        scratch_label = "Baseline trained from scratch"
        plt.plot(range(steps), [loss.detach().numpy() for loss in meta_task_lossbystep[task]], label=meta_label)
        plt.plot(range(steps), [loss.detach().numpy() for loss in scratch_task_lossbystep[task]], label=scratch_label)
        plt.xlabel("Timestep")
        plt.ylabel("MSE Loss")
        plt.title(plot_title)
        plt.legend()

        # save png / wandb
        if image:
            plt.savefig(plot_path)
            wandb.log({plot_title: wandb.Image(plot_path)})

    # return meta_task_lossbystep, scratch_task_lossbystep

def plot_sine_curves(reml, 
                     tasks, 
                     path,
                     run=None,
                     epoch=None, 
                     image=False, 
                     eval_task=None,
                     args=defaultdict()) -> dict:

    task_yhats = defaultdict(lambda: None)
    for task in tasks: 
        if task==eval_task: continue
        # set env
        env = reml.make_env(task)
        reml.model.set_env(env, force_reset=False)

        # build network in env
        obs, _ = env.reset()
        while len(env.layers)!=config['n_layers_per_network']:
            action, _ = reml.model.predict(obs)
            obs, _, _, _, _ = env.step(action)

        # run network to get yhats
        xs, ys = task.data.clone(), task.targets.clone()
        xs, ys = xs.view(len(xs), 1), ys.view(len(ys), 1)
        env.curr['x'] = xs
        env.curr['y'] = ys
        env.forward()
        yhats = env.curr['y_hat']

        # plot sine curve
        plt.figure()
        if epoch!=None and run!=None:
            plot_title = f"Sine curve (run={run}, epoch={epoch}, task={task.info['i']+1})" 
        elif epoch!=None:
            plot_title = f"Sine curve (epoch={epoch}, task={task.info['i']+1})" 
        else:
            plot_title= f"Sine curve (task={task.info['i']+1})"
        plot_path = f'{path}/{plot_title}.png'  
        plt.plot(task.data, [yhat.detach().numpy() for yhat in yhats], **args)
        plt.plot(task.data, task.targets, label='ground truth', linestyle='--')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(plot_title)
        plt.legend()

        # save png / wandb
        if image:
            plt.savefig(plot_path)
            wandb.log({plot_title: wandb.Image(plot_path)})

        # return if needed 
        xs, yhats = task.data, [yhat.detach().numpy() for yhat in yhats]
        task_yhats[task] = yhats
    # return task_yhats

def plot_data_with_variance(data, eval_task_num, title, xlabel):
    task_mean = {}
    task_std = {}
    for i, task_values in enumerate(data.values()):
        if i==eval_task_num: continue
        array = np.array(task_values)
        assert np.ndim(array)==2, f"[ERROR] Expected ndim=2, got {np.ndim(array)}"

        # rows are over n_runs, cols are epochs
        task_mean[i] = [np.mean(array[:, epoch]) for epoch in range(array.shape[1])]
        task_std[i] = [np.std(array[:, epoch]) for epoch in range(array.shape[1])] 

        plt.figure()
        plt.plot(range(len(task_mean[i])), task_mean[i], label='mean')
        plt.fill_between(
            range(len(task_mean[i])), 
            [mean - std for mean, std in zip(task_mean[i], task_std[i])], 
            [mean + std for mean, std in zip(task_mean[i], task_std[i])],
            alpha=0.2, 
            label='std')
        plt.title(f"{title} (task={i+1})")
        plt.ylabel(xlabel)
        plt.xlabel("Epochs")
        plt.legend()

def plot_few_shot_learning(reml,
                           eval_task,
                           k=5):

    # network still recieves the same 100 x values {-5, ..., 5}
    # the difference is that a target value is only provided for 5 or 10 of these 100 values

    k_data_points = torch.tensor(random.sample(list(zip(eval_task.data, eval_task.targets)), k))
    env = reml.make_env(eval_task)
    reml.model.set_env(env, force_reset=False)
    reml.task = eval_task

    # build network in env, then generate plots
    obs, _ = env.reset()
    while len(env.layers)!=config['n_layers_per_network']:
        action, _ = reml.model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done: # network built
            layers = copy.deepcopy(env.layers)
            network = RegressionModel()
            network.layers = layers
            optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
            criterion = torch.nn.MSELoss()

            k_x = k_data_points[:, 0].clone()
            k_x = k_x.view(len(k_x), 1) # reshape batch to k (k, 1)
            k_y = k_data_points[:, 1].clone()
            k_y = k_y.view(len(k_y), 1)

            # 0 grad steps / preupdate on all data
            xs = eval_task.data.clone()
            preupdate_outputs = network(xs.view(len(xs), 1)) # reshape to dataset size (100, 1)

            # 1 grad step on k 
            for _ in range(1):
                optimizer.zero_grad()
                outputs = network(k_x)
                loss = criterion(outputs, k_y)
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            outputs = network(xs.view(len(xs), 1)) # but then get curve for full dataset
            outputs_after_1_grad_step = [output.detach().numpy() for output in outputs]

            # 10 grad step
            for _ in range(10):
                optimizer.zero_grad() 
                outputs = network(k_x)
                loss = criterion(outputs, k_y)
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            outputs = network(xs.view(len(xs), 1))
            outputs_after_10_grad_step = [output.detach().numpy() for output in outputs]
        
            continue

    # pre-update plot
    plt.figure()
    plt.plot(eval_task.data, [yhat.detach().numpy() for yhat in preupdate_outputs], linestyle='--', label='pre-update')

    # ground truth plot
    plt.plot(eval_task.data, eval_task.targets, linestyle='--', label='ground truth')

    # k points plot
    plt.scatter(k_data_points[:, 0], k_data_points[:, 1], marker='^', color='b', label='k points')

    # 1 grad step training plot
    plt.plot(eval_task.data, outputs_after_1_grad_step, label='1 grad step')
        
    # 10 grad steps training plot
    plt.plot(eval_task.data, outputs_after_10_grad_step, label='10 grad step')

    plt.title(f'Few shot learning (evaluation task, k={k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

if __name__ == "__main__":

    # initialize tasks
    tasks = [InnerNetworkTask(data=tasks_data[i], targets=tasks_targets[i], info=tasks_info[i]) for i in range(config['n_tasks'])]
    eval_task = random.choice(list(tasks))
    training_tasks = list(set(tasks) - {eval_task})
    pool = LayerPool()

    # run preliminaries
    return_task_runbyepoch = defaultdict(lambda: [])
    cumuloss_task_runbyepoch = defaultdict(lambda: [])
    errors_task_runbyepoch = defaultdict(lambda: [])
    model_param_paths = []
    name = f"{config['sb3_model']}_{datetime.datetime.now().strftime('%m%d-%H-%M')}"
    data_path = f"{config['data_dir']}/{name}"
    os.makedirs(data_path, exist_ok=True)

    # e.g., return_task_runbyepoch
    # task:      
    #              epoch 1  epoch 2 ... epoch m
    #       run 1  [[return, return, ...] 
    #       run 2   [return, return, ...]
    #        ...    [        ...        ]
    #       run n   [return, return, ...]]
    for run in range(1, config['n_runs']+1):     
        print(f"[INFO] n={run}")

        # paths
        model_path = f"{data_path.split('/', 1)[-1]}_{run}"
        model_param_paths.append(model_path)

        # train reml
        reml = REML(tasks=training_tasks, layer_pool=pool, run=run, path=data_path)
        reml.train()

        # save model params
        reml.model.save(model_path)

        # save data
        for task in tasks:
            return_task_runbyepoch[str(task.info['i'])].append(reml.return_epochs[str(task.info['i'])])
            cumuloss_task_runbyepoch[str(task.info['i'])].append(reml.cumuloss_epochs[str(task.info['i'])])
            errors_task_runbyepoch[str(task.info['i'])].append(reml.errors_epochs[str(task.info['i'])])
        
        # write data
        with open(f'{data_path}/{name}_returns', 'w') as json_file:
            json.dump(return_task_runbyepoch, json_file, indent=4)
        with open(f'{data_path}/{name}_cumuloss', 'w') as json_file:
            json.dump(cumuloss_task_runbyepoch, json_file, indent=4)
        with open(f'{data_path}/{name}_errors', 'w') as json_file:
            json.dump(errors_task_runbyepoch, json_file, indent=4)

    # load reml and data 
    reml = REML(layer_pool=pool, tasks=tasks, path=data_path)
    reml.model.load(model_param_paths[1]) #TODO: which model (if more than 1) to use for sine curves and few shot learning plots?
    path = os.path.join(os.getcwd(), data_path, name)
    returns = json.load(open(f"{path}_returns", 'r'))
    cumuloss = json.load(open(f"{path}_cumuloss", 'r'))
    errors = json.load(open(f"{path}_errors", 'r'))

    # generate plots
    plot_loss_vs_step_curves(reml=reml, tasks=training_tasks, path=data_path)
    plot_sine_curves(reml=reml, tasks=tasks, path=data_path)
    plot_data_with_variance(data=returns, eval_task_num=eval_task.info['i'], xlabel="Returns", title="Return by epoch")
    plot_data_with_variance(data=cumuloss, eval_task_num=eval_task.info['i'], xlabel="Cumulative loss", title="Cumulative loss by epoch")
    plot_data_with_variance(data=errors, eval_task_num=eval_task.info['i'], xlabel="Errors", title="Errors by epoch")
    plot_few_shot_learning(reml=reml, eval_task=eval_task, k=5)
    plot_few_shot_learning(reml=reml, eval_task=eval_task, k=10)
