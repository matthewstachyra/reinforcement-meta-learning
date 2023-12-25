# sequence is adding layers from pool with negative loss learning signal
import os
import argparse
import datetime
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
from typing import (Type,List,Tuple,)
import sb3_contrib
import stable_baselines3
import wandb

def generate_config():
    config = {
        'seed' : 41,
        'device' : 'cuda',
        'wandb_tag' : '',
        'run_datetime' : '',
        'exp_file' : '',
        'n_runs' : 1,
        'epochs' : 3,
        'timesteps' : 50,
        'n_x' : 100,
        'n_tasks' : 3,
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
        'learning_rate' : 0.01,
        'meta_learning_rate' : 0.0003, # 0.01, 0.001, 0.0003
        'meta_clip_range' : 0.2,       # 0.1, 0.2 (default), 0.3 
        'meta_n_steps' : 2048,         # 5, 128, 512, 1024, 2048 (default)
        'loss_fn' : torch.nn.MSELoss(),
        'sb3_model' : 'PPO',
        'data_dir' : 'data',
        }
    config['n_pool_hidden_layers'] = config['n_tasks'] * config['n_hidden_layers_per_network']
    return config

default_config = generate_config()
parser = argparse.ArgumentParser(description="REML command line")
parser.add_argument("--exp_file", type=str, default=default_config['exp_file'], help="Path to JSON file containing list of dictionaries with experiments")
parser.add_argument('--n_runs', type=int, default=default_config['n_runs'], help='Number of runs', required=False)
parser.add_argument('--n_tasks', type=int, default=default_config['n_tasks'], help='Number of tasks to generate', required=False)
parser.add_argument('--epochs', '-e', type=int, default=default_config['epochs'], help='Epochs', required=False)
parser.add_argument('--timesteps', '-t', type=int, default=default_config['timesteps'], help='Timesteps', required=False)
parser.add_argument('--sb3_model',  type=str, default=default_config['sb3_model'], help='SB3 model to use', required=False)
parser.add_argument('--data_dir', '-o', type=str, default=default_config['data_dir'], help='Directory to save tensorboard logs', required=False)
parser.add_argument('--wandb_tag', type=str, default=default_config['wandb_tag'], help='Name of run on wandb', required=False)
parser.add_argument('--meta_learning_rate', type=float, default=default_config['meta_learning_rate'], help='Learning rate', required=False)
parser.add_argument('--meta_clip_range', type=float, default=default_config['meta_clip_range'], help='Clip range', required=False)
parser.add_argument('--meta_n_steps', type=int, default=default_config['meta_n_steps'], help='Horizon', required=False)
args = parser.parse_args()
config = { key : getattr(args, key, default_value) for key, default_value in default_config.items() }

def set_seed(seed=None):
    if seed==None:
        seed = random.randint(1, 1000)
    config['seed'] = seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
set_seed()

def generate_tasks():
    lower_bound = torch.tensor(-5).float()
    upper_bound = torch.tensor(5).float()
    X = np.linspace(lower_bound, upper_bound, config['n_x'])
    amplitude_range = torch.tensor([0.1, 5.0]).float()
    phase_range = torch.tensor([0, math.pi]).float()
    amps = torch.rand(config['n_tasks'], 1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
    phases = torch.rand(config['n_tasks'], 1) * (phase_range[1] - phase_range[0]) + phase_range[0]
    tasks_data = torch.tensor(np.array([
        X
        for _ in range(config['n_tasks'])
    ]).astype(np.float32)).float()
    tasks_targets = torch.tensor(np.array([
        [(a * np.sin(x + p)).float()
        for x in X]
        for a, p in zip(amps, phases)
    ]).astype(np.float32)).float()
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
    return tasks_data, tasks_targets, tasks_info

def setup_path(path, run_datetime):
   os.makedirs(path, exist_ok=True) 
   os.makedirs(os.path.join(path, run_datetime), exist_ok=True)
   path = f"{path}/{run_datetime}"
   print(path)
   return path

def generate_configs(experiments, base_config=config):
    configs = []
    for value_dict in experiments:
        new_config = copy.deepcopy(base_config)
        for key, value in value_dict.items():
            new_config[key] = value 
        configs.append(new_config)
    print(configs)
    return configs

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
        for _ in range(50):
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
        # wandb.log({ f'loss_task{task_num}_per_step' : self.curr['loss']})
        # wandb.log({ f'reward_task{task_num}_per_step' : self.curr['reward']})
        # wandb.log({ f'action_types_task{task_num}_per_step' : wandb.Histogram(torch.tensor([self.actions_taken]))})
        # wandb.log({ f'pool_indices_task{task_num}_per_step' : wandb.Histogram(torch.tensor(self.layers_pool_indices))})

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
        # TODO is to try reseting local max and min losees at the episode level and not
        # just at the epoch level as is now
        # self.local_max_loss = None
        # self.local_min_loss = None

        self.train()
        self.next_batch()
        self.train_inner_network()
        return self.build_state(), None

class REML:
    def __init__(
        self,
        layer_pool: LayerPool,
        tasks: List[InnerNetworkTask],
        run: str,
        config=config,
        ):
        self.layer_pool = layer_pool
        self.tasks = tasks
        self.config = config
        self.run = run
        dummy_env = self.make_env(tasks[0], layer_pool) 
        if config['sb3_model']=='PPO':
            self.model = stable_baselines3.PPO(
                'MlpPolicy', 
                dummy_env, 
                seed=config['seed'], 
                n_steps=config['meta_n_steps'], 
                learning_rate=config['meta_learning_rate'],
                clip_range=config['meta_clip_range'],
                )
        elif config['sb3_model']=='RecurrentPPO':
            self.model = sb3_contrib.RecurrentPPO(
                'MlpLstmPolicy', 
                dummy_env, 
                seed=config['seed'], 
                n_steps=config['meta_n_steps'], 
                learning_rate=config['meta_learning_rate'],
                clip_range=config['meta_clip_range'],
                )
        self.task_cumureward = defaultdict(lambda: [])
        self.task_cumuloss = defaultdict(lambda: [])
        self.task_errors = defaultdict(lambda: [])

    def __str__(self) -> str:
        return f"REML(model={self.config['sb3_model']}, policy={self.config['sb3_policy']})"
    
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
            self.model.learn(total_timesteps=self.config['timesteps'])
            config['task_min_loss'][task.info['i']] = min(self.env.task_min_loss, self.config['task_min_loss'][task.info['i']])
            config['task_max_loss'][task.info['i']] = max(self.env.task_max_loss, self.config['task_max_loss'][task.info['i']])
    
    def train(self):
        self.calibrate()
        for epoch in range(self.config['epochs']):
            print(f"[INFO] Epoch={epoch+1}/{self.config['epochs']}")
            for i, task in enumerate(self.tasks): 
                print(f'[INFO] Task={i+1}/{len(self.tasks)}')

                # each task gets its own network
                self.task = task 
                self.env = self.make_env(task, epoch=epoch)
                self.model.set_env(self.env)
                self.model.learn(total_timesteps=self.config['timesteps'])
                
                # update min and max loss for task
                local_min_loss = self.env.local_min_loss
                local_max_loss = self.env.local_max_loss
                self.config['task_min_loss'][self.task.info['i']] = min(local_min_loss, self.config['task_min_loss'][self.task.info['i']])
                self.config['task_max_loss'][self.task.info['i']] = max(local_max_loss, self.config['task_max_loss'][self.task.info['i']])

                # track reward and loss for plots
                self.task_cumureward[str(self.task.info['i'])].append(sum(self.env.reward_vals))
                self.task_cumuloss[str(self.task.info['i'])].append(sum(self.env.loss_vals))
                self.task_errors[str(self.task.info['i'])].append(self.env.errors)

                # log to wandb
                wandb.log({ f'errors_run{self.run}_task{i}_per_epoch' : self.env.errors })
                wandb.log({ f'cumulative_reward_run{self.run}_task{i}_per_epoch' : sum(self.env.reward_vals) })
                wandb.log({ f'cumulative_loss_run{self.run}_task{i}_per_epoch' : sum(self.env.loss_vals) })
                wandb.log({ f'pool_indices_run{self.run}_task{i}_per_epoch' : wandb.Histogram(torch.tensor(self.env.layers_pool_indices))})

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

def run(experiments, config, seed=41):
    set_seed(seed)
    run_datetime = datetime.datetime.now().strftime("%m%d_%H%M")
    config['run_datetime'] = run_datetime
    config['wandb_tag'] = f'tuning_{run_datetime}'
    configs = generate_configs(experiments)
    path = setup_path(path='tune', run_datetime=run_datetime)
    wandb.init(
        project='reinforcement-meta-learning',
        config=config,
        name=config['wandb_tag']
    )

    data, targets, info = generate_tasks()
    tasks = [InnerNetworkTask(data=data[i], targets=targets[i], info=info[i]) for i in range(config['n_tasks'])]
    eval_task = random.choice(list(tasks))
    training_tasks = list(set(tasks) - {eval_task})
    torch.save(training_tasks, os.path.join(path, f'trainingtasks.pth'))
    torch.save(eval_task, os.path.join(path, f'evaltask.pth'))

    exp_tasks = defaultdict(lambda: defaultdict(lambda: []))
    for exp, config in zip(experiments, configs): 
        pool = LayerPool()
        reml = REML(tasks=training_tasks, layer_pool=pool, run=exp, config=config)
        reml.train()

        # save model, layers
        exp_string = ''.join(f'{key}{value}' for key, value in exp.items())
        reml.model.save(os.path.join(path, f"model_{exp_string}"))
        layers = copy.deepcopy(pool.layers)
        layers.insert(0, pool.initial_input_layer)
        layers.append(pool.initial_output_layer)
        torch.save(layers, os.path.join(path, f"layers_{exp_string}.pth"))

        # save cumulative rewards
        exp_tasks[exp_string] = reml.task_cumureward # { task: [return, return, return, ...] }

    with open(os.path.join(path, f"exp_tasks"), 'w') as json_file:
        json.dump(exp_tasks, json_file, indent=4)

    with open(os.path.join(path, f"experiments"), 'w') as json_file:
        json.dump(experiments, json_file, indent=4)
    
    return path

def run(config, experiments=None, seed=41):
    set_seed(seed)
    run_datetime = datetime.datetime.now().strftime("%m%d_%H%M")
    config['run_datetime'] = run_datetime

    if experiments:
        path = setup_path(path='tune', run_datetime=run_datetime)
        config['wandb_tag'] = f'tune_{run_datetime}'
        configs = generate_configs(experiments)
        data, targets, info = generate_tasks()
        tasks = [InnerNetworkTask(data=data[i], targets=targets[i], info=info[i]) for i in range(config['n_tasks'])]
        eval_task = random.choice(list(tasks))
        training_tasks = list(set(tasks) - {eval_task})
        torch.save(training_tasks, os.path.join(path, f'trainingtasks.pth'))
        torch.save(eval_task, os.path.join(path, f'evaltask.pth'))
        wandb.init(
            project='reinforcement-meta-learning',
            config=config,
            name=config['wandb_tag']
        )
        exp_tasks = defaultdict(lambda: defaultdict(lambda: []))
        for exp, config in zip(experiments, configs): 
            pool = LayerPool()
            reml = REML(tasks=training_tasks, layer_pool=pool, run=exp, config=config)
            reml.train()

            # save model, layers
            exp_string = ''.join(f'{key}{value}' for key, value in exp.items())
            reml.model.save(os.path.join(path, f"model_{exp_string}"))
            layers = copy.deepcopy(pool.layers)
            layers.insert(0, pool.initial_input_layer)
            layers.append(pool.initial_output_layer)
            torch.save(layers, os.path.join(path, f"layers_{exp_string}.pth"))

            # save cumulative rewards
            exp_tasks[exp_string] = reml.task_cumureward # { task: [return, return, return, ...] }

        with open(os.path.join(path, f"exp_tasks"), 'w') as json_file:
            json.dump(exp_tasks, json_file, indent=4)

        with open(os.path.join(path, f"experiments"), 'w') as json_file:
            json.dump(experiments, json_file, indent=4)
    else:
        path = setup_path(path='evaluation', run_datetime=run_datetime)
        data, targets, info = generate_tasks()
        tasks = [InnerNetworkTask(data=data[i], targets=targets[i], info=info[i]) for i in range(config['n_tasks'])]
        eval_task = random.choice(list(tasks))
        training_tasks = list(set(tasks) - {eval_task})
        torch.save(training_tasks, os.path.join(path, f'trainingtasks.pth'))
        torch.save(eval_task, os.path.join(path, f'evaltask.pth'))
        config['wandb_tag'] = f'evaluation_{run_datetime}'
        wandb.init(
            project='reinforcement-meta-learning',
            config=config,
            name=config['wandb_tag']
        )
        task_run_cumureward = defaultdict(lambda: [])
        task_run_cumuloss = defaultdict(lambda: [])
        task_run_errors =  defaultdict(lambda: [])
        for run in range(1, config['n_runs']+1):

            set_seed(seed) # new seed per task

            pool = LayerPool()
            reml = REML(tasks=training_tasks, layer_pool=pool, run=run, config=config)
            reml.train()

            # save model, layers
            reml.model.save(os.path.join(path, f"model_{run}"))
            layers = copy.deepcopy(pool.layers)
            layers.insert(0, pool.initial_input_layer)
            layers.append(pool.initial_output_layer)
            torch.save(layers, os.path.join(path, f"layers_{run}.pth"))

            # save return, loss, and error data
            for task in tasks:
                task_run_cumureward[str(task.info['i'])].append(reml.task_cumureward[str(task.info['i'])])
                task_run_cumuloss[str(task.info['i'])].append(reml.task_cumuloss[str(task.info['i'])])
                task_run_errors[str(task.info['i'])].append(reml.task_errors[str(task.info['i'])])
            with open(os.path.join(path, f'cumureward_{run}'), 'w') as json_file:
                json.dump(task_run_cumureward, json_file, indent=4)
            with open(os.path.join(path, f'cumuloss_{run}'), 'w') as json_file:
                json.dump(task_run_cumuloss, json_file, indent=4)
            with open(os.path.join(path, f'errors_{run}'), 'w') as json_file:
                json.dump(task_run_errors, json_file, indent=4)
        
    return path

if __name__ == "__main__":
    args = parser.parse_args() 
    if args.exp_file:
        with open(args.exp_file, "r") as f:
            experiments = json.load(f)
            run(experiments, config)
    else:
        run(config)
