import argparse
import math
import random
import datetime
from collections import defaultdict
from enum import Enum
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
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, A2C
import wandb

# configuration
default_config = {
    'seed' : 41,
    'device' : 'cuda',
    'pretrain' : False,
    'episodic' : True,
    'epochs' : 10,
    'timesteps' : 1000,
    'n_x' : 100,
    'n_tasks' : 6,
    'in_features' : 1,
    'out_features' : 1,
    'n_pool_hidden_layers' : 10,
    'n_hidden_layers_per_network' : 3,
    'n_layers_per_network' : 5,
    'n_nodes_per_layer' : 40,
    'pool_layer_type' : torch.nn.Linear,
    'batch_size' : 100,
    'learning_rate' : 0.05,
    'action_cache_size' : 5,
    'num_workers' : 0,
    'loss_fn' : torch.nn.MSELoss(),
    'sb3_model' : 'RecurrentPPO',
    'sb3_policy' : 'MlpLstmPolicy',
    'log_dir' : 'wandb',
    }
parser = argparse.ArgumentParser(description="REML command line")
parser.add_argument('--device', '-d', type=str, default=default_config['device'], help='Device to run computations', required=False)
parser.add_argument('--epochs', '-e', type=int, default=default_config['epochs'], help='Epochs', required=False)
parser.add_argument('--timesteps', '-t', type=int, default=default_config['timesteps'], help='Timesteps', required=False)
parser.add_argument('--sb3_model', '-m', type=str, default=default_config['sb3_model'], help='SB3 model to use', required=False)
parser.add_argument('--sb3_policy', '-p', type=str, default=default_config['sb3_policy'], help='SB3 policy to use', required=False)
parser.add_argument('--log_dir', '-o', type=str, default=default_config['log_dir'], help='Directory to save tensorboard logs', required=False)
parser.add_argument('--n_tasks', type=int, default=default_config['n_tasks'], help='Number of tasks to generate', required=False)
parser.add_argument('--n_layers_per_network', type=int, default=default_config['n_layers_per_network'], help='Number of layers per network', required=False)
parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain layers for layer pool.', required=False)
parser.add_argument('--episodic', action='store_true', help='Whether to train in episodes rather than continual learning.', required=False)
args = parser.parse_args()
config = { key : getattr(args, key, default_value) for key, default_value in default_config.items() }

# initialize wandb
wandb.init(
    project='reinforcement-meta-learning',
    config=config
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
print(f'[INFO] Tasks created.')

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(config['in_features'], 40),  
            torch.nn.Linear(40, 40), 
            torch.nn.Linear(40, 40),  
            torch.nn.Linear(40, 40),  
            torch.nn.Linear(40, config['out_features'])  
        ])

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# pretrain layers
if config['pretrain']:
    layers = []
    data = []
    for x, y in zip(tasks_data, tasks_targets):
        data.append([x,y])
    for i, (x, y) in enumerate(data):
        print(f'[INFO] Task {i+1} layers are pre-initialized.')
        model = RegressionModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20000
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(x.view(-1, 1))
            loss = criterion(outputs, y.view(-1,1))
            loss.backward()
            optimizer.step()

        # save layers for layer pool
        layers.extend(model.layers)

        print(f'[INFO] Layers pre-initialized on tasks.')

class LayerPool:
    def __init__(self, 
                size: int=config['n_pool_hidden_layers'], 
                layer_constructor: Type[torch.nn.Linear]=config['pool_layer_type'],
                in_features: int=config['in_features'],
                out_features: int=config['out_features'],
                num_nodes_per_layer: int=config['n_nodes_per_layer'],
                layers: List[torch.nn.Linear]=None):
        self.size = size if layers is None else len(layers)
        self.layer_constructor = layer_constructor
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes_per_layer = num_nodes_per_layer

        if layers is None:
            self.layers = [self.layer_constructor(in_features=num_nodes_per_layer, out_features=num_nodes_per_layer)
                for _ in range(size)]
            for _ in range(config['n_tasks']):
                self.layers.append(self.layer_constructor(in_features=in_features, out_features=num_nodes_per_layer))
                self.layers.append(self.layer_constructor(in_features=num_nodes_per_layer, out_features=out_features))
            [torch.nn.init.xavier_uniform_(layer.weight) for layer in self.layers]
        else:
            self.layers = layers
            config['n_pool_hidden_layers'] = len(self.layers)
        
    def __str__(self) -> str:
        return f"LayerPool(size={self.size}, layer_type={config['pool_layer_type']}, num_nodes_per_layer={config['n_nodes_per_layer']}"


# TODO is to move this to utils
def get_params_and_gradients(layers):
    num_layers = len(layers)
    max_num_hidden_layers = config['n_layers_per_network'] - 2
    num_hidden_layers = num_layers - 2
    hidden_layers = layers[1:-1]
    params = [layer.weight.detach() for layer in hidden_layers]
    gradients = [layer.weight.grad for layer in hidden_layers]
    if num_hidden_layers < config['n_layers_per_network'] - 2: # zero pad the difference 
        zero_pad = [torch.zeros((config['n_nodes_per_layer'], config['n_nodes_per_layer']), dtype=torch.float32)] * (max_num_hidden_layers - num_hidden_layers)
        zero_pad_tensor = torch.stack(zero_pad)
        if len(params) > 0 and len(gradients) > 0:
            params = torch.stack(params)
            gradients = torch.stack(gradients)
            params = torch.cat((params, zero_pad_tensor))
            gradients = torch.cat((gradients, zero_pad_tensor)) 
        else:
            params = zero_pad_tensor
            gradients = zero_pad_tensor
    else:
        params = torch.stack(params)
        gradients = torch.stack(gradients)
    assert params.shape==(max_num_hidden_layers, config['n_nodes_per_layer'], config['n_nodes_per_layer']), f"[ERROR] Expected params shape={max_num_hidden_layers, config['n_nodes_per_layer'], config['n_nodes_per_layer']}, got {params.shape}"
    return params.view(-1), gradients.view(-1)

class InnerNetworkAction(Enum):
    TRAIN = 0
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

def get_params_and_gradients(layers):
    num_layers = len(layers)
    max_num_hidden_layers = config['n_hidden_layers_per_network']
    num_hidden_layers = num_layers - 2
    hidden_layers = layers[1:-1]
    params = [layer.weight.detach() for layer in hidden_layers]
    gradients = [layer.weight.grad for layer in hidden_layers]
    if num_hidden_layers < config['n_hidden_layers_per_network']:
        zero_pad = [torch.zeros((config['n_nodes_per_layer'], config['n_nodes_per_layer']), dtype=torch.float32)] * (max_num_hidden_layers - num_hidden_layers)
        zero_pad_tensor = torch.stack(zero_pad)
        if len(params) > 0 and len(gradients) > 0:
            params = torch.stack(params)
            gradients = torch.stack(gradients)
            params = torch.cat((params, zero_pad_tensor))
            gradients = torch.cat((gradients, zero_pad_tensor)) 
        else:
            params = zero_pad_tensor
            gradients = zero_pad_tensor
    else:
        params = torch.stack(params)
        gradients = torch.stack(gradients)
    assert params.shape==(max_num_hidden_layers, config['n_nodes_per_layer'], config['n_nodes_per_layer']), f"[ERROR] Expected params shape={max_num_hidden_layers, config['n_nodes_per_layer'], config['n_nodes_per_layer']}, got {params.shape}"
    return params.view(-1), gradients.view(-1)

def get_latent_space(latent_space):
    flattened_latent_space = latent_space.view(-1)
    flattened_size = flattened_latent_space.numel()
    target_size = config['batch_size'] * config['n_nodes_per_layer']
    if flattened_size < target_size:
        num_elements_to_pad = target_size - flattened_size
        padding_tensor = torch.zeros(num_elements_to_pad)
        padded_tensor = torch.cat((flattened_latent_space, padding_tensor), dim=0)
        return padded_tensor
    else:
        return flattened_latent_space

class InnerNetwork(gymnasium.Env, torch.nn.Module):
    def __init__(self, 
                task: InnerNetworkTask,
                layer_pool: LayerPool,
                epoch: int=0,
                in_features: int=config['in_features'],
                out_features: int=config['out_features'],
                learning_rate: float=config['learning_rate'],
                batch_size: int=config['batch_size'],
                action_cache_size: float=config['action_cache_size'],
                num_workers: int=config['num_workers'],
                shuffle: bool=True,
                ):
        super(InnerNetwork, self).__init__()
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.layer_pool = layer_pool
        self.task = task
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_cache_size = action_cache_size
        self.num_workers = num_workers
        self.prev = defaultdict(lambda: None)
        self.curr = defaultdict(lambda: None)
        self.data_loader = DataLoader(task, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iter = iter(self.data_loader)
        self.initial_layer = random.choice([layer for layer in self.layer_pool.layers if getattr(layer, 'in_features', None) == in_features])
        self.final_layer = random.choice([layer for layer in self.layer_pool.layers if getattr(layer, 'out_features', None) == out_features])
        self.layers = torch.nn.ModuleList([self.initial_layer, self.final_layer]) 
        self.layers_pool_indices = [self.layer_pool.layers.index(layer) for layer in self.layers] 
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        self.actions_taken = [InnerNetworkAction.TRAIN] * config['action_cache_size']
        self.timestep = 0
        self.epoch_reward = 0 
        self.cum_loss = 0
        self.cum_reward = 0
        self.errors = 0
        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.eval()
        self.next_batch()
        self.run_inner_network()
        self.observation_space = gymnasium.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=self.build_state().shape)
        self.action_space = gymnasium.spaces.discrete.Discrete(len(self.layer_pool.layers) + 1)
        self.termination = False

    def step(self, action: np.int64) -> Tuple[torch.Tensor, float, bool, dict]: 
        assert action.shape == (), f'[ERROR] Expected action shape () for scalar {self.action_space.n}, got: {action.shape}'
        assert action.dtype == np.int64, f'[ERROR] Expected np.int64 dtype, got: {action.dtype}'

        self.timestep += 1
        self.train()
        self.next_batch()
        self.update_inner_network(action)
        self.run_inner_network()
        s_prime = self.build_state()
        reward = self.simple_reward()
        # reward = self.simple_reward()
        # reward = self.unregularized_reward()
        self.update_internal_trackers()
        self.log()

        return (
            s_prime,
            reward, 
            self.termination,
            False,
            {}
        )
    
    def decide_action_type(self, action: np.int64) -> np.int64:
        add_action_type = action < self.layer_pool.size
        if (add_action_type):
            if (len(self.layers)==config['n_layers_per_network']):
                self.curr['action_type'] = InnerNetworkAction.ERROR
            elif self.layer_pool.layers[action].in_features==self.in_features or self.layer_pool.layers[action].out_features==self.out_features:
                self.curr['action_type'] = InnerNetworkAction.ERROR
            # limitation of layer pool design: can only use each layer once if updated the pool after
            elif self.layer_pool.layers[action] in self.layers:
                self.curr['action_type'] = InnerNetworkAction.ERROR
            else:
                self.curr['action_type'] = InnerNetworkAction.ADD
        else:
            self.curr['action_type'] = InnerNetworkAction.TRAIN
        self.actions_taken.append(self.curr['action_type']) 
        self.termination = True if config['episodic'] and self.curr['action_type']==InnerNetworkAction.ERROR else False
        return action 
    
    def update_inner_network(self, action: np.int64) -> None:
        index = self.decide_action_type(action)
        if (self.curr['action_type']==InnerNetworkAction.ADD): 
            new_layer = self.layer_pool.layers[index]
            final_layer = self.layers.pop(-1) 
            final_layer_index = self.layers_pool_indices.pop(-1)
            self.layers.append(new_layer)
            self.layers.append(final_layer) 
            self.layers_pool_indices.append(index)
            self.layers_pool_indices.append(final_layer_index)
            
    def next_batch(self, throw_exception=False) -> None:
        self.prev = self.curr
        self.curr = defaultdict(lambda: None)

        if (throw_exception):
            batch = next(self.data_iter)
            self.curr['x'] = batch['x'].view(-1, 1)
            self.curr['y'] = batch['y'].view(-1, 1)
            self.curr['info'] = batch['info']
        else: 
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_loader = DataLoader(self.task, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)
            finally:
                self.curr['x'] = batch['x'].view(-1 ,1)
                self.curr['y'] = batch['y'].view(-1, 1)
                self.curr['info'] = batch['info']
    
    def run_inner_network(self) -> None: 
        if self.training:
            self.train()
            if self.curr['action_type']==InnerNetworkAction.ADD and len(self.layers) < config['n_layers_per_network']:
                self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate) 
            self.opt.zero_grad()
            self.forward(self.curr['x'])
            loss = self.curr['loss']
            loss.backward()
            params_before = [param.data.clone() for param in self.layers.parameters()]
            self.opt.step()
            params_after = [param.data.clone() for param in self.layers.parameters()]
            for layers_index, (pool_index, param_before, param_after) in enumerate(zip(self.layers_pool_indices, params_before, params_after)):
                if torch.all(torch.eq(param_before, param_after)):
                    print('[INFO] Layer in environment not updated.')
                    # print(f'layer pool layer {pool_index} not updated')
                    # print(f'layers layer {self.layers[layers_index]}')
                    # print(f'weights, gradients {self.layers[layers_index].weight}, {self.layers[layers_index].weight.grad}')
        else:
            self.forward(self.curr['x'])

    def forward(self, x) -> torch.tensor:
        for i in range(len(self.layers) - 1): 
            x = torch.nn.functional.relu(self.layers[i](x))
        self.curr['latent_space'] = x
        self.curr['y_hat'] = self.layers[-1](x) 
        y = self.curr['y']
        self.curr['loss'] = self.loss_fn(y, self.curr['y_hat'])
        return self.curr['y_hat']
    
    def build_state(self) -> np.ndarray:
        task_info = torch.tensor([self.task.info['amp'], self.task.info['phase_shift']]).squeeze()
        loss = torch.Tensor([self.curr['loss']])
        latent_space = get_latent_space(self.curr['latent_space'])
        _, gradients = get_params_and_gradients(self.layers)
        one_hot_layers = torch.tensor(np.array([1 if self.layer_pool.layers[i] in self.layers else 0 for i in range(len(self.layer_pool.layers))]))
        h = torch.tensor([action_enum.value for action_enum in self.actions_taken[-self.action_cache_size:]])
        layer_indices = self.layers_pool_indices.copy()
        while len(layer_indices) < config['n_layers_per_network']:
            layer_indices.insert(0, 0)
        layer_indices = torch.tensor(layer_indices)
        
        return torch.concat((
            task_info,
            latent_space,
            layer_indices,
            loss,
            gradients,
            one_hot_layers,
            h
        ), dim=0).detach().numpy()
    
    def simple_reward(self) -> torch.Tensor:
        self.curr['reward'] = self.curr['loss']
        return self.curr['reward']
    
    def regularized_reward(self) -> torch.Tensor:
        prev_loss = self.prev['loss'] or None
        curr_loss = self.curr['loss']
        loss_delta = prev_loss - curr_loss if prev_loss is not None else curr_loss
        if (self.curr['action_type'] == InnerNetworkAction.ERROR):
            reward = -5
        elif (self.curr['action_type'] == InnerNetworkAction.TRAIN):
            reward = 1 if loss_delta > 0 else -1
        else:
            reward = 1 * (0.99 ** self.timestep) if loss_delta > 0 else -1 * (0.99 ** self.timestep)
        self.curr['reward'] = reward
        return reward
    
    def unregularized_reward(self) -> torch.Tensor:
        prev_loss = self.prev['loss'] or None
        curr_loss = self.curr['loss']
        loss_delta = prev_loss - curr_loss if prev_loss is not None else curr_loss
        if (self.curr['action_type'] == InnerNetworkAction.ERROR):
            reward = -5
        elif (self.curr['action_type'] == InnerNetworkAction.TRAIN):
            reward = 1 if loss_delta > 0 else -1
        else:
            reward = 1 * (0.99 ** self.timestep) if loss_delta > 0 else -1 * (0.99 ** self.timestep)
        self.curr['reward'] = reward
        return reward

    def update_internal_trackers(self) -> None:
        self.cum_loss += self.curr['loss']
        self.cum_reward += self.curr['reward']
        self.epoch_reward += self.curr['reward']
        if config['episodic'] and self.curr['action_type']==InnerNetworkAction.ERROR:
            self.steps_per_episode.append(self.timestep)
            self.rewards_per_episode.append(self.cum_reward)
            self.errors += 1

    def log(self):
        # task_num = str(self.curr['info']['i'].item())
        task_num = 0
        if self.timestep%100==0 and not config['episodic']:
            wandb.log({ f'running_loss_task{task_num}_per_100steps' : self.cum_loss})
            wandb.log({ f'running_reward_task{task_num}_per_100steps' : self.cum_reward})
            self.cum_loss = 0
            self.cum_reward = 0
        wandb.log({ f'pool_indices_task{task_num}_per_step' : wandb.Histogram(torch.tensor(self.layers_pool_indices))})
        wandb.log({ f'action_types_task{task_num}_per_step' : wandb.Histogram(torch.tensor([e.value for e in self.actions_taken]))})
        wandb.log({ f'num_layers_task{task_num}_per_step' : len(self.layers) })

    def reset(self, seed=None) -> np.ndarray:
        # print(f'[INFO] Reset at {self.timestep}')
        self.timestep = 0
        self.cum_reward = 0
        self.cum_loss = 0
        self.eval()
        self.next_batch()
        self.run_inner_network()
        return self.build_state(), None

class REML:
    def __init__(
        self,
        layer_pool: LayerPool,
        tasks: List[InnerNetworkTask],
        model=config['sb3_model'],
        policy=config['sb3_policy'],
        epochs: int=config['epochs'],
        timesteps: int=config['timesteps'],
        device: str=config['device'],
        log_dir: str=f"./{config['log_dir']}/{config['sb3_model']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        ):
        self.layer_pool = layer_pool
        self.tasks = tasks
        if config['sb3_model']=='PPO':
            model = PPO
        elif config['sb3_model']=='A2C':
            model = A2C
        elif config['sb3_model']=='RecurrentPPO':
            model = RecurrentPPO
        dummy_env = self.make_env(tasks[0], layer_pool)
        self.model = model(policy, dummy_env, tensorboard_log=log_dir, n_epochs=5, batch_size=32, n_steps=32)
        # self.model = model(policy, dummy_env, tensorboard_log=log_dir)
        self.policy = policy
        self.epochs = epochs
        self.timesteps = timesteps
        self.device = device # TODO is to check whether cuda is used as assumed
        self.log_dir = log_dir

    
    def __str__(self) -> str:
        return f'REML(model={self.model}, policy={self.policy})'
    
    def make_env(self, task, epoch=None) -> gymnasium.Env:
        return gymnasium.wrappers.NormalizeObservation(InnerNetwork(task, self.layer_pool, epoch=epoch))

    def train(self):
        # wraps stablebaselines learn() so we call it n * m times
        # n is the number of epochs where we run all m tasks
        # we use the same policy, swapping out envs for the n tasks, m times. 
        for epoch in range(self.epochs):
            print(f'[INFO] Epoch={epoch + 1}/{self.epochs}')

            epoch_reward = 0
            for i, task in enumerate(self.tasks): 
                self.task = task
                print(f'[INFO] Task={i+1}/{len(self.tasks)}')

                # each task gets its own network
                self.env = self.make_env(self.task, epoch=epoch)
                self.model.set_env(self.env)
                self.model.learn(total_timesteps=self.timesteps)

		        # pool update
                for pool_index in self.env.layers_pool_indices:
                    index_within_layers = self.env.layers_pool_indices.index(pool_index)
                    updated_pool_layer = self.env.layers[index_within_layers]
                    self.layer_pool.layers[pool_index] = self.env.layers[index_within_layers]
                    assert updated_pool_layer.in_features==self.layer_pool.layers[pool_index].in_features
                    assert updated_pool_layer.out_features==self.layer_pool.layers[pool_index].out_features
                    env_layer_params = [param.data.clone() for param in self.env.layers.parameters()]
                    corresponding_pool_layer_params = [param.data.clone() for param in torch.nn.ModuleList([self.env.layer_pool.layers[index] for index in self.env.layers_pool_indices]).parameters()]
                    for param_before, param_after in zip(env_layer_params, corresponding_pool_layer_params):
                        if not torch.all(torch.eq(param_before, param_after)):
                            print('[ERROR] Layers in pool not updated.')

                # wandb per task
                if config['episodic']:
                    wandb.log({ f'average_reward_per_episode_task{i}_per_epoch' : sum(self.env.rewards_per_episode) / len(self.env.rewards_per_episode) }) 
                    wandb.log({ f'average_steps_per_episode_task{i}_per_epoch' : sum(self.env.steps_per_episode) / len(self.env.steps_per_episode) }) 
                    wandb.log({ f'errors_per_epoch_task{i}_per_epoch' : self.env.errors })
                
                epoch_reward += self.env.epoch_reward

                # sine curves
                self.generate_sine_curve(epoch=epoch, task=i, image=True, title='training_sine_curves', args={'label' : f'task_{i}'})
                plt.plot(self.task.data, self.task.targets, linestyle='--', label='ground truth')

            # wandb per epoch
            wandb.log({ f'total_reward_per_epoch' : self.env.epoch_reward})
    
    def evaluate_loss_curves(self, steps=100) -> dict:
        # generates loss curve over 'steps' per task

        lossperstep_bytask = defaultdict(lambda: [])

        for task in self.tasks: 
            env = gymnasium.wrappers.NormalizeObservation(InnerNetwork(task, self.layer_pool))
            self.model.set_env(env, force_reset=False)
            obs, _ = env.reset()

            while len(env.layers) < config['n_layers_per_network']:
                action, _ = self.model.predict(obs)
                obs, _, _, _, _ = env.step(action)

            for _ in range(steps):
                action, _ = self.model.predict(obs)
                obs, _, _, _, _ = env.step(action)
                env.next_batch()
                yhats = env.forward(env.curr['x'])
                loss = env.loss_fn(yhats, env.curr['y'] )
                lossperstep_bytask[task].append(loss)

        return lossperstep_bytask

    def generate_sine_curve(self, env=None, data=None, epoch=None, task=None, image=False, new_figures=False, title=None, args=defaultdict()) -> List:
        # generates sine curve after 'env.layers' is full, with option to set env, limit to 
        # subset of env data (for few shot evaluation), and to create png

        if env is not None:
            self.env = env
            self.model.set_env(env, force_reset=False)

        self.env.eval()
        obs, _ = self.env.reset()
        
        while len(self.env.layers)!=config['n_layers_per_network']:
            action, _ = self.model.predict(obs)
            obs, _, _, _, _ = self.env.step(action)
        
        # if data is specified, wrap in new task
        # if data is not specified, the iterator is used over set
        if data is not None:
            dataset = InnerNetworkTask(data=data[:, 0].clone(), targets=data[:, 1].clone(), info=self.task.info)
        else: 
            dataset = self.task

        xs = dataset.data.clone()
        xs = xs.view(len(xs), 1)
        for i in range(len(self.env.layers) - 1): 
            xs = torch.nn.functional.relu(self.env.layers[i](xs))
        yhats = self.env.layers[-1](xs) 

        if new_figures:
            plt.figure()
        plot_title = title if title!=None else f'sine_curve_epoch_{epoch}_task_{task}' if epoch!=None and task!=None else 'sine_curve'
        plot_path = f'{self.log_dir}/{plot_title}.png'  
        plt.plot(dataset.data, [yhat.detach().numpy() for yhat in yhats], **args)
        # plt.plot(dataset.data, dataset.targets, label='ground truth', linestyle='--')
        plt.title(plot_title)
        plt.legend()

        if image:
            plt.savefig(plot_path)
            wandb.log({plot_title: wandb.Image(plot_path)})
       
        xs, yhats = dataset.data, [yhat.detach().numpy() for yhat in yhats]
        return xs, yhats

if __name__ == "__main__":
    tasks = [InnerNetworkTask(data=tasks_data[i], targets=tasks_targets[i], info=tasks_info[i]) for i in range(config['n_tasks'])]
    eval_task = random.choice(list(tasks))
    training_tasks = list(set(tasks) - {eval_task})
    pool = LayerPool(layers=layers) if config['pretrain'] else LayerPool(layers=None)
    reml = REML(layer_pool=pool, tasks=training_tasks)

    reml.train()
    path = f"meta_{config['sb3_model']}_{datetime.datetime.now().strftime('%H-%M')}"
    reml.model.save(path)
