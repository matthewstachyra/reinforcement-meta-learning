import math
import random
import datetime
from collections import defaultdict
from enum import Enum
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
import gymnasium as gym
from gym import Env
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.utils.env_checker import check_env
from typing import (
    Type,
    OrderedDict,
    List,
    Tuple,
    Callable,
)
import matplotlib.pyplot as plt
import stable_baselines3
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch.utils.tensorboard import SummaryWriter


config = {
    'SEED' : 41,
    'DEVICE' : 'cuda',
    'EPOCHS' : 2,
    'TIMESTEPS' : 20000,
    'N_X' : 100,
    'N_TASKS' : 5,
    'IN_FEATURES' : 1,
    'OUT_FEATURES' : 1,
    'POOL_N_LAYERS' : 30,
    'N_NODES_PER_LAYER' : 32,
    'POOL_LAYER_TYPE' : torch.nn.Linear,
    'ACTION_SPACE_SHAPE' : (3,),
    'EPSILON' : 0.1,
    'BATCH_SIZE' : 1,
    'LEARNING_RATE' : 0.01,
    'ACTION_CACHE_SIZE' : 5,
    'NUM_WORKERS' : 0,
    'LOSS_FN' : torch.nn.MSELoss(),
    'SB3_MODEL' : RecurrentPPO,
    'SB3_POLICY' : 'MlpLstmPolicy',
    }

print(f"[INFO] Running with config: {config}")

# create tasks
# (20, 100) shape
lower_bound = torch.tensor(-5).float()
upper_bound = torch.tensor(5).float()
X = np.linspace(lower_bound, upper_bound, config['N_X'])
amplitude_range = torch.tensor([0.1, 5.0]).float()
phase_range = torch.tensor([0, math.pi]).float()
amps = torch.from_numpy(np.linspace(amplitude_range[0], amplitude_range[1], config['N_TASKS'])).float()
phases = torch.from_numpy(np.linspace(phase_range[0], phase_range[1], config['N_TASKS'])).float()
tasks_data = torch.tensor([ 
        X
        for _ in range(config['N_TASKS'])
        ]).float()
tasks_targets = torch.tensor([
        [((a * np.sin(x)) + p).float()
        for x in X] 
        for a, p in zip(amps, phases)
        ]).float()
tasks_info = [
        [{'i' : i, 
         'amp' : a, 
         'phase_shift' : p, 
         'lower_bound' : lower_bound, 
         'upper_bound' : upper_bound, 
         'amplitude_range_lower_bound' : amplitude_range[0], 
         'amplitude_range_upper_bound' : amplitude_range[1], 
         'phase_range_lower_bound' : phase_range[0],
         'phase_range_lower_bound' : phase_range[1]}
         for _ in X]
        for i, (a, p) in enumerate(zip(amps, phases))
]

print(f"[INFO] {config['N_TASKS']} tasks created.")

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(1, 32),  
            torch.nn.Linear(32, 32), 
            torch.nn.Linear(32, 32),  
            torch.nn.Linear(32, 1)  
        ])

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


layers = []
data = []
for x, y in zip(tasks_data, tasks_targets):
    data.append([x,y])
fig, axs = plt.subplots(5, 2, figsize=(12, 15))
for i, (x, y) in enumerate(data):
    print('[INFO] Training layers for task {i}')
    model = RegressionModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train before graphing
    num_epochs = 20000
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad() 
        outputs = model(x.view(-1, 1))
        loss = criterion(outputs, y.view(-1,1))
        loss.backward()
        optimizer.step()
    
    # get predicitons to graph
    model.eval()
    with torch.no_grad():
        outputs = model(x.view(-1, 1))
        test_loss = criterion(outputs, y.view(-1, 1))

    # save layers for layer pool
    layers.extend(model.layers)

print(f"[INFO] Generated {len(layers)} layeers to pre-initialize a layer pool.")

class Layer:
    def __init__(self, 
                params: Type[torch.nn.Linear]=config['POOL_LAYER_TYPE']):
        self.params = params
        self.used = False
        self.times_used = 0

class LayerPool:
    # pool of uniform Layer objects each with the same type and shape
    def __init__(self, 
                size: int=config['POOL_N_LAYERS'], 
                layer_type: Type[torch.nn.Linear]=config['POOL_LAYER_TYPE'],
                in_features: int=config['IN_FEATURES'],
                out_features: int=config['OUT_FEATURES'],
                num_nodes_per_layer: int=config['N_NODES_PER_LAYER'],
                layers: List[torch.nn.Linear]=None):
        self.size = size if layers is None else len(layers)
        self.layer_type = layer_type
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes_per_layer = num_nodes_per_layer

        # if no layers are passed, create and initialize layer pool from scratch
        if layers is None:
            self.layers = {
                i : Layer(params=self.layer_type(in_features=num_nodes_per_layer, out_features=num_nodes_per_layer))
                for i in range(size)}
            for i in range(config['N_TASKS']):
                self.layers[size + i] = self.layer_type(in_features=in_features, out_features=num_nodes_per_layer)
            for i in range(config['N_TASKS']):
                self.layers[size + i] = self.layer_type(in_features=num_nodes_per_layer, out_features=out_features)
            [torch.nn.init.xavier_uniform_(layer.params.weight) for layer in self.layers.values()]
        else:
            self.layers = {
                i : Layer(params=layer) for i, layer in enumerate(layers)}
            config['POOL_N_LAYERS'] = len(self.layers)
        
    def __str__(self):
        return f"LayerPool(size={self.size}, layer_type={config['POOL_LAYER_TYPE']}, num_nodes_per_layer={config['N_NODES_PER_LAYER']}"

class InnerNetworkAction(Enum):
    NA = 0
    ADD = 1
    DELETE = 2
    ERROR = 3

class InnerNetworkTask(Dataset):
    def __init__(self, data, targets, info):
        self.data = data 
        self.targets = targets
        self.info = info

    def __len__(self):
        assert len(self.data) == config['N_X'], '[ERROR] Length should be the same as N_X.'
        return len(self.data)

    def __getitem__(self, index):
        assert self.data[index].dtype == torch.float32, f'[ERROR] Expected type torch.float32, got type: {self.data[index].dtype}'
        assert self.targets[index].dtype == torch.float32, f'[ERROR] Expected type torch.float32, got type: {self.targets[index].dtype}'
        sample = {
            'x' : self.data[index],
            'y' : self.targets[index],
            'info' : self.info[index]
        }
        return sample
    
    def __str__(self):
        return f'[INFO] InnerNetworkTask(data={self.data, self.targets}, info={self.info})'

class InnerNetwork(gym.Env, Module):
    def __init__(self, 
                task: InnerNetworkTask,
                layer_pool: LayerPool,
                in_features: int=config['IN_FEATURES'],
                out_features: int=config['OUT_FEATURES'],
                learning_rate: float=config['LEARNING_RATE'],
                batch_size: int=config['BATCH_SIZE'],
                epsilon: float=config['EPSILON'],
                action_cache_size: float=config['ACTION_CACHE_SIZE'],
                num_workers: int=config['NUM_WORKERS'],
                shuffle: bool=True,
                log_dir: str='runs',
                ):
        super(InnerNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.layer_pool = layer_pool
        self.task = task
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_loader = DataLoader(task, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iter = iter(self.data_loader)
        self.prev = defaultdict(lambda: None)
        self.curr = defaultdict(lambda: None)

        # need to add initial and final layers for reset() and step() in case action is NA (i.e., train)
        self.initial_layer = random.choice([layer 
                                          for layer in self.layer_pool.layers.values() 
                                          if getattr(layer.params, 'in_features', None) == in_features])
        self.final_layer = random.choice([layer 
                                          for layer in self.layer_pool.layers.values() 
                                          if getattr(layer.params, 'out_features', None) == out_features])

        self.layers = torch.nn.ModuleList([self.initial_layer.params, self.final_layer.params]) 
        self.pool_indices = [] 
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        self.action_cache_size = action_cache_size
        self.actions_taken = [InnerNetworkAction.NA] * config['ACTION_CACHE_SIZE']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.cum_reward = 0
        self.timestep = 0
        self.end_condition = False
        self.state = self.reset()
        state_shape = self.build_state().shape
        self.observation_space = gym.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=state_shape) # TODO is to normalize
        self.action_space = gym.spaces.discrete.Discrete(self.layer_pool.size * 3)

    def step(self, action: np.int64) -> Tuple[torch.Tensor, float, bool, dict]: 
        assert action.shape == (), f'[ERROR] Expected action shape () for scalar {self.action_space.n}, got: {action.shape}'
        assert action.dtype == np.int64, f'[ERROR] Expected np.int64 dtype, got: {action.dtype}'
        self.prev = self.curr
        self.curr = defaultdict(lambda: None)

        # RL updates network or not
        action = self.handle_action(action)
        self.update_inner_network(action)

        # run inner network and update values to build s'
        self.next_batch()
        self.curr['x'] = self.batch['x']
        self.curr['y'] = self.batch['y'] 
        self.curr['info'] = self.batch['info']
        self.forward_inner_network()
        s_prime = self.build_state()

        # calculate reward based on loss
        reward = self.reward()
        self.cum_reward += reward

        # log
        self.timestep += 1
        self.tensorboard_log()

        return (
            s_prime,
            reward, 
            self.end_condition, 
            self.end_condition,
            {}
        )
    
    def handle_action(self, action: np.int64) -> None:
        # take random action with epsilon probability
        if random.random() < self.epsilon: 
            action = random.randint(0, self.action_space.n - 1)

        # ADD action
        if (action < self.layer_pool.size):
            self.curr['action_type'] = InnerNetworkAction.ADD
            self.actions_taken.append(InnerNetworkAction.ADD)
        
        # DELETE action
        elif (action >= self.layer_pool.size and (action < self.layer_pool.size * 2)):
            # adjusting in this discrete action case to calculate pool indices
            adjusted_action_index = action - self.layer_pool.size
            # ERROR cases
            if adjusted_action_index not in self.pool_indices: 
                self.curr['action_type'] = InnerNetworkAction.ERROR
                self.actions_taken.append(InnerNetworkAction.ERROR)
            elif self.layer_pool.layers[adjusted_action_index].params.in_features==self.in_features or self.layer_pool.layers[adjusted_action_index].params.out_features==self.out_features:
                self.curr['action_type'] = InnerNetworkAction.ERROR
                self.actions_taken.append(InnerNetworkAction.ERROR)
            else:
                self.curr['action_type'] = InnerNetworkAction.DELETE
                self.actions_taken.append(InnerNetworkAction.DELETE)

        # NA Action
        else:
            self.curr['action_type'] = InnerNetworkAction.NA
            self.actions_taken.append(InnerNetworkAction.NA)
        
        return action 
    
    def update_inner_network(self, action: np.int64) -> None:
        if (self.curr['action_type']==InnerNetworkAction.ADD): 
            next_layer = self.layer_pool.layers[action].params
            self.pool_indices.append(action)
            # handle new initial layer
            if next_layer.in_features==self.in_features:
                self.layers.pop(0) 
                self.layers.insert(0, next_layer)
            # handle new final layer
            elif next_layer.out_features==self.out_features:
                final_layer = self.layers.pop(-1)
                self.layers.append(final_layer)
            # handle hidden layers
            else:
                final_layer = self.layers.pop(-1) 
                self.layers.append(next_layer)  
                self.layers.append(final_layer) 
        elif (self.curr['action_type']==InnerNetworkAction.DELETE):
            adjusted_action_index = action - self.layer_pool.size
            self.pool_indices.remove(adjusted_action_index)
            layer_to_delete_weights = self.layer_pool.layers[adjusted_action_index].params
            network_index = self.get_layer_index(layer_to_delete_weights)
            assert layer_to_delete_weights == self.layers[network_index], '[ERROR] Wrong layer would be deleted from inner network params.'
            self.layers.pop(network_index)
        elif (self.curr['action_type']==InnerNetworkAction.NA or self.curr['action_type']==InnerNetworkAction.ERROR):
            return # no architecture updates / only train this timestep
        else: 
            raise Exception(f"[ERROR] Unexpected action returned by inner network: {self.curr['action_type']}")

    def get_layer_index(self, target_weights: torch.nn.Linear) -> int:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear) and \
            torch.all(torch.eq(layer.weight, target_weights.weight)) and \
            torch.all(torch.eq(layer.bias, target_weights.bias)):
                return i
        return -1 
            
    def next_batch(self, throw_exception=False) -> None:
        if (throw_exception):
            return next(self.data_iter)
        else: 
            try:
                self.batch = next(self.data_iter)
            except StopIteration:
                self.data_loader = DataLoader(self.task, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
                self.data_iter = iter(self.data_loader)
                self.batch = next(self.data_iter)

    def forward_inner_network(self, train=True) -> None: 

        def _forward():
            x = self.curr['x']
            for i in range(len(self.layers) - 1): x = torch.nn.functional.relu(self.layers[i](x))
            self.curr['latent_space'] = x
            self.curr['y_hat'] = self.layers[-1](x) 
            self.curr['prev_loss'] = self.curr['loss']
            self.curr['loss'] = self.loss_fn(self.curr['y'], self.curr['y_hat'])
            assert self.curr['latent_space'].dtype == torch.float32
            assert self.curr['y_hat'].dtype == torch.float32

        if train:
            self.train() # call before _forward to save gradient data for backward() 
            self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate) 
            self.opt.zero_grad()
            _forward()
            loss = self.curr['loss']
            loss.backward()
            self.opt.step()
        else:
            self.eval()
            _forward()

    def build_state(self) -> np.ndarray:
        one_hot_network_layers = torch.tensor(np.array([1 if self.layer_pool.layers[i].params in self.layers else 0 
                                   for i in range(len(self.layer_pool.layers))]))
        # sequence_network_layers = torch.tensor(np.array([index + 1 for index in self.pool_indices] + [0] * (self.layer_pool.size - len(self.pool_indices))))
        num_add_actions = torch.tensor(len(list(filter(lambda e : e == InnerNetworkAction.ADD, self.actions_taken)))).unsqueeze(0)
        num_delete_actions = torch.tensor(len(list(filter(lambda e : e == InnerNetworkAction.DELETE, self.actions_taken)))).unsqueeze(0)
        num_layers = torch.tensor(len(self.layers)).unsqueeze(0)
        h = torch.tensor([action_enum.value for action_enum in self.actions_taken[-self.action_cache_size:]])
        task_info = torch.tensor([float(value) for value in self.curr['info'].values()])
        return torch.concat((
            # about the task
            task_info,
            self.curr['x'],

            # about the inner network's choices
            self.curr['latent_space'],
            self.curr['y'],
            self.curr['y_hat'],

            # about the meta network's choices
            one_hot_network_layers,
            num_add_actions,
            num_delete_actions,
            num_layers,
            h
        ), dim=0).detach().numpy()
    
    def reward(self) -> torch.Tensor:
        prev_loss = self.curr['prev_loss'] or None
        curr_loss = self.curr['loss']
        loss_delta = prev_loss - curr_loss if prev_loss is not None else curr_loss
        if (self.curr['action_type'] == InnerNetworkAction.ADD):
            reward = loss_delta / math.sqrt(len(self.layers))
        elif (self.curr['action_type'] == InnerNetworkAction.DELETE or self.curr['action_type'] == InnerNetworkAction.NA):
            reward = loss_delta
        elif (self.curr['action_type'] == InnerNetworkAction.ERROR):
            reward = -1000
        else:
            raise Exception('[ERROR] Invalid action type.')
        self.curr['reward'] = reward
        return reward

    def tensorboard_log(self):
        task_num = str(self.curr['info']['i'].item())
        self.writer.add_scalar(f'loss_{task_num}', self.curr['loss'], global_step=self.timestep) 
        self.writer.add_scalar(f'num_layers_{task_num}', len(self.layers), global_step=self.timestep) 
        self.writer.add_scalar(f'reward_{task_num}', self.curr['reward'], global_step=self.timestep)
        if (len(self.pool_indices)!=0):
            self.writer.add_histogram(f'pool_indices_{task_num}', torch.tensor(self.pool_indices).long(), global_step=self.timestep) 
        self.writer.add_histogram(f'action_types_{task_num}', torch.tensor([e.value for e in self.actions_taken]).long(), global_step=self.timestep) 

    def reset(self, seed=None) -> np.ndarray:
        self.next_batch()
        self.curr['x'] = self.batch['x']
        self.curr['y'] = self.batch['y'] 
        self.curr['info'] = self.batch['info']
        self.forward_inner_network(train=False)
        return self.build_state(), None

class REML:
    def __init__(
        self,
        layer_pool: LayerPool,
        tasks: List[InnerNetworkTask],
        model=config['SB3_MODEL'],
        policy=config['SB3_POLICY'],
        epochs: int=config['EPOCHS'],
        timesteps: int=config['TIMESTEPS'],
        device: str=config['DEVICE'],
        overwrite: bool=True,  # TODO is to revisit this param
        intra_update: bool=True, # TODO is to revisit this param
        log_dir: str='runs',
        ):
        self.layer_pool = layer_pool
        self.tasks = tasks
        self.model = model
        self.policy = policy
        self.epochs = epochs
        self.timesteps = timesteps
        self.device = device
        self.overwrite = overwrite 
        self.intra_update = intra_update
        self.log_dir = log_dir
        print(f"[INFO] Created: self.__str__()")
    
    def __str__(self):
        return f'REML(model={self.model}, policy={self.policy})'

    def train(self):
        # wraps stablebaselines learn() so we call it n * m times
        # n is the number of epochs where we run all m tasks
        # we use the same policy, swapping out envs for the n tasks, m times. 
        # therefore the number of steps is (timesteps)*(n)*(m)
        first_run = True
        for epoch in range(self.epochs):
            print(f'[INFO] Epoch {epoch + 1}/{self.epochs}')
            for i, task in enumerate(self.tasks): 
                print(f'[INFO] Task num={i+1}/{len(self.tasks)}')

                # each task gets its own network
                self.env = InnerNetwork(task, self.layer_pool, log_dir=self.log_dir)
                if first_run: 
                    model = self.model(self.policy, 
                                       self.env,
                                       n_steps=config['N_X'],
                                       tensorboard_log=self.log_dir,
                                       verbose=1,)
                    first_run = False
                    
                else: 
                    model.set_env(self.env)

                # train meta learner on task
                model.learn(total_timesteps=self.timesteps, 
                            tb_log_name=f'epoch_{epoch}_task_{i}',
                            reset_num_timesteps=True)

                # epoch log
                self.env.writer.add_scalar(f'loss/epoch_{epoch}_task_{i}', self.env.curr['loss'], global_step=epoch) 
                self.env.writer.add_scalar(f'cumulative_reward/epoch_{epoch}_task_{i}', self.env.cum_reward, global_step=epoch) 
                yhats = self.evaluate_inner_network()
                for x, yhat in zip(self.env.task.data, yhats):
                    self.env.writer.add_scalar(f'sin_curve/epoch_{epoch}', yhat, global_step=x)
                self.env.writer.close()

                # update pool
                # for i in range(len(self.env.pool_indices)):
                #     pool_index = self.env.pool_indices[i]
                #     updated_layer_copy = self.env.layers[i+1]
                #     self.layer_pool.layers[pool_index].params = updated_layer_copy
                #     self.layer_pool.layers[pool_index].used = True
                #     self.layer_pool.layers[pool_index].times_used += 1
    
    def evaluate_inner_network(self):
        self.env.eval()
        y_hats = []
        try:
            # reset data_iter for full dataset 
            self.env.data_loader = DataLoader(self.env.task, batch_size=self.env.batch_size, shuffle=self.env.shuffle, num_workers=self.env.num_workers)
            self.env.data_iter = iter(self.env.data_loader)
            while True:
                batch = self.env.next_batch(throw_exception=True)
                x = batch['x']
                for i in range(len(self.env.layers)-1):
                    x = torch.nn.functional.relu(self.env.layers[i](x))
                x = self.env.layers[-1](x) # do not relu the very last calculation
                y_hats.append(x)
        except StopIteration:
            pass
        return y_hats


def main():
    tasks = [InnerNetworkTask(data=tasks_data[i], targets=tasks_targets[i], info=tasks_info[i]) for i in range(config['N_TASKS'])]
    pool = LayerPool(layers=layers)
    log_dir = f'./runs/ppo_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # train REML
    model = REML(layer_pool=pool, tasks=tasks, log_dir=log_dir)
    model.train()
    
if __name__=="__main__":
    main()
