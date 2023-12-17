import argparse
import math
import copy 
import random
import datetime
from collections import defaultdict
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
from stable_baselines3 import SAC, TD3
import wandb
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee', 'notebook', 'bright'])
plt.rcParams.update({'figure.dpi': '75'})

# configuration
default_config = {
    'seed' : 20,
    'device' : 'cuda',
    'wandb_run' : '',
    'n_runs' : 1,
    'epochs' : 1,
    'timesteps' : 1000,
    'n_x' : 100,
    'n_tasks' : 2,
    'task_min_loss' : defaultdict(lambda: None),
    'task_max_loss' : defaultdict(lambda: None),
    'in_features' : 1,
    'out_features' : 1,
    'n_hidden_layers_per_network' : 1,
    'n_layers_per_network' : 3,
    'n_nodes_per_layer' : 8,
    'batch_size' : 64,
    'learning_rate' : 0.005,
    'nudge_size' : 0.001,
    'action_cache_size' : 5,
    'num_workers' : 0,
    'loss_fn' : torch.nn.MSELoss(),
    'sb3_model' : 'TD3',
    'sb3_policy' : 'MlpPolicy',
    'log_dir' : 'wandb',
    }
config = default_config

parser = argparse.ArgumentParser(description="REML command line")
parser.add_argument('--wandb_run', '-w', type=str, default=default_config['wandb_run'], help='Name of run in wandb', required=False)
parser.add_argument('--device', '-d', type=str, default=default_config['device'], help='Device to run computations', required=False)
parser.add_argument('--n_runs', '-n', type=int, default=default_config['n_runs'], help='Number of runs', required=False)
parser.add_argument('--epochs', '-e', type=int, default=default_config['epochs'], help='Epochs', required=False)
parser.add_argument('--timesteps', '-t', type=int, default=default_config['timesteps'], help='Timesteps', required=False)
parser.add_argument('--sb3_model', '-m', type=str, default=default_config['sb3_model'], help='SB3 model to use', required=False)
parser.add_argument('--sb3_policy', '-p', type=str, default=default_config['sb3_policy'], help='SB3 policy to use', required=False)
parser.add_argument('--log_dir', '-o', type=str, default=default_config['log_dir'], help='Directory to save tensorboard logs', required=False)
parser.add_argument('--n_tasks', type=int, default=default_config['n_tasks'], help='Number of tasks to generate', required=False)
parser.add_argument('--n_layers_per_network', type=int, default=default_config['n_layers_per_network'], help='Number of layers per network', required=False)
parser.add_argument('--nudge_size', type=float, default=default_config['nudge_size'], help='Max magnitude of nudge to make to weights', required=False)
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

# what the pool has become
initial_input_layer = torch.nn.Linear(in_features=config['in_features'], out_features=config['n_nodes_per_layer'])
initial_hidden_layer = torch.nn.Linear(in_features=config['n_nodes_per_layer'], out_features=config['n_nodes_per_layer'])
initial_output_layer = torch.nn.Linear(in_features=config['n_nodes_per_layer'], out_features=config['out_features'])
torch.nn.init.xavier_uniform_(initial_input_layer.weight)
torch.nn.init.xavier_uniform_(initial_hidden_layer.weight)
torch.nn.init.xavier_uniform_(initial_output_layer.weight)

class InnerNetwork(gymnasium.Env, torch.nn.Module):
    def __init__(self, 
                task: InnerNetworkTask,
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
        self.task = task
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_cache_size = action_cache_size
        self.num_workers = num_workers
        
        self.timestep = 0
        self.loss_vals = []
        self.reward_vals = []
        self.prev = defaultdict(lambda: None)
        self.curr = defaultdict(lambda: None)
        self.data_loader = DataLoader(task, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_iter = iter(self.data_loader)

        # TODO is check whether need to min max scale the reward 
        
        self.initial_input_layer = copy.deepcopy(initial_input_layer)
        self.initial_hidden_layer = copy.deepcopy(initial_hidden_layer)
        self.target_layer = copy.deepcopy(initial_hidden_layer)
        self.initial_output_layer = copy.deepcopy(initial_output_layer)
        self.layers = torch.nn.ModuleList([self.initial_input_layer, self.initial_hidden_layer, self.initial_output_layer]) 
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)

        # create oracle (trained hidden layer) for reward signal
        self.train()
        for _ in range(1000):
            self.next_batch()
            self.opt.zero_grad()
            self.forward(self.curr['x'])
            loss = self.loss_fn(self.curr['y'], self.curr['y_hat'])
            loss.backward()
            self.opt.step()
        self.oracle_layer = self.layers[-2]

        # replace oracle with target 
        self.layers[-2] = self.target_layer

        # check initial loss with target
        self.next_batch()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        self.forward(self.curr['x'])
        self.curr['loss'] = self.loss_fn(self.curr['y'], self.curr['y_hat'])
        self.curr['loss'].backward()

        self.observation_space = gymnasium.spaces.box.Box(low=float('-inf'), high=float('inf'), shape=self.build_state().shape)
        # actions are now "steps" so they need to be smaller
        self.action_space = gymnasium.spaces.box.Box(low=-config['nudge_size'], high=config['nudge_size'], shape=(config['n_nodes_per_layer'] * config['n_nodes_per_layer'], ), dtype=np.float32)
        print(f"[INFO] Initialized target network for task")

    def step(self, action: np.int64) -> Tuple[torch.Tensor, float, bool, dict]: 
        self.timestep += 1

        # update target layer
        nudges = action
        print(f"weights before={self.target_layer.weight}")
        print(f"nudges={nudges}")
        weights = self.target_layer.weight.reshape((config['n_nodes_per_layer'] * config['n_nodes_per_layer'], ))
        new_weights = weights.detach().numpy() + nudges
        self.target_layer.weight.data = torch.tensor(new_weights).reshape((config['n_nodes_per_layer'], config['n_nodes_per_layer']))
        print(f"weights after={self.target_layer.weight}")
        print()

        # update gradient
        self.prev = self.curr
        self.curr = defaultdict(lambda: None)
        self.next_batch()
        self.train()
        self.opt.zero_grad()
        self.forward(self.curr['x'])
        self.curr['loss'] = self.loss_fn(self.curr['y'], self.curr['y_hat'])
        self.curr['loss'].backward()

        # get next state
        # get learning signal
        # log data
        s_prime = self.build_state() # layer, gradient, task info
        termination = torch.any(torch.abs(self.target_layer.weight.data) > 3)
        learning_signal = self.learning_signal()
        self.log()

        return (
            s_prime,
            learning_signal, 
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
                self.data_loader = DataLoader(self.task, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)
            finally:
                self.curr['x'] = batch['x'].view(-1 ,1)
                self.curr['y'] = batch['y'].view(-1, 1)
                self.curr['info'] = batch['info']

    def forward(self, x) -> None:
        for i in range(len(self.layers) - 1): 
            x = torch.nn.functional.relu(self.layers[i](x))
        self.curr['y_hat'] = self.layers[-1](x) 
    
    def build_state(self) -> np.ndarray:
        # characterize the task
        task_info = torch.tensor([self.task.info['amp'], self.task.info['phase_shift']]).squeeze()
        loss = torch.Tensor([self.curr['loss']])
        yhat_scale = torch.Tensor([torch.Tensor(torch.max(torch.abs(self.curr['y_hat']))).detach().item()])
        # get gradient for layer being developed
        gradients = [layer.weight.grad for layer in self.layers]
        gradient = gradients[-2].reshape((config['n_nodes_per_layer'] * config['n_nodes_per_layer'], ))
        # get params of layer being developed
        params = self.layers[-2].weight.reshape((config['n_nodes_per_layer'] * config['n_nodes_per_layer'],))
        
        return torch.concat((
            # target 
            params,

            # helpful info
            task_info,
            loss,
            yhat_scale,
            gradient,
        ), dim=0).detach().numpy()
    
    def learning_signal(self) -> torch.Tensor: 
        # TODO: try target network's loss
        # TODO: try difference between deltas rather than difference between layers
        signal = - self.curr['loss']

        loss_fn = torch.nn.MSELoss()
        # signal = - loss_fn(self.target_layer.weight.data, self.oracle_layer.weight.data)
        high_magnitude = torch.any(torch.abs(self.target_layer.weight.data) > 3)
        if high_magnitude:
            signal -= 100
        self.curr['reward'] = signal
        
        print(f"[INFO] signal={signal}")
        return signal
        # TODO: try combined learning signal as (difference from oracle) + (loss)

    def log(self):
        task_num = str(self.task.info['i'])
        self.loss_vals.append(copy.copy(self.curr['loss']))
        self.reward_vals.append(copy.copy(self.curr['reward']))
        wandb.log({ f'loss_task{task_num}_per_step' : self.curr['loss']})
        wandb.log({ f'reward_task{task_num}_per_step' : self.curr['reward']})

    def reset(self, seed=None) -> np.ndarray:
        print(f'[INFO] Reset at {self.timestep}')
        self.initial_input_layer = copy.deepcopy(initial_input_layer)
        self.initial_hidden_layer = copy.deepcopy(initial_hidden_layer)
        self.target_layer = copy.deepcopy(initial_hidden_layer)
        self.initial_output_layer = copy.deepcopy(initial_output_layer)
        self.layers = torch.nn.ModuleList([self.initial_input_layer, self.initial_hidden_layer, self.initial_output_layer]) 
        self.loss_fn = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)

        # create oracle (trained hidden layer) for reward signal
        self.train()
        for _ in range(1000):
            self.next_batch()
            self.opt.zero_grad()
            self.forward(self.curr['x'])
            loss = self.loss_fn(self.curr['y'], self.curr['y_hat'])
            loss.backward()
            self.opt.step()
        self.oracle_layer = self.layers[-2]

        # replace oracle with target 
        self.layers[-2] = self.target_layer

        # check initial loss with target
        self.next_batch()
        self.opt = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        self.forward(self.curr['x'])
        self.curr['loss'] = self.loss_fn(self.curr['y'], self.curr['y_hat'])
        self.curr['loss'].backward()
        return self.build_state(), None

class REML:
    def __init__(
        self,
        tasks: List[InnerNetworkTask],
        model=config['sb3_model'],
        policy=config['sb3_policy'],
        epochs: int=config['epochs'],
        timesteps: int=config['timesteps'],
        device: str=config['device'],
        log_dir: str=f"./{config['log_dir']}/{config['sb3_model']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        ):
        self.tasks = tasks
        if config['sb3_model']=='TD3':
            model = TD3
        elif config['sb3_model']=='SAC':
            model = SAC
        dummy_env = self.make_env(tasks[0])
        self.model = model(policy, dummy_env, tensorboard_log=log_dir)
        self.policy = policy
        self.epochs = epochs
        self.timesteps = timesteps
        self.device = device # TODO is to check whether cuda is used as assumed
        self.log_dir = log_dir
        self.return_epochs = defaultdict(lambda: [])
        self.cumuloss_epochs = defaultdict(lambda: [])

    
    def __str__(self) -> str:
        return f'REML(model={self.model}, policy={self.policy})'
    
    def make_env(self, task, epoch=None) -> gymnasium.Env:
        return gymnasium.wrappers.NormalizeObservation(InnerNetwork(task, epoch=epoch))

    def train(self):
        # wraps stablebaselines learn() so we call it n * m times
        # n is the number of epochs where we run all m tasks
        # we use the same policy, swapping out envs for the n tasks, m times. 

        # to calculate variance
        # e.g., task: [ n: [epoch: [100 values]] ] / array with n rows, epoch columns 
        # where cell @ [nth run][mth epoch] is cumulative loss/reward
        return_taskkey_epochcol = defaultdict(lambda: [])
        cumuloss_taskkey_epochcol = defaultdict(lambda: [])
        for epoch in range(self.epochs):
            print(f'[INFO] Epoch={epoch + 1}/{self.epochs}')
            for i, task in enumerate(self.tasks): 
                self.task = task
                print(f'[INFO] Task={i+1}/{len(self.tasks)}')

                # each task gets its own network
                self.env = self.make_env(self.task, epoch=epoch)
                self.model.set_env(self.env)
                self.model.learn(total_timesteps=self.timesteps)

                # track reward and loss for plots
                self.return_epochs[str(self.task.info['i'])].append(sum(self.env.reward_vals))
                self.cumuloss_epochs[str(self.task.info['i'])].append(sum(self.env.loss_vals))

                # log to wandb
                wandb.log({ f'cumulative_reward_task{i}_per_epoch' : sum(self.env.reward_vals) })
                wandb.log({ f'cumulative_loss_task{i}_per_epoch' : sum(self.env.loss_vals) })

                # sine curves
                self.generate_sine_curve(epoch=epoch, task=i, image=True, title='training_sine_curves', args={'label' : f'task_{i}'}, new_figures=True)
                
        return return_taskkey_epochcol, cumuloss_taskkey_epochcol

    def generate_sine_curve(self, env=None, data=None, epoch=None, task=None, image=False, new_figures=False, title=None, args=defaultdict()) -> List:
        # generates sine curve after 'env.layers' is full, with option to set env, limit to 
        # subset of env data (for few shot evaluation), and to create png

        # if env is not None:
        #     self.env = env
        #     self.model.set_env(env, force_reset=False)

        # self.env.eval()
        # obs, _ = self.env.reset()
        
        # while len(self.env.layers)!=config['n_layers_per_network']:
        #     action, _ = self.model.predict(obs)
        #     obs, _, _, _, _ = self.env.step(action)
        
        # if data is specified, wrap in new task
        # if data is not specified, the iterator is used over set
        if data is not None:
            dataset = InnerNetworkTask(data=data[:, 0].clone(), targets=data[:, 1].clone(), info=self.task.info)
        else: 
            dataset = self.task

        xs, ys = dataset.data.clone(), dataset.targets.clone()
        xs, ys = xs.view(len(xs), 1), ys.view(len(ys), 1)
        for i in range(len(self.env.layers) - 1): 
            xs = torch.nn.functional.relu(self.env.layers[i](xs))
        yhats = self.env.layers[-1](xs) 

        if new_figures:
            plt.figure()
        plot_title = title if title!=None else f'sine_curve_epoch_{epoch}_task_{task}' if epoch!=None and task!=None else 'sine_curve'
        plot_path = f'{self.log_dir}/{plot_title}.png'  
        plt.plot(dataset.data, [yhat.detach().numpy() for yhat in yhats], **args)
        plt.plot(dataset.data, dataset.targets, label='ground truth', linestyle='--')
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

    ########################################################################
    # train 
    ########################################################################

    return_task_runbyepoch = defaultdict(lambda: [])
    cumuloss_task_runbyepoch = defaultdict(lambda: [])
    errors_task_runbyepoch = defaultdict(lambda: [])

    # e.g., return_task_runbyepoch
    #
    # task:      
    #              epoch 1  epoch 2 ... epoch m
    #       run 1  [[return, return, ...] 
    #       run 2   [return, return, ...]
    #        ...    [        ...        ]
    #       run n   [return, return, ...]]

    for n in range(config['n_runs']):     

        # run REML epoch times on all tasks
        print(f"[INFO] n={n+1}")
        path = f"{config['sb3_model']}_{datetime.datetime.now().strftime('%H-%M')}"
        reml = REML(tasks=training_tasks)
        reml.train()
        reml.model.save(path)
        
        # save data to json
        for task in tasks:
            return_task_runbyepoch[str(task.info['i'])].append(reml.return_epochs[str(task.info['i'])])
            cumuloss_task_runbyepoch[str(task.info['i'])].append(reml.cumuloss_epochs[str(task.info['i'])])
            errors_task_runbyepoch[str(task.info['i'])].append(reml.errors_epochs[str(task.info['i'])])
        with open(f'returns_{path}', 'w') as json_file:
            json.dump(return_task_runbyepoch, json_file, indent=4)
        with open(f'cumuloss_{path}', 'w') as json_file:
            json.dump(cumuloss_task_runbyepoch, json_file, indent=4)
        with open(f'errors_{path}', 'w') as json_file:
            json.dump(errors_task_runbyepoch, json_file, indent=4)
        
        # evaluation plots

        # show it trains (json)
        # (1) loss with variance across 5 runs        
        # (2) return with variance across 5 runs      
        # (3) errors with variance across 5 runs      

        # show learning (json)
        # (4) sine waves for 10 tasks              
        # - can use the same design with map from task to data, with only 
        # 1 column and n rows for the n runs
        # (plots made from the json)

        # show transfer learning (model)
        # (5) convergence speed for 10 tasks     

        # show meta learning (model)
        # (6) k-shot learning for evaluation task     
