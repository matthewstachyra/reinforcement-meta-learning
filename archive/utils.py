import torch
from torch import nn
import gym
import numpy as np
from nptyping import Float
from typing import List, Tuple

def convert_to_tensor_as_needed(data: torch.Tensor | Tuple | np.ndarray, 
           dtype: Float=np.float32) -> torch.Tensor:
    '''Converts type of 'np_array' to float as needed, and returns 
    'np_array' input as tensor object.
    '''
    if isinstance(data, torch.Tensor):
        return data

    if isinstance(data, Tuple):
        data = data[0]

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)


    # if np_array.dtype != dtype:
    #     np_array = np_array.astype(dtype)


def init_layers_with_normal_dist(layers: List[nn.Linear]) -> None:
    '''Initialize torch nn layers with values from normal
    distribution with mean=0 annd std=0.1.
    '''
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.constant_(layer.bias, 0.)


def update_global_network_params(opt: torch.optim.Adam,
                                 lnet: nn.Module,
                                 gnet: nn.Module,
                                 done: bool,
                                 s_: Tuple[np.ndarray],
                                 bs: Tuple[np.ndarray],
                                 ba: List[np.ndarray],
                                 br: torch.Tensor,
                                 gamma: float) -> None:
    '''Update global network with updates from workers (local networks).
    '''

    if done:
        v_s_ = 0. # value of terminal state
    else:
        _, _, values = lnet.forward(convert_to_tensor_as_needed(s_[None, :]))
        v_s_ = values[-1].data.numpy()[0,0]
    
    buffer_v_target = []
    # traverse the rewards in reverse (i.e., considering
    # the most recent rewards first)
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        # 'np.vstack' stacks the elements vertically, adding a dimension
        # e.g., [1,2, ...] becomes [[1],
        #                          [2],
        #                          ...]
        # and so on.
        convert_to_tensor_as_needed(np.vstack(bs)),
        convert_to_tensor_as_needed(np.array(ba), dtype=np.int64) if ba[0].dtype==np.int64 else convert_to_tensor_as_needed(np.vstack(ba)),
        convert_to_tensor_as_needed(np.array(buffer_v_target)[:, None])
    )

    # set gradients to zero before backpropagation (i.e., updating weights
    # and biases) because pytorch accumulates the gradients
    opt.zero_grad() 
    loss.backward() # calculates gradients and stores them in each tensor
    
    # set the global network's parameters to be the same the 
    # local network's that were passed in
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad

    # optimizer takes "step" based on gradient, where 'step' means
    # update the parameters. So what does 'loss.backward()' do then
    # you might ask?
    opt.step() # updates each tensor based on the gradient stored from 'backward()'

    lnet.load_state_dict(gnet.state_dict())

def record(global_ep,
           global_ep_r,
           ep_r,
           res_queue,
           name) -> None:
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
        res_queue.put(global_ep_r.value)
        print(
            name,
            "Ep:", global_ep.value,
            "| Ep_r: %.0f" % global_ep_r.value,
        ) 

class SharedAdam(torch.optim.Adam):
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 betas=(0.9, 0.99), 
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()