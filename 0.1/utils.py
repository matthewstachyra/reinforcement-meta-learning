from torch import nn
import torch
import numpy as np
from nptyping import Float
from typing import List

def v_wrap(np_array: np.ndarray, 
           dtype: Float=np.float32) -> torch.Tensor:
    '''Returns np_array as tensor object.
    '''

    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)

    return torch.from_numpy(np_array)


def push_and_pull(opt: torch.optim.Adam,
                  lnet: nn.Module,
                  gnet: nn.Module,
                  done: bool,
                  s_: torch.Tensor,
                  bs: List[torch.Tensor],
                  ba: List[np.ndarray],
                  br: torch.Tensor,
                  gamma: float) -> None:
    '''Update global network with updates from workers (local networks).
    '''
    if done:
        v_s_ = 0. # value of terminal state
    else:
        _, _, values = lnet.forward(v_wrap(s_[None, :]))
        v_s_ = values[-1].data.numpy()[0,0]
    
    buffer_v_target = []