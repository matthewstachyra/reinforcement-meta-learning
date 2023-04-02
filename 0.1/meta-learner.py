import torch
import torch.nn as nn
import numpy as np
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
from typing import Tuple




class MetaLearner(nn.Module):
    # Purview
    # - Creates and trains 'Learner'(s) for 'Target'
    # - Holds 'Target'(s)
    # - Maintains pool of 'TargetLayer'(s) for 'Target' 
    #   along with their state
    #
    # How it works
    # - It creates a 'Learner' for every dimension
    #   change in the network (i.e., wherever the dimension
    #   of the einput to the next layer is different from
    #   the dimension of the input to the previous layer). This
    #   is pre-decided by the network designer.
    # - The 'Learner' learns a policy for which detached layers
    #   to attach
    # 
    # Attributes
    # - Learner(s)
    # - Target(s)
    # - DetachedLayer(s)
    pass

class Learner(nn.Module):
    # A3C to generate next layer (action) given
    # state of some dimension. These learners are 
    # task agnostic (i.e., it learns to output a
    # 'good' layer of its dimension regardless of
    # the task).
    pass

class LearnerThread(mp.Process):
    # Part of A3C being asynchronous.
    pass

class Target(nn.Module):
    # CNN 
    pass

class DetachedLayer(nn.Module):
    # Layers for 'Target'
    pass

