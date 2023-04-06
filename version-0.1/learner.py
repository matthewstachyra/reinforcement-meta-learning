import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import math, os
from typing import Tuple
from utils import *
from models import A3C

# TODO(ms): decide how to break these classes into files
# TODO(ms): decide good class naming conventions

class Learner: 
    '''Returns a model (LayerTarget) for a task (LayerTask).

    The returned model is built by a series of LayerHandler 
    objects. The architecture of the Learner instance (which
    includes the number of LayerHandlers) is informted by a 
    LearnerInfo object.
    
    Attributes:
        learner_info: ...
        learner_task: ...
        learner_target: ...
        layer_handlers: ...


    '''
    # TODO(ms): decide whether Learner returns an object like LayeredModel
    #           that is a model or whether it has a call to keep returning
    #           a classification for some input with a bool like 'train'
    #           to inform whether it should first learn how to classify
    #           for the target
    pass

class LearnerInfo:
    # architecture
    pass

class LearnerTask:
    # dataset
    pass

class LearnerTarget(nn.Module):
    # model
    pass

class LayerHandler(mp.Process):
    # takes in Model object which is our RL agent returning Layer objects and
    # Global list of Layer objects that Model can access
    pass

class Layer(nn.Module):
    # layer owned by LayerHandler object. LayerHandler object can return
    # the Layer instance. Layer instance inherits architecture from LayerHandler
    # Layer instance is decided by RLLayerEngine
    pass

class HandlerEngine(A3C):
    # RL model to output Layer object for some input (task agnostic) 
    # informed by LearnerInfo object
    pass



