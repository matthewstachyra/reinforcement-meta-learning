import torch
import torch.nn as nn
import numpy as np
from utils import convert_to_float_and_tensor, \
                  init_layers_with_normal_dist, \
                  update_global_network_params, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
from typing import Tuple

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# One class for both actor and critic
# because the 'Net' has 2 outputs with
# the policy (actor) and the value function (critic)
#
# it is used as the base class for the global network
# that recieves updates from worker networks - each 
# a 'Net' themselves
class A3C(nn.Module):
    def __init__(self, 
                 s_dim: int, 
                 a_dim: int):
        super(A3C, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        # layers
        self.a1 = nn.Linear(s_dim, 200)
        self.mu =  nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)

        # helper function to initialize the weights of the layers
        # uses normal distribution
        init_layers_with_normal_dist([self. a1, self.mu, self.sigma, self.c1, self.v])

        self.distribution = torch.distributions.Normal

    def forward(self, 
                state: gym.Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # relu6 is a modified relu where the output
        # is limited to 6
        a1 = F.relu6(self.a1(state))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001
        c1 = F.relu6(self.c1(state))
        values = self.v(c1)

        # This output represents the output of both the
        # actor and the critic. The mu and sigma are outputs of 
        # the actor. The values is the output for the critic.
        #
        # The action is a sample from a normal distribution with
        # mean 'mu' and standard deviation 'sigma'.
        return mu, sigma, values
    
    def choose_action(self, 
                      s: gym.Sequence[int]) -> np.ndarray:
        '''Returns next conv2d layer of some dimension.
        '''
        self.training = False

        mu, sigma, _ = self.forward(s)

        # 'torch.tensor.view' returns a new with the same data as the 'self'
        # tensor but of a different shape
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        
        return m.sample().numpy()
    
    def loss_func(self, 
                  s: gym.Sequence[int], 
                  a :np.ndarray, 
                  v_t: np.ndarray) -> np.ndarray:
        # 'nn.Module' is put into 'train' mode
        # this controls the behavior of certain layers like
        # dropout and batch norm. For example, dropout is
        # disabled in 'eval' mode. 'Eval' mode is the same
        # as train(False) or setting 'train' mode to false.
        self.train()
        mu, sigma, values = self.forward(s)

        # Temporal difference. The difference between 
        # what we had the value of the state and what
        # the network now outputs as the value of it.
        td = v_t - values

        # 'Critic' loss is the td or temporal-difference to
        # the 2nd power (probably to remove negatives). The
        # temporal-difference is the difference between the 
        # current value of V(t) and the outputted value.
        c_loss = td.pow(2)

        # 'Actor' loss is the log probability times
        # the temporal difference with an exploration term
        # added. 
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 * 0.5 * math.log(2 * math.pi) + torch.log(m.scale) # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v

        # Total loss for the network is the averae of the 'Critic'
        # loss and the 'Actor' loss.
        total_loss = (a_loss + c_loss).mean()

        return total_loss

class Worker(mp.Process):
    '''
    Every worker gets its own instance of an A3C learner and the environment.
    '''
    def __init__(self, 
                 gnet, 
                 opt, 
                 global_ep, 
                 global_ep_r, 
                 res_queue, 
                 name) -> None:
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt

        # instances of A3C and env
        self.lnet = A3C(N_S, N_A) # local network
        self.env = gym.make('Pendulum-v0').unwrapped
    
    def run(self) -> None:
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):

                # this is C-like string formatting
                if self.name == 'w0':
                    self.env.render()

                # v_wrap is a utility function that converts
                # the numpy array to a tensor
                a = self.lnet.choose_action(convert_to_float_and_tensor(s[None, :]))

                # we clip the action beceause Gaussian distribution
                # isn't bounded. This way we keep it in a valid range
                s_, r, done, _ = self.env.step(a.clip(-2, 2))

                # we are 'done' when we reach a specified episode
                # because this is 'continual' learning rather than
                # 'episodic' learning
                if t==MAX_EP_STEP - 1:
                    done = True
                
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1) # normalize
                
                if total_step % UPDATE_GLOBAL_ITER==0 or done: 
                    # sync local networks up to the global network
                    update_global_network_params(self.opt, 
                                                 self.lnet, 
                                                 self.gnet, 
                                                 done, s_, 
                                                 buffer_s, 
                                                 buffer_a,
                                                 buffer_r, 
                                                 GAMMA)
                    
                    if done: 
                        record(self.g_ep,
                               self.g_ep_r,
                               ep_r,
                               self.res_queue,
                               self.name)
                
                s = s_total step += 1
        
        self.res_queue.put(None)

if __name__ == "__main__":
    gnet = A3C(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))

    global_ep = mp.Value('i', 0)  
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # train workers in parallel
    workers = [Worker(gnet, 
                      opt,
                      global_ep,
                      global_ep_r,
                      res_queue,
                      i)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
        [w.join() for w in workers]

        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()
