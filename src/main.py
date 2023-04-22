from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig

config = A3CConfig() 
config = config.training(lr=0.01, grad_clip=30.0) 
config = config.resources(num_gpus=0) 
config = config.rollouts(num_rollout_workers=4) 
config = config.environment("CartPole-v1") 

print(config.to_dict())  

if __name__ == "__main__":
    algo = config.build()  
    algo.train()  