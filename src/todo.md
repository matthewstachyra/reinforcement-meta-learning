## Experiments 
# [x] increasing # of layers
# [x] changing ratio of pool size to network depth, to tasks*network depth
# [x] looping back through tasks to give outer-loop opportunity to try different actions/layers (current design is that we create multiple target networks for each task, as we loop through the tasks)
# [] add action to add *no* layer (one design is append nothing when the outputted action is the designated *no* layer)
# [] set termination condition to be when max depth is reached (this would resolve the problem of trying to set a fixed number of training steps; also, network may learn use the *no* layer action more often. however, this could come at the cost of relying on that / re training a small subset of layers over and over)
# [] figure out how to deal with overwriting updated params in subsequent training tasks
# [] figoure what to do whether it makes sense to play with replacing 1 layer versus all layers in the target network (as solution to balance training of inner and outer loop)

## Todo
# [] (efficiency) using stablebaselines3 vecenv to run 4 target networks at a time for each target network
# [] (tweak) using difference in performance w/ and w/o new layer as the loss (rather than just the performance w/ the new layer)
# [] (metrics) use sum of absolute value of gradient components to see which layers contribute (transfered layers?)