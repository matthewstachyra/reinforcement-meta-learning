# [x] increasing # of layers
# [] changing ratio of pool size to network depth, to tasks*network depth
# [] add action to add *no* layer 
# [] set termination condition to be when max depth is reached
# [] overwriting updated params in subsequent training tasks
# [] looping back through tasks to give outer-loop opportunity to try different actions/layers
# [] using stablebaselines3 vecenv to run 4 target networks at a time for each target network
# [] using difference in performance w/ and w/o new layer as the loss (rather than just the performance w/ the new layer)
# [] use sum of absolute value of gradient components to see which layers contribute (transfered layers?)
# [] play with replacing 1 layer versus all layers in the target network (as solution to balance training of inner and outer loop)