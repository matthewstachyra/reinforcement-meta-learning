# What if the dimension of X changes run to run for target networks?
E.g., I have a (100,) dimension X, or 100 data points for all of my tasks. Then on another task I have (50,) dimension or 50 data points for some sine curve.
The network will still output an action because it considers each data point (whether out of 50 or 100) individually and converts that to an action. One consideration is whether the data points represent the tassk well. If we significantly reduce the number of datapoints, then we're not capturing the task well enough for the network to learn. E.g., if we only get half of the sine curve. How can the network know what the curve looks like?

# Does it make sense to feed the whole task into the network at once? 
E.g., the whole sine curve (X, y)
Unsure. One consideration is this reduces the size of your training data by the length of X. So that makes the task harder for the network in some fashion.

# How to decide the size of the layer pool?
One idea is to tie the size to some other hyperparameter. 
We want the layer pool to have enough layers to capture the diversity of the tasks but not too many that the rml can't find good layers for the next task. So one implementation of this idea is to tie the size of the layer pool to the number of tasks. Another thought is to also tie it to the number of layers composed for each target network. 

# How to balance training the *outer-loop* rml and the *inner-loop* target network?
The initial balance skews towards providing the inner-loop more training data. This is because the initial algorithm first select some number of layers to compose via the outer-loop, after which the remaining timesteps are used to train / update the params of the composed layers. No changes are made to these layers (i.e., none are removed or replaced).

Another way to ask this question is *how does the meta-learner change a layer its composed once its composed it*. While it is training, it still doesn't "know" what the best layer to compose is. There is a layer in the pool that is better than at least 1 layer that is already composed in the target network. One approach is split into these phases. Training is focused on learning what layers are best.

- During training the rml class needs to be learning which layers to compose into the target network for the task. The learned mapping is between the latent space for that task and the next layer to add. 
- The idea is to increase the training data for rml. That means providing more opportunity to output actions. Since there is a depth limit (e.g., 5) then for some timesteps (e.g., 10000) the vast, vast, vast majority of steps are to optimize some layers. That skews learning heavily to the inner-loop over the outer-loop
- One design for training is to allow the rml outer-loop to only provide 1 configuration for the target network, which is then trained over some set number of timesteps. However, this does nothing for the training skew described in the point above.
- Another design is to allow more than 1 configuration. With multi-processing, could have >1 instance of TargetNetwork that is trying different layers from within the layer pool. Let's say there are 4 on 4 cores. This potentially provides 4x the data if each instance of TargetNetwork has completely different layers composed into it. 

# If we used vectorized environments (multi-processing), won't each output the same target network (in terms of composed layers) so that we don't get more information?
#TODO

# Why is the state into the agent the latent space?
Because we want to know how well the composed network performs. We want to know how good are these layers in this order.

# What loss function?
One is just the cross-entropy loss of the target network. This answers the question of how is this network performing with all composed layers.
Another is the difference between the loss of a "control" network (before the next layer is added) and the "new" network with the added layer. This answers the question of how good is the single layer as an action. It is more targeted. 

# What are some future extensions?
One is to enable the outer-loop to also control whether to compose another layer, and not just which next layer to compose. This could be another neural network that is a binary classifier that outputs a decision on whether to output another layer to compose or whether to do some other activity (e.g., train the existing set of layers).