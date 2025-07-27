import os
from mcts import MCTS

class Coach:

    def __init__(self, model, nnet_args):
        self.model = model
        self.mcts = MCTS(model, num_simulations=nnet_args.get('num_simulations', 1000), exploration_weight=nnet_args.get('exploration_weight', 3.0))
        self.checkpoint_dir = nnet_args.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)