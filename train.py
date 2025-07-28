import os
from tqdm import tqdm
from OthelloState import OthelloState
from mcts import MCTS
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from arena import Arena

class Coach:

    numIters = 1000
    numEps = 100
    trainHistoryLength = 1000
    maxDequeLength = 10000

    def __init__(self, model, nnet_args):
        self.model = model
        self.pmodel = self.model.__class__()
        self.mcts = MCTS(model, num_simulations=nnet_args.get('num_simulations', 1000), exploration_weight=nnet_args.get('exploration_weight', 3.0))
        self.trainExampleHistory = []
        self.checkpoint_dir = nnet_args.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def execute_episode(self):
        train_examples = []
        current_state = OthelloState()
        episode_step = 0
        while True:
            episode_step += 1
            canonical_state = current_state.get_canonical_board()
            action, _, probs = self.mcts.search(canonical_state)
            train_examples.append((current_state, probs))
            current_state = current_state.apply_action(action)
            if current_state.is_terminal():
                result = current_state.get_game_result()
                if result > -1:
                    return [(state, probs, result ** (state.to_play == current_state.to_play)) for state, probs in train_examples]
    
    def learn(self):
        for i in range(self.numIters):
            iterationTrainExamples = deque()
            for _ in tqdm(range(self.numEps), desc=f"Self Play"):
                self.mcts = MCTS(self.model, num_simulations=1000, exploration_weight=3.0)
                iterationTrainExamples.extend(self.execute_episode())
            
            self.trainExampleHistory.extend(iterationTrainExamples)

            if len(self.trainExampleHistory) > self.trainHistoryLength:
                self.trainExampleHistory.pop()
            
            self.save_train_examples(i - 1)

            train_examples = []
            for e in self.trainExampleHistory:
                train_examples.append(e)
            shuffle(train_examples)

            self.model.save_checkpoint(self.checkpoint_dir, filename = "temp.pth.tar")
            self.pmodel.load_checkpoint(self.checkpoint_dir, filename = "temp.pth.tar")
            pmcts = MCTS(self.pmodel, num_simulations=1000, exploration_weight=3.0)

            arena = Arena(pmcts, self.mcts)
            prev_wins, cur_wins, draws = arena.play_games(num_games=100, verbose=True)
            if cur_wins + draws >=`` 55:
                # Save new model
                self.model.save_checkpoint(self.checkpoint_dir, filename=f"checkpoint_iter_{i}.pth.tar")
                self.model.save_checkpoint(self.checkpoint_dir, filename="best.pth.tar")
            else:
                self.model.load_checkpoint(self.checkpoint_dir, filename="temp.pth.tar")

    def save_train_examples(self, iteration, folder="./train_examples"):
        """
        Save the training examples to a file.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/train_examples_iter_{iteration}.pkl", 'wb') as f:
            Pickler(f).dump(self.trainExampleHistory)
        f.close()
    
    def load_train_examples(self, iteration, folder="./train_examples"):
        """
        Load the training examples from a file.
        """
        with open(f"{folder}/train_examples_iter_{iteration}.pkl", 'rb') as f:
            self.trainExampleHistory = Unpickler(f).load()
        f.close()