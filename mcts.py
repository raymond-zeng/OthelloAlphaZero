import numpy as np

def ucb_score(parent, child, exploration_weight=1.0):
    prior_score = exploration_weight * child.prior * np.sqrt(parent.visits) / (1 + child.visits)
    if child.visits == 0:
        return prior_score
    return child.value() + prior_score

class mcts_node:

    def __init__(self, state, to_play, prior, parent=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.results = 0
        self.to_play = to_play

    def expand(self, action_probs):
        for prob, action in action_probs:
            if action not in self.children:
                new_state = self.state.apply_action(action)
                self.children[action] = mcts_node(new_state, -self.to_play, prob, parent=self)
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.results / self.visits
    
    def select_action(self, temperature=0.001, exploration_weight=1.0):
        if len(self.children) == 0:
            return None
        actions = self.children.keys()
        visit_counts = np.array([child.visits for child in self.children.values()])
        if temperature == 0:
            action = max(visit_counts.items(), key=lambda x: x[1])[0]
            return action
        else:
            probabilities = visit_counts ** (1 / temperature)
            probabilities /= np.sum(probabilities)
            return np.random.choice(actions, p=probabilities)
    
    def select_child(self, exploration_weight=1.0):
        best_score = -np.inf
        best_action = None
        best_child = None
        for action, child in self.children.items():
            score = ucb_score(self, child, exploration_weight)
            if score > best_score:
                best_score = score
                best_child = child
                best_action = action
        return best_child, best_action

class mcts:

    def __init__(self, game, model, num_simulations=1000, exploration_weight=1.0):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
    
    def search(self, state, to_play):
        root = mcts_node(state, to_play, 0, parent=None)
        valid_moves = self.game.get_valid_moves(state, to_play) 
        action_probs, value = self.model.predict(state, to_play)
        action_probs = action_probs * valid_moves
        action_probs /= np.sum(action_probs)
        root.expand(zip(action_probs, valid_moves))

        for _ in range(self.num_simulations):
            node = root
            path = []
            
            while node.children:
                node, action = node.select_child(self.exploration_weight)
                path.append((node, action))
            leaf_state = node.state
            cannonical_state = self.game.canonicalize(leaf_state, node.to_play)

            if not self.game.is_terminal(cannonical_state):
                action_probs, value = self.model.predict(cannonical_state, node.to_play)
                valid_moves = self.game.get_valid_moves(cannonical_state, node.to_play)
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)
                node.expand(zip(action_probs, valid_moves))
            
            self.backpropagate(path, value, node.to_play)
        return root    


    def backpropagate(self, path, value, to_play):
        for node, action in reversed(path):
            node.visits += 1
            if to_play == node.to_play:
                node.results += value
            else:
                node.results -= value