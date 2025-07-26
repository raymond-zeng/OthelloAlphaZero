import numpy as np

def ucb_score(parent, child, exploration_weight=1.0):
    prior_score = exploration_weight * child.prior * np.sqrt(parent.visits) / (1 + child.visits)
    if child.visits == 0:
        return prior_score
    return child.value() + prior_score

class MCTSNode:

    def __init__(self, state, prior, parent=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.results = 0

    def expand(self, action_probs):
        for prob, action in action_probs:
            if action not in self.children:
                new_state = self.state.apply_action(action)
                self.children[action] = MCTSNode(new_state, prob, parent=self)

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
            action = actions[np.argmax(visit_counts)]
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

class MCTS:

    def __init__(self, model, num_simulations=1000, exploration_weight=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
    
    def search(self, state):
        root = MCTSNode(state, 0, parent=None)

        for _ in range(self.num_simulations):
            node = root
            
            while node.children:
                node, action = node.select_child(self.exploration_weight)
            leaf_state = node.state
            cannonical_leaf_state = leaf_state.to_canonical_form()
            
            if not leaf_state.is_terminal():
                probs, value = self.model.predict(cannonical_leaf_state)
                valid_moves = leaf_state.get_valid_moves()
                if node == root:
                    epsilon = 0.25
                    alpha = 0.3
                    noise = np.random.dirichlet(np.full(len(probs), alpha))
                    probs = (1 - epsilon) * probs + epsilon * noise
                probs = probs * valid_moves
                probs /= np.sum(probs)
                action_probs = [(p, i) for i, p in enumerate(probs) if p > 0]
                node.expand(action_probs)
            
            self.backpropagate(node, value)
        action = root.select_action(temperature=0.001, exploration_weight=self.exploration_weight)
        new_node = root.children[action]
        new_node.parent = None
        return action, new_node

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.results += value
            node = node.parent
            value = -value