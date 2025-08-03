import time
from tqdm import tqdm
from OthelloState import OthelloState
from AlphaBetaStrategy import AlphaBetaStrategy
from mcts import MCTS 

class Arena:
    """
    A class to pit two AI agents (players) against each other in a series of games.
    It manages the game flow, tracks scores, and ensures fair competition by
    swapping player sides between games.
    """
    def __init__(self, player1, player2, verbose=True):
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose

    def _play_one_game(self, p1_starts=True):
        """
        Plays a single game of Othello between the two players.
        Returns the game result from player1's perspective (1 if p1 wins, -1 if p2 wins, 0 for a draw).
        """
        players = {
            -1: self.player1 if p1_starts else self.player2,
            1: self.player2 if p1_starts else self.player1
        }
        
        current_state = OthelloState() # Creates the starting board
        
        while not current_state.is_terminal():
            current_player_agent = players[current_state.to_play]
            
            # Pass the entire state object to the player
            action = current_player_agent.play(current_state)

            valid_moves = current_state.get_valid_moves()
            if not valid_moves: # Player must pass
                current_state = OthelloState(current_state.to_array(), -current_state.to_play)
                continue
            
            assert action in valid_moves, f"Agent returned an invalid move: {action}"
            current_state = current_state.apply_action(action)
        
        game_result = current_state.get_game_result() * -1
        
        return game_result if p1_starts else -game_result

    def play_games(self, num_games):
        """
        Plays a specified number of games, swapping starting players each time.
        """
        num_games_per_side = num_games // 2
        if num_games_per_side == 0: num_games_per_side = 1 # Play at least one game per side
        
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for _ in tqdm(range(num_games_per_side), desc="Player 1 Starting"):
            result = self._play_one_game(p1_starts=True)
            if result == 1: p1_wins += 1
            elif result == -1: p2_wins += 1
            else: draws += 1
        
        for _ in tqdm(range(num_games_per_side), desc="Player 2 Starting"):
            result = self._play_one_game(p1_starts=False)
            if result == 1: p1_wins += 1      # Result is from P1's perspective, so 1 is a P1 win
            elif result == -1: p2_wins += 1   # -1 is a P2 win
            else: draws += 1
        
        if self.verbose:
            total_games = p1_wins + p2_wins + draws
            print("\n----- FINAL RESULTS -----")
            print(f"Total Games: {total_games}")
            print(f"Player 1 Wins: {p1_wins}")
            print(f"Player 2 Wins: {p2_wins}")
            print(f"Draws: {draws}")
            print("-------------------------")
        return p1_wins, p2_wins, draws

## --------------------------------------------- ##
## Player Wrapper Classes for the Arena          ##
## --------------------------------------------- ##

class AlphaBetaPlayer:
    """
    Wrapper for the AlphaBetaStrategy to make it compatible with the Arena.
    """
    def __init__(self, search_depth=5):
        self.strategy = AlphaBetaStrategy()
        self.search_depth = search_depth

    def play(self, state):
        _, action = self.strategy._search_driver(
            state=state,
            is_endgame=False,
            max_depth=self.search_depth,
        )
        return action

class MCTSPlayer:
    """
    Wrapper for the MCTS engine to make it compatible with the Arena.
    """
    def __init__(self, model, num_simulations=400):
        self.mcts = MCTS(model=model, num_simulations=num_simulations)

    def play(self, state):
        canonical_state = state.get_canonical_board()
        canonical_state = OthelloState(canonical_state, state.to_play if state.to_play == 1 else -state.to_play)
        action, _, _ = self.mcts.search(canonical_state)
        return action