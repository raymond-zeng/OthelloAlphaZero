import numpy as np
import time
from OthelloState import OthelloState

class AlphaBetaStrategy:
    """
    An Othello AI strategy using a Negamax alpha-beta search algorithm.
    This class is responsible for the search logic and delegates all game
    rules and state manipulation to the OthelloState class.
    """
    
    # Heuristic weights remain part of the strategy
    WEIGHT_CORNERS = 45.0
    WEIGHT_MOBILITY = 15.0
    WEIGHT_STABILITY = 25.0
    WEIGHT_PARITY = 15.0
    
    STABILITY_MAP = np.array([
        [100, -20, 10,  5,  5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [ 10,  -2, -1, -1, -1, -1,  -2,  10],
        [  5,  -2, -1,  1,  1, -1,  -2,   5],
        [  5,  -2, -1,  1,  1, -1,  -2,   5],
        [ 10,  -2, -1, -1, -1, -1,  -2,  10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10,  5,  5, 10, -20, 100],
    ], dtype=np.float32)

    def __init__(self):
        # Caches for search efficiency
        self.eval_cache = {}
        # Player mapping from the original problem spec
        self.player_map = {'o': 1, 'x': -1}

    # --- Public Interface ---

    def best_strategy(self, board_str: str, player_char: str, best_move_q, running_flag):
        """
        Main entry point for the strategy. Creates an OthelloState object
        and runs the appropriate search.
        """
        player = self.player_map[player_char]
        initial_state = OthelloState(board_str, to_play=player)
        
        board = initial_state.to_array() # Get array for piece counting
        empty_squares = int(np.sum(board == 0))
        
        # Determine game phase and start search
        if empty_squares <= 12: # Endgame
            _, move = self._search_driver(initial_state, is_endgame=True, max_depth=empty_squares)
        else: # Mid-game (iterative deepening)
            best_move_for_depth = -1
            for depth in range(1, empty_squares):
                if not running_flag.value:
                    break
                
                score, move = self._search_driver(initial_state, is_endgame=False, max_depth=depth)
                if move != -1:
                    best_move_for_depth = move
                
                # Update the shared memory value with the best move found so far for any depth
                best_move_q.value = best_move_for_depth

                if abs(score) > 1e5: # Stop if a forced win/loss is found
                    break
            return # The loop handles updating best_move_q

        if move != -1:
            best_move_q.value = move

    # --- Core Search Logic ---

    def _search_driver(self, state: OthelloState, is_endgame: bool, max_depth: int):
        """Drives the alpha-beta search from the root, returning the best move."""
        best_move = -1
        best_score = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        
        possible_moves = state.get_valid_moves()
        if not possible_moves:
            return 0, -1 # No moves available

        # Move ordering: simple heuristic to check corners first
        possible_moves.sort(key=lambda m: self.STABILITY_MAP[m // 8, m % 8], reverse=True)

        for move in possible_moves:
            next_state = state.apply_action(move)
            score = -self._alphabeta(next_state, max_depth - 1, -beta, -alpha, is_endgame)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_score, best_move

    def _alphabeta(self, state: OthelloState, depth: int, alpha: float, beta: float, is_endgame: bool) -> float:
        """Negamax implementation of alpha-beta search on an OthelloState object."""
        # Use the state's array representation for caching
        board_key = (state.to_array().tobytes(), state.to_play)
        
        if is_endgame and board_key in self.eval_cache:
            return self.eval_cache[board_key]
            
        if state.is_terminal():
            # The game result is from player 1's perspective, so adjust for current player
            return state.get_game_result() * state.to_play * 1e6

        if depth == 0:
            return self._evaluate_board(state)

        possible_moves = state.get_valid_moves()
        
        # Handle a pass move
        if not possible_moves:
            pass_state = OthelloState(state.to_array(), -state.to_play)
            return -self._alphabeta(pass_state, depth, -beta, -alpha, is_endgame)

        # --- Recursive Step ---
        best_score = -float('inf')
        for move in possible_moves:
            next_state = state.apply_action(move)
            score = -self._alphabeta(next_state, depth - 1, -beta, -alpha, is_endgame)
            
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            
            if alpha >= beta:
                break # Beta cutoff
                
        if is_endgame:
            self.eval_cache[board_key] = best_score
            
        return best_score

    # --- Heuristic Evaluation ---

    def _evaluate_board(self, state: OthelloState) -> float:
        """Calculates a heuristic score for a given OthelloState."""
        board = state.to_array()
        player = state.to_play
        opponent = -player

        # Corner Score
        corners = board[[0, 0, 7, 7], [0, 7, 0, 7]]
        corner_score = np.sum(corners == player) - np.sum(corners == opponent)
        
        # Mobility Score
        player_moves = len(state.get_valid_moves())
        # To get opponent moves, we need a temporary state for their turn
        opponent_state = OthelloState(board, opponent)
        opponent_moves = len(opponent_state.get_valid_moves())
        mobility_score = 0
        if player_moves + opponent_moves != 0:
            mobility_score = (player_moves - opponent_moves) / (player_moves + opponent_moves)

        # Parity (Piece Count) Score
        player_pieces = np.sum(board == player)
        opponent_pieces = np.sum(board == opponent)
        parity_score = 0
        if player_pieces + opponent_pieces != 0:
            parity_score = (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
        
        # Stability Score
        player_stability = np.sum(self.STABILITY_MAP[board == player])
        opponent_stability = np.sum(self.STABILITY_MAP[board == opponent])
        stability_score = 0
        if player_stability + opponent_stability != 0:
            stability_score = (player_stability - opponent_stability) / (abs(player_stability) + abs(opponent_stability))

        final_score = (
            self.WEIGHT_CORNERS * corner_score +
            self.WEIGHT_MOBILITY * mobility_score +
            self.WEIGHT_STABILITY * stability_score +
            self.WEIGHT_PARITY * parity_score
        )
        # Score is from the perspective of the current player
        return final_score * player