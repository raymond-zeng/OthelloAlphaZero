import numpy as np
import time

class AlphaBetaStrategy:
    """
    An Othello/Reversi AI strategy using NumPy for board representation
    and a Negamax alpha-beta search algorithm.
    
    The board is represented as an 8x8 NumPy array where:
    -  1: Player 'o' (maximizer)
    - -1: Player 'x' (minimizer)
    -  0: Empty square
    """
    
    # --- Constants ---
    PLAYER_O, EMPTY, PLAYER_X = 1, 0, -1
    BOARD_SIZE = 8
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Heuristic weights
    WEIGHT_CORNERS = 45.0
    WEIGHT_MOBILITY = 15.0
    WEIGHT_STABILITY = 25.0
    WEIGHT_PARITY = 15.0
    
    # Positional weights for a simple stability heuristic
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
        # Memoization caches to store results for previously seen states
        self.move_cache = {}
        self.eval_cache = {}
        
        # Player mapping from original character representation
        self.player_map = {'o': self.PLAYER_O, 'x': self.PLAYER_X}
        self.char_map = {v: k for k, v in self.player_map.items()}

    # --- Public Interface ---

    def best_strategy(self, board_str: str, player_char: str, best_move_q, running_flag):
        """
        Main entry point for the strategy.
        Converts input to NumPy arrays and runs the appropriate search.
        """
        board = self._str_to_array(board_str)
        player = self.player_map[player_char]
        
        empty_squares = int(np.sum(board == self.EMPTY))
        
        # Endgame: Use a perfect solver
        if empty_squares <= 12:
            score, move = self._search_driver(board, player, is_endgame=True, max_depth=empty_squares)
            if move != -1:
                best_move_q.value = move
            return

        # Mid-game: Use iterative deepening with heuristics
        for depth in range(1, empty_squares):
            if not running_flag.value:
                break
            
            score, move = self._search_driver(board, player, is_endgame=False, max_depth=depth)
            
            if move != -1:
                best_move_q.value = move

            if abs(score) > 1e5: # Stop if a winning/losing score is found
                break

    # --- Core Search Logic ---

    def _search_driver(self, board: np.ndarray, player: int, is_endgame: bool, max_depth: int):
        """Drives the alpha-beta search from the root, returning the best move."""
        best_move = -1
        best_score = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        
        possible_moves = self._find_all_moves(board, player)
        if not possible_moves:
            return 0, -1

        for move in possible_moves:
            new_board = self._place_move(board, move, player)
            score = -self._alphabeta(new_board, self._get_opponent(player), max_depth - 1, -beta, -alpha, is_endgame)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        return best_score, best_move

    def _alphabeta(self, board: np.ndarray, player: int, depth: int, alpha: float, beta: float, is_endgame: bool) -> float:
        """Negamax implementation of alpha-beta search."""
        board_key = board.tobytes() # Using tobytes() is a fast way to get a hashable key for a NumPy array
        
        if is_endgame and (board_key, player) in self.eval_cache:
            return self.eval_cache[(board_key, player)]
            
        opponent = self._get_opponent(player)
        possible_moves = self._find_all_moves(board, player)
        
        # --- Base Cases ---
        if not possible_moves:
            if not self._find_all_moves(board, opponent):
                return self._final_score(board) * player

            return -self._alphabeta(board, opponent, depth, -beta, -alpha, is_endgame)

        if depth == 0:
            return self._evaluate_board(board, player) if not is_endgame else self._final_score(board) * player

        # --- Recursive Step ---
        best_score = -float('inf')
        for move in possible_moves:
            new_board = self._place_move(board, move, player)
            score = -self._alphabeta(new_board, opponent, depth - 1, -beta, -alpha, is_endgame)
            
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            
            if alpha >= beta:
                break
                
        if is_endgame:
            self.eval_cache[(board_key, player)] = best_score
            
        return best_score

    # --- Heuristic Evaluation ---

    def _evaluate_board(self, board: np.ndarray, player: int) -> float:
        """Calculates a heuristic score for a given board state for the current player."""
        opponent = self._get_opponent(player)
        
        corners = board[[0, 0, 7, 7], [0, 7, 0, 7]] # Fancy indexing works the same in NumPy
        corner_score = np.sum(corners == player) - np.sum(corners == opponent)
        
        player_moves = len(self._find_all_moves(board, player))
        opponent_moves = len(self._find_all_moves(board, opponent))
        mobility_score = 0
        if player_moves + opponent_moves != 0:
            mobility_score = (player_moves - opponent_moves) / (player_moves + opponent_moves)

        player_pieces = np.sum(board == player)
        opponent_pieces = np.sum(board == opponent)
        parity_score = (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
        
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
        return final_score * player

    def _final_score(self, board: np.ndarray) -> float:
        """Calculates the exact score at the end of the game."""
        score = np.sum(board)
        if score == 0: return 0
        return (score / abs(score)) * 1e6

    # --- Board Manipulation Helpers ---
    
    def _is_valid(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def _find_all_moves(self, board: np.ndarray, player: int) -> list:
        board_key = (board.tobytes(), player)
        if board_key in self.move_cache:
            return self.move_cache[board_key]

        opponent = self._get_opponent(player)
        valid_moves = []
        
        # Use np.argwhere to find coordinates of all empty squares
        empty_squares = np.argwhere(board == self.EMPTY)
        
        for r, c in empty_squares:
            for dr, dc in self.DIRECTIONS:
                r_scan, c_scan = r + dr, c + dc
                
                if self._is_valid(r_scan, c_scan) and board[r_scan, c_scan] == opponent:
                    while self._is_valid(r_scan, c_scan) and board[r_scan, c_scan] == opponent:
                        r_scan += dr
                        c_scan += dc
                    
                    if self._is_valid(r_scan, c_scan) and board[r_scan, c_scan] == player:
                        valid_moves.append(r * self.BOARD_SIZE + c)
                        break
        
        self.move_cache[board_key] = valid_moves
        return valid_moves

    def _place_move(self, board: np.ndarray, move: int, player: int) -> np.ndarray:
        """Returns a new board state after a player makes a move."""
        new_board = board.copy() # Use .copy() in NumPy
        opponent = self._get_opponent(player)
        
        r, c = move // self.BOARD_SIZE, move % self.BOARD_SIZE
        new_board[r, c] = player
        
        for dr, dc in self.DIRECTIONS:
            r_scan, c_scan = r + dr, c + dc
            pieces_to_flip = []
            
            while self._is_valid(r_scan, c_scan) and new_board[r_scan, c_scan] == opponent:
                pieces_to_flip.append((r_scan, c_scan))
                r_scan += dr
                c_scan += dc
            
            if self._is_valid(r_scan, c_scan) and new_board[r_scan, c_scan] == player:
                for fr, fc in pieces_to_flip:
                    new_board[fr, fc] = player
                    
        return new_board

    # --- Utility Helpers ---

    def _get_opponent(self, player: int) -> int:
        return -player

    def _str_to_array(self, board_str: str) -> np.ndarray:
        """Converts the 1D string board representation to a 2D NumPy array."""
        board_list = [self.player_map.get(c, self.EMPTY) for c in board_str]
        return np.array(board_list, dtype=np.int8).reshape(self.BOARD_SIZE, self.BOARD_SIZE)