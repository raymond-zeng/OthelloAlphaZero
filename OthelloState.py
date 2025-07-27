import numpy as np
import torch

class OthelloState:
    """
    Encapsulates the state of an Othello game.

    The class manages the board, the current player, and provides methods
    for game logic like finding valid moves and applying actions. It also
    handles conversions between different board representations.
    """
    BOARD_SIZE = 8
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, board=None, to_play=-1):
        """
        Initializes the state.

        Args:
            board: Can be an 8x8 NumPy array, a 64-char string, or a PyTorch tensor.
            to_play (int): The player whose turn it is (1 or -1).
        """
        if board is None:
            # Create the standard starting board if none is provided
            self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
            self.board[3, 3] = 1  # 'o'
            self.board[4, 4] = 1  # 'o'
            self.board[3, 4] = -1 # 'x'
            self.board[4, 3] = -1 # 'x'
        elif isinstance(board, str):
            self.board = self._from_str(board)
        elif isinstance(board, torch.Tensor):
            # Assuming a (C, H, W) or (H, W) tensor
            self.board = board.cpu().numpy()
            if self.board.ndim == 3:
                # If it's a multi-channel tensor, extract the base board
                # This assumes player 1 is channel 0 and player -1 is channel 1
                self.board = self.board[0] - self.board[1]
        else: # Assumes NumPy array
            self.board = board.copy()
            
        self.to_play = to_play
        self._cached_valid_moves = None

    ## ----------------- ##
    ## Core Game Methods ##
    ## ----------------- ##

    def get_valid_moves(self) -> list:
        """Returns a list of valid moves (as integer indices) for the current player."""
        if self._cached_valid_moves is not None:
            return self._cached_valid_moves

        valid_moves = []
        opponent = -self.to_play
        empty_squares = np.argwhere(self.board == 0)

        for r, c in empty_squares:
            for dr, dc in self.DIRECTIONS:
                r_scan, c_scan = r + dr, c + dc
                if self._is_valid(r_scan, c_scan) and self.board[r_scan, c_scan] == opponent:
                    while self._is_valid(r_scan, c_scan) and self.board[r_scan, c_scan] == opponent:
                        r_scan += dr
                        c_scan += dc
                    
                    if self._is_valid(r_scan, c_scan) and self.board[r_scan, c_scan] == self.to_play:
                        valid_moves.append(r * self.BOARD_SIZE + c)
                        break
        
        self._cached_valid_moves = list(set(valid_moves)) # Remove duplicates
        return self._cached_valid_moves

    def apply_action(self, action: int):
        """
        Applies an action and returns a *new* OthelloState for the next turn.
        """
        new_board = self.board.copy()
        opponent = -self.to_play
        r, c = action // self.BOARD_SIZE, action % self.BOARD_SIZE

        new_board[r, c] = self.to_play

        for dr, dc in self.DIRECTIONS:
            r_scan, c_scan = r + dr, c + dc
            pieces_to_flip = []
            while self._is_valid(r_scan, c_scan) and new_board[r_scan, c_scan] == opponent:
                pieces_to_flip.append((r_scan, c_scan))
                r_scan += dr
                c_scan += dc
            
            if self._is_valid(r_scan, c_scan) and new_board[r_scan, c_scan] == self.to_play:
                for fr, fc in pieces_to_flip:
                    new_board[fr, fc] = self.to_play
        
        # Return a new state object for the opponent's turn
        return OthelloState(new_board, -self.to_play)

    def is_terminal(self) -> bool:
        """Checks if the game is over (neither player has a valid move)."""
        if len(self.get_valid_moves()) > 0:
            return False
        
        # If the current player has no moves, check if the opponent has any
        opponent_state = OthelloState(self.board, -self.to_play)
        if len(opponent_state.get_valid_moves()) > 0:
            return False
            
        return True

    def get_game_result(self) -> int | None:
        """Returns the game result (1, -1, 0) if terminal, otherwise None."""
        if not self.is_terminal():
            return None
        
        score = np.sum(self.board)
        if score > 0: return 1
        if score < 0: return -1
        return 0
    
    def get_canonical_board(self):
        """
        Returns the board state from the perspective of the current player.
        The neural network is always trained from the perspective of player 1.
        This method ensures the network always sees the board from its trained viewpoint.
        """
        # If it's player 1's turn (self.to_play = 1), multiplying by 1 changes nothing.
        # If it's player -1's turn, multiplying by -1 flips all pieces,
        # making the board appear as if it were player 1's turn.
        return self.board * self.to_play

    ## ----------------- ##
    ## Format Converters ##
    ## ----------------- ##

    def to_array(self) -> np.ndarray:
        """Returns the board as a NumPy array."""
        return self.board.copy()

    def to_str(self) -> str:
        """Returns the board as a 64-character string."""
        char_map = {1: 'o', -1: 'x', 0: '.'}
        return "".join([char_map[piece] for piece in self.board.flatten()])

    def to_tensor(self, canonical=True, device='cpu') -> torch.Tensor:
        """
        Returns the board as a 3-channel PyTorch tensor for the neural network.
        
        Args:
            canonical (bool): If True, represents the board from the perspective
                              of the current player.
            device (str): The torch device to send the tensor to ('cpu' or 'cuda').
        """
        player = self.to_play if canonical else 1
        board_view = self.board * player

        my_pieces = (board_view == 1).astype(np.float32)
        opponent_pieces = (board_view == -1).astype(np.float32)
        turn_plane = np.full((8, 8), self.to_play if player == 1 else 0, dtype=np.float32)

        return torch.from_numpy(np.stack([my_pieces, opponent_pieces, turn_plane])).to(device)
    
    ## ----------------- ##
    ## Private Helpers   ##
    ## ----------------- ##

    def _is_valid(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def _from_str(self, board_str: str) -> np.ndarray:
        player_map = {'o': 1, 'x': -1, '.': 0}
        board_list = [player_map[c] for c in board_str]
        return np.array(board_list, dtype=np.int8).reshape(self.BOARD_SIZE, self.BOARD_SIZE)
