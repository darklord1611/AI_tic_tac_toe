# Source: https://github.com/MauroLuzzatto/OpenAI-Gym-TicTacToe-Environment/blob/master/gym-TicTacToe/gym_TicTacToe/envs/tictactoe_env.py

from typing import Tuple
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces



class TicTacToeEnv(gym.Env):
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Discrete(self.board_size * self.board_size)
        self.players = [1, 2]
        self.small = -1
        self.large = self.board_size ** 2 + 1
        self.done = False
        self.reset()
    
    # observation -> board, actions made by each player
    def reset(self):
        self.state: np.ndarray = np.zeros(
            (self.board_size, self.board_size), dtype=int
        )
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info

    def step(self, user_action: np.ndarray) -> np.ndarray[np.ndarray, int, bool, bool, dict]:
        """
        step function of the tictactoeEnv

        Args:
          Tuple(int, int):
            action (int): integer between 0-8, each representing a field on the board
            player (int): 1 or 2, representing the player currently playing

        Returns:
          self.state (np.array): state of the current board position, 0 means empty, 1 or 2 are are marked as X and O respectively
          reward (int): reward of the currrent step
          done (boolean): true, if the game is finished
          (dict): empty dict for future game related information
        """
        action, player = user_action

        if not self.action_space.contains(action):
            raise ValueError(f"action '{action}' is not in action_space")

        if not player in self.players:
            raise ValueError(f"player '{player}' is not an allowed player")

        row, col = self.decode_action(action)

        if self.state[row, col] == 0:
            self.state[row, col] = player
        else:
            return self.state, -10, True, False, self.info

        terminated = self._is_winner(player)

        if terminated:
            reward = 1
        elif self._is_winner(3 - player):
            reward = -1
        

        self.info["players"][player]["actions"].append(action)
        return self.state, reward, terminated, False, self.info
    
    def _is_winner(self, player: int) -> bool:
        """check if there is a winner

        Args:
            color (int): of the player

        Returns:
            bool: indicating if there is a winner
        """
        # check rows
        for row in range(self.board_size):
            if all(self.state[row, :] == player):
                return True
        
        # check columns
        for col in range(self.board_size):
            if all(self.state[:, col] == player):
                return True
        
        # check diagonals
        if all(self.state.diagonal() == player) or all(np.fliplr(self.state).diagonal() == player):
            return True
        
    def check_game_status(self):
        


    def decode_action(self, action: int) -> Tuple[int, int]:
        """decode the action integer into a colum and row value

        0 = upper left corner
        8 = lower right corner

        Args:
            action (int): action

        Returns:
            List[int, int]: a list with the [row, col] values
        """
        col = action % self.board_size
        row = action // self.board_size
        assert 0 <= col < self.board_size
        return [row, col]

    def render(self, mode="human") -> None:
        """render the board

        The following charachters are used to represent the fields,
            '-' no stone
            'O' for player 0
            'X' for player 1

        example:
            ╒═══╤═══╤═══╕
            │ O │ - │ - │
            ├───┼───┼───┤
            │ - │ X │ - │
            ├───┼───┼───┤
            │ - │ - │ - │
            ╘═══╧═══╧═══╛
        """
        # board = np.zeros((3, 3), dtype=str)
        # for ii in range(3):
        #     for jj in range(3):
        #         if self.state[ii, jj] == 0:
        #             board[ii, jj] = "-"
        #         elif self.state[ii, jj] == 1:
        #             board[ii, jj] = "X"
        #         elif self.state[ii, jj] == 2:
        #             board[ii, jj] = "O"

        # if mode == "human":
        #     board = tabulate(board, tablefmt="fancy_grid")
        # return board


if __name__ == "__main__":
    env = gym.envs.make("TTT-v0", board_size=5)