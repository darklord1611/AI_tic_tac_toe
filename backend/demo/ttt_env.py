# Source: https://github.com/MauroLuzzatto/OpenAI-Gym-TicTacToe-Environment/blob/master/gym-TicTacToe/gym_TicTacToe/envs/tictactoe_env.py

from typing import Tuple
import numpy as np
import pygame

import gym
from gym import spaces



class TicTacToeEnv(gym.Env):
    def __init__(self, board_size=3, player=1):
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size, self.board_size), dtype=int)
        self.players = [1, 2]
        self.player = player
        self.reset()
    
    def _get_obs(self):
        ...

    def _get_valid_actions(self):
        temp = self.state.flatten()
        return [i for i in range(self.board_size ** 2) if temp[i] == 0]

    # observation -> board, actions made by each player
    def reset(self, seed=None):
        self.state: np.ndarray = np.zeros(
            (self.board_size, self.board_size), dtype=int
        )
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info

    def step(self, user_action):
        """
        step function of the tictactoeEnv

        Args:
          Tuple(int, int):
            action (int): integer between 0 - board_size ** 2, each representing a field on the board
            player (int): 1 or 2, representing the player currently playing

        Returns:
          self.state (np.array): state of the current board position, 0 means empty, 1 or 2 are are marked as X and O respectively
          reward (int): reward of the currrent step
          terminated (boolean): true, if the game is finished
          truncated (boolean): true, if the game is truncated
          (dict): empty dict for future game related information
        """
        action, cur_player = user_action

        valid_actions = self._get_valid_actions()

        if not action in valid_actions:
            raise ValueError(f"action '{action}' is not a valid action")
        if not self.action_space.contains(action):
            raise ValueError(f"action '{action}' is not in action_space")

        if not cur_player in self.players:
            raise ValueError(f"player '{cur_player}' is not an allowed player")

        row, col = self.decode_action(action)


        self.state[row, col] = cur_player
        reward = 0
        
        terminated = self._is_winner(cur_player)
        if terminated:
            if cur_player == self.player:
                reward = 1
            else:
                reward = -1
        elif self.check_draw():
            reward = 0
            terminated = True
        

        self.info["players"][cur_player]["actions"].append(action)
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
        
    def check_draw(self):
        return all(self.state.flatten() != 0)


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


if __name__ == "__main__":
    env = gym.envs.make("TTT-v0", board_size=5)