from gymnasium.envs.registration import register

register(
    id="gym_examples/TicTacToe-v0",
    entry_point="gym_examples.envs:TicTacToeEnv",
    max_episode_steps=300,
)
