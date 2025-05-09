import numpy as np
import gym
import pygame
from gym import spaces
from game import Game

class TetrisEnv(gym.Env):
    """
    Gym environment for Tetris, exposing the grid and actions to an RL agent.
    Actions:
      0: move left
      1: move right
      2: rotate
      3: soft drop (one step down)
      4: hard drop (instant placement)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Initialize the Tetris game
        self.game = Game()

        # Infer grid dimensions from the game
        grid = self.game.grid.grid
        self.grid_height = len(grid)
        self.grid_width = len(grid[0])

        # Define action space including hard drop
        self.action_space = spaces.Discrete(5)

        # Observation space: binary occupancy grid
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_height, self.grid_width),
            dtype=np.int8
        )

        # Track score to compute reward deltas
        self.last_score = 0
        self._screen = None

    def reset(self):
        """
        Reset the game to start a new episode. Returns initial observation.
        """
        self.game.reset()
        self.last_score = self.game.score
        return self._get_observation()

    def step(self, action):
        """
        Perform an action in the game, advance one game tick (gravity), and return
        (obs, reward, done, info).
        """
        # Apply player action
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.rotate()
        elif action == 3:
            # Soft drop: move down one
            self.game.move_down()
        elif action == 4:
            # Hard drop: instant placement
            if hasattr(self.game, 'hard_drop'):
                self.game.hard_drop()
            else:
                # fallback: repeatedly move down until locked
                while not self.game.locked:
                    self.game.move_down()

        # Advance gravity / game tick
        self.game.step()

        # Compute reward as change in score (e.g., line clears)
        current_score = self.game.score
        reward = current_score - self.last_score
        self.last_score = current_score

        # Check if game has ended
        done = self.game.game_over

        # Get next observation
        obs = self._get_observation()
        info = {'score': current_score}
        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Render the game using Pygame.
        """
        if self._screen is None:
            pygame.init()
            cell_size = 30  # pixels per cell
            width = self.grid_width * cell_size
            height = self.grid_height * cell_size
            self._screen = pygame.display.set_mode((width, height))
        self._screen.fill((0, 0, 0))
        self.game.draw(self._screen)
        pygame.display.flip()

    def _get_observation(self):
        """
        Retrieve the current board state as a NumPy array of 0s and 1s.
        """
        grid = self.game.grid.grid
        arr = np.array(grid, dtype=np.int8)
        return (arr > 0).astype(np.int8)

    def close(self):
        """
        Clean up resources (e.g., close the Pygame window).
        """
        if self._screen:
            pygame.quit()

    def seed(self, seed=None):
        """
        Seed the environment's RNG for reproducibility.
        """
        self.np_random, seed = spaces.np_random(seed)
        return [seed]
