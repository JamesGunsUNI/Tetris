import os
# Suppress audio by using dummy driver
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import numpy as np
import gym
import pygame
from gym import spaces

# Initialize Pygame without real audio device
pygame.init()
try:
    pygame.mixer.pre_init()
    pygame.mixer.init()
except pygame.error:
    pass

# Monkey-patch Sound and music to no-op to avoid audio initialization in Game
class DummySound:
    def play(self): pass
    def stop(self): pass
pygame.mixer.Sound = lambda *args, **kwargs: DummySound()
pygame.mixer.music.load = lambda *args, **kwargs: None
pygame.mixer.music.play = lambda *args, **kwargs: None
pygame.mixer.music.stop = lambda *args, **kwargs: None

# Now import Game after audio is patched
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
        # Instantiate Game (audio calls are no-op now)
        self.game = Game()

        # Infer grid size
        grid = self.game.grid.grid
        self.grid_height = len(grid)
        self.grid_width = len(grid[0])

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_height, self.grid_width),
            dtype=np.int8
        )

        # Score tracking for reward calculation
        self.last_score = 0
        self._screen = None

    def reset(self):
        self.game.reset()
        self.last_score = self.game.score
        return self._get_observation()

    def step(self, action):
        # Apply chosen action
        if action == 0:
            self.game.move_left()
        elif action == 1:
            self.game.move_right()
        elif action == 2:
            self.game.rotate()
        elif action == 3:
            self.game.move_down()
        elif action == 4:
            # Hard drop if available
            if hasattr(self.game, 'hard_drop'):
                self.game.hard_drop()
            # Otherwise skip fallback

        # Advance gravity / game tick / game tick
        self.game.move_down()

        # Reward = delta score
        current_score = self.game.score
        reward = current_score - self.last_score
        self.last_score = current_score

        # Episode done?
        done = self.game.game_over
        info = {'score': current_score}

        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        if self._screen is None:
            pygame.init()
            cell_size = 30
            w = self.grid_width * cell_size
            h = self.grid_height * cell_size
            self._screen = pygame.display.set_mode((w, h))
        self._screen.fill((0, 0, 0))
        self.game.draw(self._screen)
        pygame.display.flip()

    def _get_observation(self):
        grid = self.game.grid.grid
        arr = np.array(grid, dtype=np.int8)
        return (arr > 0).astype(np.int8)

    def close(self):
        pygame.quit()

    def seed(self, seed=None):
        self.np_random, seed = spaces.np_random(seed)
        return [seed]
