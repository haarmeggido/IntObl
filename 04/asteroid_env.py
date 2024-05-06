from typing import Dict, Any, ClassVar, Tuple, Optional

import numpy as np
import pygame
import gym
from gym import spaces
from gym.core import ActType, ObsType


class AstEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 60,
    }

    SIZE_X = 50
    SIZE_Y= 100
    RADIUS = 2

    SHIP_WIDTH = 20
    SHIP_HEIGHT = 20

    MAX_SHIP_SPEED = 2.
    MAX_SCORE = 2000
    HIGH_SCORE_RANGE = 0.2

    PADDING = 50
    WINDOW_SIZE_MODIFIER = 10
    DEFAULT_WIDTH = 5

    def __init__(
            self,
            render_mode = None,
            size = None,
            ship_size = None,
            max_points = None,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f'Unknown render mode: \'{render_mode}\'!')

        self.render_mode = render_mode

        if size is None:
            size = (AstEnv.SIZE_X, AstEnv.SIZE_Y, AstEnv.RADIUS)

        x_size, y_size, radius = size

        if x_size <= 0 or y_size <= 0 or radius <= 0:
            raise ValueError(f"""Game dimensions are incorrect, because they contain negative or zero-like numbers:
                \'{x_size=}\' or \'{y_size}\' or \'{radius}\'!""")

        self._size: np.ndarray = np.array(size, dtype=np.uint16)

        self.observation_space = spaces.Dict({
            "rock_position": spaces.Box(
                np.array([0., 0.], dtype=np.float32),
                np.array([float(x_size), float(y_size)], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            ),
            "rock_velocity": spaces.Box(
                np.array([-1., -1.], dtype=np.float32),
                np.array([1., 1.], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            ),
            "ship_position": spaces.Box(
                0.,
                float(x_size),
                shape=(1,),
                dtype=np.float32
            ),
            "ship_velocity": spaces.Box(
                -AstEnv.MAX_SHIP_SPEED,
                AstEnv.MAX_SHIP_SPEED,
                shape=(1,),
                dtype=np.float32
            ),
        })

        if ship_size is None:
            ship_size = (AstEnv.SHIP_WIDTH, AstEnv.SHIP_HEIGHT)

        ship_width, ship_height = ship_size

        if ship_width <= 0 or ship_height <= 0:
            raise ValueError(
                f"""
                ship dimensions are incorrect, because they contain negative or zero-like numbers:
                \'{ship_width=}\' or \'{ship_height}\'!
                """.lstrip().rstrip()
            )

        self._ship_size: np.ndarray = np.array(ship_size, dtype=np.uint16)

        # Change in speed of the ship
        self.action_space: spaces.Space[ActType] = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        if max_points is None:
            max_points = AstEnv.MAX_SCORE

        if max_points <= 0:
            raise ValueError(f'Max score cannot be negative or zero-like: \'{max_points=}\'!')

        self.window = None
        self.clock = None

        self._rock_position = None
        self._rock_velocity = None
        self._ship_position = None
        self._ship_velocity = None

        self._points = 0.
        self._max_points = float(max_points)
        self.reward_range = (self._points, self._max_points)

    def _get_observations(self):
        return {
            "rock_position": self._rock_position,
            "rock_velocity": self._rock_velocity,
            "ship_position": self._ship_position,
            "ship_velocity": self._ship_velocity,
        }

    def _get_info(self):
        return {

            "ship_left_distance": self._ship_position[0] - 0.,
            "ship_right_distance": float(self._size[0]) - self._ship_position[0],
            "ship_actual_left_distance": self._ship_position[0] - 0. - self._ship_size[0],
            "ship_actual_right_distance": float(self._size[0]) - self._ship_position[0] - self._ship_size[0],
        }

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed, options=options)

        size_x, size_y, radius = self._size
        rng: np.random.Generator = self.np_random
        x = ((rng.random(dtype=np.float32) * size_x + radius) * 0.9)
        y = (size_y) / 10
        self._rock_position = np.array(
            (x, y),
            dtype=np.float32
        )
        self._rock_velocity = np.array(
            [
                rng.random(dtype=np.float32) * 2. - 1.,
                rng.random(dtype=np.float32) * 0.5 + 0.5,
            ],
            dtype=np.float32
        )
        self._ship_position = np.array(
            [self._size.astype(np.float32)[0] / 2.],
            dtype=np.float32
        )
        self._ship_velocity = np.array(
            [0.],
            dtype=np.float32
        )
        self._points = 0.

        observation = self._get_observations()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):
        
        rng: np.random.Generator = self.np_random

        # Cap max ship speed
        self._ship_velocity = np.clip(
            self._ship_velocity + action,
            -AstEnv.MAX_SHIP_SPEED,
            AstEnv.MAX_SHIP_SPEED
        )

        # Do not allow out of bounds for ship
        self._ship_position = np.clip(
            self._ship_position + self._ship_velocity,
            float(self._ship_size[0] / 2.),
            float(self._size[0] - self._ship_size[0] / 2.)
        )

        self._rock_position += self._rock_velocity

        rock_x, rock_y = self._rock_position
        size_x, size_y, radius = self._size

        # Top collision
        if rock_y < radius:
            self._rock_velocity[1] *= -1.
            
        # Left collision and Right collision
        if rock_x < radius or rock_x > size_x - radius:
            self._rock_velocity[0] *= -1.

        game_over = False

        half_ship_length = self._ship_size[0] // 2
        ship_start = self._ship_position - half_ship_length
        ship_end = self._ship_position + half_ship_length

        # Bottom collision
        if rock_y > size_y - radius:
            if ship_start <= rock_x <= ship_end:
                game_over = True
            else:
              x = ((rng.random(dtype=np.float32) * size_x + radius) * 0.9)
              y = (size_y) / 10
              self._rock_position = np.array(
                  (x, y),
                  dtype=np.float32
              )
              self._rock_velocity = np.array(
                  [
                      rng.random(dtype=np.float32) * 2. - 1.,
                      rng.random(dtype=np.float32) * 0.5 + 0.5,
                  ],
                  dtype=np.float32)
                
              self._points += 1

        game_over |= self._points >= self._max_points

        reward = self._points
        # reward = self._points - 50
        observation = self._get_observations()
        info = self._get_info()

        self.render()

        return observation, reward, game_over, False, info

    def render(self) -> None:
        if self.render_mode == "human":
            self._render_frame()
      
    def _render_frame(self) -> None:
        game_size = (self._size * AstEnv.WINDOW_SIZE_MODIFIER)[:2]
        game_x, game_y = game_size
        window_x, window_y = game_size + 2 * AstEnv.PADDING

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_x, window_y))
            pygame.display.set_caption('Asteroid hunter!')
            pygame.display.set_icon(pygame.image.load("spaceship.jpg"))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Draw sky
        canvas = pygame.Surface((window_x, window_y))
        canvas.fill((4,12,36))

        # Draw stars as small rectangles
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.24, game_y * 0.25, 5, 5),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.61, game_y * 0.54, 4,4),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.82, game_y * 0.41, 7,7),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.15, game_y * 0.72, 5,5),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.46, game_y * 0.37, 6,6),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.81, game_y * 0.82, 3,3),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.31, game_y * 0.77, 5,5),)
        pygame.draw.rect(canvas,(250,250,250),pygame.Rect(game_x * 0.90, game_y * 0.10, 6,6),)

        # Draw border
        pygame.draw.lines(
            canvas,
            (50,50,0),
            False,
            [
                (AstEnv.PADDING, AstEnv.PADDING + game_y),
                (AstEnv.PADDING, AstEnv.PADDING),
                (AstEnv.PADDING + game_x, AstEnv.PADDING),
                (AstEnv.PADDING + game_x, AstEnv.PADDING + game_y),
            ],
            width=AstEnv.DEFAULT_WIDTH,
        )

        # Draw ship
        ship_point_position = self._ship_position[0]
        pygame.draw.rect(
            canvas,
            (255,215,0),
            pygame.Rect(
                (ship_point_position - self._ship_size[0] / 2.) * AstEnv.WINDOW_SIZE_MODIFIER + AstEnv.PADDING,
                game_y + AstEnv.PADDING,
                *(self._ship_size * AstEnv.WINDOW_SIZE_MODIFIER)
            ),
        )
        pygame.draw.rect(
            canvas,
            (0,0,0),
            pygame.Rect(
                (ship_point_position - self._ship_size[0] / 2.) * AstEnv.WINDOW_SIZE_MODIFIER + AstEnv.PADDING,
                game_y + AstEnv.PADDING,
                *(self._ship_size * AstEnv.WINDOW_SIZE_MODIFIER)
            ),
            width=AstEnv.DEFAULT_WIDTH,
        )
        pygame.draw.rect(
            canvas,
            (255,100,20),
            pygame.Rect(
                (ship_point_position - self._ship_size[0] / 6.) * AstEnv.WINDOW_SIZE_MODIFIER + AstEnv.PADDING,
                game_y + AstEnv.PADDING - self._ship_size[1] * 3,
                *(self._ship_size * AstEnv.WINDOW_SIZE_MODIFIER * (1/3,1/3))
            ),
        )
        pygame.draw.rect(
            canvas,
            (255,215,0),
            pygame.Rect(
                (ship_point_position - self._ship_size[0] / 2.) * AstEnv.WINDOW_SIZE_MODIFIER + AstEnv.PADDING,
                game_y + AstEnv.PADDING - self._ship_size[1] * 2,
                *(self._ship_size * AstEnv.WINDOW_SIZE_MODIFIER * (1/8, 1))
            ),
        )
        pygame.draw.rect(
            canvas,
            (255,215,0),
            pygame.Rect(
                (ship_point_position + self._ship_size[0] / 2.7) * AstEnv.WINDOW_SIZE_MODIFIER + AstEnv.PADDING,
                game_y + AstEnv.PADDING - self._ship_size[1] * 2 ,
                *(self._ship_size * AstEnv.WINDOW_SIZE_MODIFIER * (1/8, 1))
            ),
        )

        # Draw asteroid, one circle with two lighter circles inside
        ast_x, ast_y = self._rock_position[:2] * AstEnv.WINDOW_SIZE_MODIFIER
        _, _, radius = self._size * AstEnv.WINDOW_SIZE_MODIFIER
        pygame.draw.circle(
            canvas,
            (100, 100, 100),
            (ast_x + AstEnv.PADDING, ast_y + AstEnv.PADDING),
            radius
        )
        pygame.draw.circle(
            canvas,
            (170, 200, 180),
            (ast_x + AstEnv.PADDING + radius * 0.4, ast_y + AstEnv.PADDING - radius * 0.3),
            radius * 0.3
        )
        pygame.draw.circle(
            canvas,
            (200, 190, 200),
            (ast_x + AstEnv.PADDING - radius * 0.3, ast_y + AstEnv.PADDING + radius * 0.2),
            radius * 0.4
        )

      # Draw score in the top left corner in white
        font = pygame.font.Font(None, 36)

        text = font.render(f"Score: {self._points}", True, (240, 240, 240))
        canvas.blit(text, (AstEnv.PADDING, AstEnv.PADDING))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
