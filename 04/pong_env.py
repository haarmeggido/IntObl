from typing import Dict, Any, ClassVar, Tuple, Optional

import numpy as np
import pygame
import gym
from gym import spaces
from gym.core import ActType, ObsType


class PongEnv(gym.Env):
    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
        ],
        "render_fps": 60,
    }

    SIZE_X: ClassVar[int] = 100
    SIZE_Y: ClassVar[int] = 20
    RADIUS: ClassVar[int] = 2

    PALLET_WIDTH: ClassVar[int] = 15
    PALLET_HEIGHT: ClassVar[int] = 1

    MAX_PALLET_SPEED: ClassVar[float] = 4.
    MAX_SCORE: ClassVar[int] = 25
    HIGH_SCORE_RANGE: ClassVar[float] = 0.2

    PADDING: ClassVar[int] = 30
    WINDOW_SIZE_MODIFIER: ClassVar[int] = 10
    DEFAULT_WIDTH: ClassVar[int] = 5

    def __init__(
            self,
            render_mode: Optional[str] = None,
            size: Optional[Tuple[int, int, int]] = None,
            pallet_size: Optional[Tuple[int, int]] = None,
            max_points: Optional[int] = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f'Unknown render mode: \'{render_mode}\'!')

        self.render_mode = render_mode

        if size is None:
            size = (PongEnv.SIZE_X, PongEnv.SIZE_Y, PongEnv.RADIUS)

        x_size, y_size, radius = size

        if x_size <= 0 or y_size <= 0 or radius <= 0:
            raise ValueError(f"""Game dimensions are incorrect, because they contain negative or zero-like numbers:
                \'{x_size=}\' or \'{y_size}\' or \'{radius}\'!""")

        self._size: np.ndarray = np.array(size, dtype=np.uint16)

        self.observation_space = spaces.Dict({
            "ball_position": spaces.Box(
                np.array([0., 0.], dtype=np.float32),
                np.array([float(x_size), float(y_size)], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            ),
            "ball_velocity": spaces.Box(
                np.array([-1., -1.], dtype=np.float32),
                np.array([1., 1.], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            ),
            "pallet_position": spaces.Box(
                0.,
                float(x_size),
                shape=(1,),
                dtype=np.float32
            ),
            "pallet_velocity": spaces.Box(
                -PongEnv.MAX_PALLET_SPEED,
                PongEnv.MAX_PALLET_SPEED,
                shape=(1,),
                dtype=np.float32
            ),
        })

        if pallet_size is None:
            pallet_size = (PongEnv.PALLET_WIDTH, PongEnv.PALLET_HEIGHT)

        pallet_width, pallet_height = pallet_size

        if pallet_width <= 0 or pallet_height <= 0:
            raise ValueError(
                f"""
                Pallet dimensions are incorrect, because they contain negative or zero-like numbers:
                \'{pallet_width=}\' or \'{pallet_height}\'!
                """.lstrip().rstrip()
            )

        self._pallet_size: np.ndarray = np.array(pallet_size, dtype=np.uint16)

        # Change in speed of the pallet
        self.action_space: spaces.Space[ActType] = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        if max_points is None:
            max_points = PongEnv.MAX_SCORE

        if max_points <= 0:
            raise ValueError(f'Max score cannot be negative or zero-like: \'{max_points=}\'!')

        self.window = None
        self.clock = None

        self._ball_position = None
        self._ball_velocity = None
        self._pallet_position = None
        self._pallet_velocity = None

        self._points: float = 0.
        self._max_points: float = float(max_points)
        self.reward_range = (self._points, self._max_points)

    def _get_observations(self) -> Dict[str, Any]:
        return {
            "ball_position": self._ball_position,
            "ball_velocity": self._ball_velocity,
            "pallet_position": self._pallet_position,
            "pallet_velocity": self._pallet_velocity,

            # "game_size": self._size,
            # "pallet_size": self._pallet_size,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {

            "pallet_left_distance": self._pallet_position[0] - 0.,
            "pallet_right_distance": float(self._size[0]) - self._pallet_position[0],
            "pallet_actual_left_distance": self._pallet_position[0] - 0. - self._pallet_size[0],
            "pallet_actual_right_distance": float(self._size[0]) - self._pallet_position[0] - self._pallet_size[0],
        }

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        rng: np.random.Generator = self.np_random

        self._ball_position = np.array(
            (self._size.astype(np.float32) / 2.)[:2],
            dtype=np.float32
        )
        self._ball_velocity = np.array(
            [
                # rng.random(dtype=np.float32) * 2. - 1.,
                # rng.random(dtype=np.float32) * 0.5 + 0.5,

                1 * 2. - 1.,
                0.5 * 0.5 + 0.5,
            ],
            dtype=np.float32
        )
        self._pallet_position = np.array(
            [self._size.astype(np.float32)[0] / 2.],
            dtype=np.float32
        )
        self._pallet_velocity = np.array(
            [0.],
            dtype=np.float32
        )
        self._points: float = 0.

        observation = self._get_observations()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action: float | np.ndarray) -> Tuple[ObsType, float, bool, bool, dict]:
        # Cap max speed
        self._pallet_velocity = np.clip(
            self._pallet_velocity + action,
            -PongEnv.MAX_PALLET_SPEED,
            PongEnv.MAX_PALLET_SPEED
        )

        # Do not allow out of bounds for pallet
        self._pallet_position = np.clip(
            self._pallet_position + self._pallet_velocity,
            float(self._pallet_size[0] / 2.),
            float(self._size[0] - self._pallet_size[0] / 2.)
        )

        self._ball_position += self._ball_velocity

        ball_x, ball_y = self._ball_position
        size_x, size_y, radius = self._size

        # Top collision
        if ball_y < radius:
            self._ball_velocity[1] *= -1.

            start_range_modifier: float = 0.5 - PongEnv.HIGH_SCORE_RANGE
            end_range_modifier: float = 0.5 + PongEnv.HIGH_SCORE_RANGE

            # if size_x * start_range_modifier <= ball_x <= size_x * end_range_modifier:
            #     self._points += 3.
            # else:
            #     self._points += 1.

        # Left collision and Right collision
        if ball_x < radius or ball_x > size_x - radius:
            self._ball_velocity[0] *= -1.

        game_over: bool = False

        half_pallet_length = self._pallet_size[0] // 2
        pallet_start: float = self._pallet_position - half_pallet_length
        pallet_end: float = self._pallet_position + half_pallet_length

        # Bottom collision
        if ball_y > size_y - radius:
            if pallet_start <= ball_x <= pallet_end:
                self._ball_velocity[1] *= -1.
                
                self._points += 1
            else:
                game_over = True

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
        game_size = (self._size * PongEnv.WINDOW_SIZE_MODIFIER)[:2]
        game_x, game_y = game_size
        window_x, window_y = game_size + 2 * PongEnv.PADDING

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_x, window_y))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_x, window_y))
        canvas.fill((255, 255, 255))

        # Draw border
        pygame.draw.lines(
            canvas,
            (0, 0, 255),
            False,
            [
                (PongEnv.PADDING, PongEnv.PADDING + game_y),
                (PongEnv.PADDING, PongEnv.PADDING),
                (PongEnv.PADDING + game_x, PongEnv.PADDING),
                (PongEnv.PADDING + game_x, PongEnv.PADDING + game_y),
            ],
            width=PongEnv.DEFAULT_WIDTH,
        )

        # Draw high score range
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            (
                game_x * (0.5 - PongEnv.HIGH_SCORE_RANGE) + PongEnv.PADDING,
                PongEnv.PADDING / 2
            ),
            (
                game_x * (0.5 + PongEnv.HIGH_SCORE_RANGE) + PongEnv.PADDING,
                PongEnv.PADDING / 2
            ),
            width=PongEnv.DEFAULT_WIDTH,
        )

        # Draw pallet
        pallet_point_position = self._pallet_position[0]
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                (pallet_point_position - self._pallet_size[0] / 2.) * PongEnv.WINDOW_SIZE_MODIFIER + PongEnv.PADDING,
                game_y + PongEnv.PADDING,
                *(self._pallet_size * PongEnv.WINDOW_SIZE_MODIFIER)
            ),
        )

        # Draw pong
        pong_x, pong_y = self._ball_position[:2] * PongEnv.WINDOW_SIZE_MODIFIER
        _, _, radius = self._size * PongEnv.WINDOW_SIZE_MODIFIER
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (pong_x + PongEnv.PADDING, pong_y + PongEnv.PADDING),
            radius
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
