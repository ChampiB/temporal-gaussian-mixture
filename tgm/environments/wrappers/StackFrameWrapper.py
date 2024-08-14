import collections
import time
from tkinter import Tk

import gym
import numpy
import numpy as np
from gym.spaces import Box

from zoo.gui.ImageDisplay import ImageDisplay


class StackFrameWrapper(gym.ObservationWrapper):
    """
    A wrapper that stack the last n observations to produce the current observation.
    """

    def __init__(self, env, image_shape, n=3):
        """
        Create a wrapper that stacks the last n frame to produce the current observation
        :param env: the environment to wrap
        :param image_shape: the shape of the individual images
        :param n: the number of frames to stack
        """
        super().__init__(env)

        # Fill the queue with black images.
        self.queue = collections.deque(maxlen=n)
        image_shape = (image_shape[1], image_shape[2], image_shape[0])
        for _ in range(n):
            self.queue.append(numpy.zeros(image_shape))

        # Store the observation shape.
        image_shape = (image_shape[0], image_shape[1], image_shape[2] * n)
        self.image_shape = image_shape
        self.observation_space = Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

    def observation(self, obs):
        """
        Resize each observation
        :param obs: the obs to resize
        :return: the resized observation
        """
        self.queue.append(obs)
        return numpy.concatenate([self.queue[i] for i in range(len(self.queue))], axis=-1)
