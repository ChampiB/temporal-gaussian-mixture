import time
from tkinter import Tk

import gym
import numpy
import numpy as np
import skimage
from gym.spaces import Box
from matplotlib import pyplot as plt

from zoo.gui.ImageDisplay import ImageDisplay


class ResizeWrapper(gym.ObservationWrapper):
    """
    A wrapper that resizes the observations
    """

    def __init__(self, env, image_shape):
        """
        Create a wrapper that resizes the observations
        :param env: the environment to wrap
        """
        super().__init__(env)
        image_shape = (image_shape[1], image_shape[2], image_shape[0])
        self.image_shape = image_shape
        self.observation_space = Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

    def observation(self, obs):
        """
        Resize each observation
        :param obs: the obs to resize
        :return: the resized observation
        """
        obs = skimage.transform.resize(obs, self.image_shape, anti_aliasing=True)
        if len(obs.shape) == 4:
            obs = numpy.transpose(obs, (0, 3, 1, 2))
            return obs
        return obs
