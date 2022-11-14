"""
Functions for 3D projection, OpenGL-like.
"""

import numpy as np
from dataclasses import dataclass, replace
from numpy.typing import ArrayLike

import nvix.utils as vu
import nvix.matrix as vm


@dataclass(frozen=True)
class Camera:
    """Camera parameters.

    Parameters
    ----------
    eye: array_like
        location of the camera
    target: array_like
        a point in the direction that the center of camera is looking at
    up: array_like
        up direction of the camera
    fov_up: float
        field of view in the up direction of the camera
    aspect: float
        fov_side / fov_up
    near: float
        distance of the camera to the near plane, positive
    far: float
        distance of the camera to the far plane, positive
    """
    eye: ArrayLike
    target: ArrayLike
    up: ArrayLike

    fov_up: float
    aspect: float
    near: float
    far: float

    ptype: str = 'persp'

    def __post_init__(self):
        object.__setattr__(self, 'eye', np.asarray(self.eye))
        object.__setattr__(self, 'target', np.asarray(self.target))
        object.__setattr__(self, 'up', np.asarray(self.up))

    @property
    def M_view(self):
        """The view matrix."""
        return vm.view(self.eye, self.target, self.up)

    @property
    def M_proj(self):
        """The projection matrix."""
        return vm.projection(self.fov_up, self.near, self.far,
                             self.aspect, ptype=self.ptype)


def rotate(cam, axis, a, rad=False):
    """Rotate the camera around axis for angle a."""
    rot = vu.rotation(axis, a, rad)
    eye = rot @ cam.eye
    up = rot @ cam.up
    return replace(cam, eye=eye, up=up)


def spin(cam, a, rad=False):
    """Camera spins for an angle."""
    pass


def shift(cam, disp):
    """Shift the camera by a displacement vector."""
    eye = cam.eye + disp
    return replace(cam, eye=eye)
