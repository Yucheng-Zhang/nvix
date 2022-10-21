"""
Functions for 3D projection, OpenGL-like.
"""

import numpy as np
import jax.numpy as jnp

import nvix.utils as nvixu


def model():
    """Generate the model matrix: model to world coordinates."""
    return jnp.eye(4)


def view(eye, target, up):
    """Generate the view matrix: world to eye coordinates.

        Parameters
        ----------
        eye: array of 3
            location of the camera
        target: array of 3
            a point in the direction that the center of camera is looking at
        up: array of 3
            up direction of the camera

        Returns
        -------
        M: matrix of 4 x 4

        Notes
        -----
        Ref: http://www.songho.ca/opengl/gl_camera.html
    """
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    # inverse translation
    Mt = np.eye(4)
    Mt[:3, 3] = - eye

    # inverse rotation
    Mr = np.zeros((4, 4))
    f = eye - target
    f = f / np.linalg.norm(f)  # forward
    l = np.cross(up, f)
    l = l / np.linalg.norm(l)  # left
    u = np.cross(f, l)  # up
    Mr[0, :3] = l
    Mr[1, :3] = u
    Mr[2, :3] = f
    Mr[3, 3] = 1

    M = Mr @ Mt
    return jnp.array(M)


def modelview(eye, target, up=(0, 1, 0), M_model=None):
    """Generate the modelview matrix: model to eye coordinates.
        Parameters
        ----------
        eye: array of 3
            location of the camera
        target: array of 3
            a point in the direction that the center of camera is looking at
        up: array of 3
            up direction of the camera
        M_model: matrix of 4 x 4
            model matrix, identity if None

        Returns
        -------
        M: matrix of 4 x 4
    """
    M = view(eye, target, up)
    if M_model is not None:
        M_model = jnp.array(M_model)
        M = M @ M_model
    return M


def projection(fovy, near, far, aspect=1, extent=None, type='persp'):
    """Generate the projection matrix: eye to clip coordinates.

        Parameters
        ----------
        fovy: float
            field of view in the y direction
        near: scalar
            distance of the camera to the near plane, positive
        far: scalar
            distance of the camera to the far plane, positive
        aspect: float
            fovx / fovy
        extent: array of 4
            (left, right, bottom, top) of the near plane
        type: str
            persp or ortho

        Returns
        -------
        M: matrix of 4 x 4

        Notes
        -----
        Ref: http://www.songho.ca/opengl/gl_projectionmatrix.html
    """
    if extent:
        l, r, b, t = extent
    else:
        l, r, b, t = nvixu.fov2extent(fovy, aspect, near)
    n, f = near, far
    M = np.zeros((4, 4))

    if type == 'persp':
        M[0, 0] = 2 * n / (r - l)
        M[0, 2] = (r + l) / (r - l)
        M[1, 1] = 2 * n / (t - b)
        M[1, 2] = (t + b) / (t - b)
        M[2, 2] = - (f + n) / (f - n)
        M[2, 3] = - 2 * f * n / (f - n)
        M[3, 2] = - 1

    if type == 'ortho':
        M[0, 0] = 2 / (r - l)
        M[0, 3] = - (r + l) / (r - l)
        M[1, 1] = 2 / (t - b)
        M[1, 3] = - (t + b) / (t - b)
        M[2, 2] = - 2 / (f - n)
        M[2, 3] = - (f + n) / (f - n)
        M[4, 4] = 1

    return jnp.array(M)


def persp_div(X):
    """Perspective division: clip to normalized device coordinates,
       i.e. return the Euclidean coordinates given the Homogeneous coordinates.

        Parameters
        ----------
        X: array of (4, N)
            homogeneous coordinates with the last row being the weight

        Returns
        -------
        Euclidean coordinates after perspective division
    """
    return X[:-1] / X[-1]


def clip(X):
    """Apply the clipping on the normalized device coordinates.

        Parameters
        ----------
        X: array of (3, N)
            normalized device coordinates
    """
    return X[:, ((X >= -1) & (X <= 1)).all(axis=0)]


def viewport(X, window):
    """Viewport transformation: normalized device to window coordinates.

        Parameters
        ----------
        X: array of (>2, N)
            positions in normalized device coordinates (NDC)
        window: array of 2
            the window size on the x and y axes

        Notes
        -----
        We transform the NDC (-1, 1) [x and y] to
        window coordinates (0, window[0]) [x] and (0, window[1]) [y].
    """
    X = jnp.array(X)
    # shift to (0, 2)
    X = X.at[:2].set(X[:2] + 1)
    # rescale
    X = X.at[0].set(window[0] / 2 * X[0])
    X = X.at[1].set(window[1] / 2 * X[1])
    return X


def shutter(X, eye, target, fovy, near, far, aspect=1, extent=None,
            up=(0, 1, 0), type='persp', M_model=None, window=None):
    """Click the shutter, which combines the tranformations
       and apply to the data.

        Parameters
        ----------
        X: array of (3 or 4, N)
            data to be projected

        Returns
        -------
        X: array of (3, N)
            projected data (first two axes)
    """
    X = jnp.vstack((X, jnp.ones(X.shape[1])))  # to homo coord
    M_modelview = modelview(eye, target, up=up, M_model=M_model)
    M_proj = projection(fovy, near, far, aspect=aspect,
                        extent=extent, type=type)
    X = M_proj @ M_modelview @ X
    X = persp_div(X)
    X = clip(X)
    if window is not None:
        X = viewport(X, window)
    return X


def paint(X, range, pixels=(256, 256)):
    """Paint point data on the image.

        Parameters
        ----------
        X: array of (>2, N)
            point positions to be painted
        range: array of (2, 2)
            range for x and y coords
        pixels: array of 2
            number of pixels for x and y
    """
    H, xedges, yedges = np.histogram2d(
        X[0], X[1], range=range, normed=True, bins=pixels)
    return H
