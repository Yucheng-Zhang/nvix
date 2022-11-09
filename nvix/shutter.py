import numpy as np
import jax.numpy as jnp


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
            the window size in the side and up directions

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


def shutter(X, cam, M_model=None, window=None):
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
    M_modelview = cam.M_view if M_model is None else cam.M_view @ M_model
    M_proj = cam.M_proj
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
