import numpy as np

import nvix.utils as nvixu


def model():
    """Generate the model matrix: model to world coordinates."""
    return np.eye(4)


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
    return np.array(M)


def modelview(eye, target, up, M_model=None):
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
        M_model = np.array(M_model)
        M = M @ M_model
    return M


def projection(fov_up, near, far, aspect, extent=None, ptype='persp'):
    """Generate the projection matrix: eye to clip coordinates.

        Parameters
        ----------
        fov_up: float
            field of view in the up direction of the camera
        near: scalar
            distance of the camera to the near plane, positive
        far: scalar
            distance of the camera to the far plane, positive
        aspect: float
            fov_side / fov_up
        extent: array of 4
            (left, right, bottom, top) of the near plane
        ptype: str
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
        l, r, b, t = nvixu.fov2extent(fov_up, aspect, near)
    n, f = near, far
    M = np.zeros((4, 4))

    if ptype == 'persp':
        M[0, 0] = 2 * n / (r - l)
        M[0, 2] = (r + l) / (r - l)
        M[1, 1] = 2 * n / (t - b)
        M[1, 2] = (t + b) / (t - b)
        M[2, 2] = - (f + n) / (f - n)
        M[2, 3] = - 2 * f * n / (f - n)
        M[3, 2] = - 1

    if ptype == 'ortho':
        M[0, 0] = 2 / (r - l)
        M[0, 3] = - (r + l) / (r - l)
        M[1, 1] = 2 / (t - b)
        M[1, 3] = - (t + b) / (t - b)
        M[2, 2] = - 2 / (f - n)
        M[2, 3] = - (f + n) / (f - n)
        M[4, 4] = 1

    return np.array(M)
