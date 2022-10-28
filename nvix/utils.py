import numpy as np


def rotation(axis, a, rad=False):
    """Return the rotation matrix for an angle around a certain axis."""
    if not rad:
        a = a / 180 * np.pi
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def extent2fov(extent, D):
    """Given a distance, convert extent (left, right, bottom, top) to 
       (FoVy, aspect [width / height]).
    """
    l, r, b, t = extent
    w, h = r - l, t - b  # width and height
    aspect = w / h
    fovy = np.arctan2(h/2, D) * 2
    return fovy, aspect


def fov2extent(fovy, aspect, D):
    """Given a distance, convert (FoVy, aspect [width / height]) to 
       extent (left, right, bottom, top), assuming origin being at center."""
    hh = D * np.tan(0.5 * fovy)  # half height
    hw = hh * aspect  # half width
    return np.array([-hw, hw, -hh, hh])


def draw_box(vertices, ax):
    """Draw a box given the vertices.

        Parameters
        ----------
        vertices: array of (2, 8)
            vertices in a specific order
    """
    def draw_line(p1, p2):
        pp = np.vstack([p1, p2])
        ax.plot(pp[:, 0], pp[:, 1], c='grey', alpha=0.3)

    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for p in pairs:
        draw_line(vertices[:2, p[0]], vertices[:2, p[1]])

    return ax
