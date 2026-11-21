"""Finite-volume odor bench for the same wall geometry used for rendering."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Wall:
    x: float = 4.5
    half_width: float = .20
    half_length: float = 2.55
    height: float = 3.0

    def contains_xy(self, xy, clearance=0.):
        return ((abs(xy[..., 0] - self.x) <= self.half_width + clearance)
                & (abs(xy[..., 1]) <= self.half_length + clearance))


FRUIT = np.array([9., 0., .8])
FRUIT_RADIUS = .60
DX = .2
X = np.arange(-3., 14. + DX / 2, DX)
Y = np.arange(-6.6, 6.6 + DX / 2, DX)
POINTS = np.stack(np.meshgrid(X, Y, indexing='ij'), axis=-1)


def lattice(blocked):
    ids = np.full(blocked.shape, -1, dtype=np.int32)
    ids[~blocked] = np.arange(np.count_nonzero(~blocked))
    pairs = []
    for first, second in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        both = (first >= 0) & (second >= 0)
        pairs.append((first[both], second[both]))
    left, right = (np.concatenate(col) for col in zip(*pairs, strict=True))
    rows, cols = np.r_[left, right], np.r_[right, left]
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(np.count_nonzero(~blocked),) * 2).tocsr()
    return ids, adjacency


def grid_index(xy):
    i, j = np.rint((np.asarray(xy) - (X[0], Y[0])) / DX).astype(int)
    if not (0 <= i < len(X) and 0 <= j < len(Y)):
        raise ValueError('sensor outside modeled arena')
    return i, j


class OdorField:
    """Steady 2D diffusion and decay. Solid faces have no exchange."""

    def __init__(self, wall=None, *, source_strength=1.):
        if not np.isfinite(source_strength) or source_strength < 0.:
            raise ValueError('odor source must be finite and nonnegative')
        self.blocked = np.zeros(POINTS.shape[:2], dtype=bool)
        if wall is not None:
            self.blocked = wall.contains_xy(POINTS, DX / 2)
        self.ids, graph = lattice(self.blocked)
        source = self.ids[grid_index(FRUIT[:2])]
        assert source >= 0
        from scipy.sparse import diags

        # Reflecting wall faces are omitted from the operator, including the
        # boundary of the arena. Positive decay gives a unique solution.
        decay = .15
        diffusion = 2. / DX**2
        degree = np.asarray(graph.sum(axis=1)).ravel()
        operator = diags(degree * diffusion + decay) - graph * diffusion
        rhs = np.zeros(len(degree))
        rhs[source] = source_strength / DX**2
        solution = spsolve(operator, rhs)
        assert np.isfinite(solution).all() and solution.min() >= -1e-12
        self.value = np.full(self.ids.shape, np.nan)
        self.value[~self.blocked] = solution
        self.residual = float(np.max(np.abs(operator @ solution - rhs)))
        self.balance_error = abs(float(decay * solution.sum() * DX**2) - source_strength)

    def sample(self, xyz):
        """Sample at the actual two antenna points projected onto the arena plane."""
        xy = np.asarray(xyz)[:, :2]
        i0, j0 = np.floor((xy - (X[0], Y[0])) / DX).astype(int).T
        if np.any(i0 < 0) or np.any(j0 < 0) or np.any(i0 + 1 >= len(X)) or np.any(j0 + 1 >= len(Y)):
            raise ValueError('sensor outside modeled arena')
        fx, fy = ((xy - np.c_[X[i0], Y[j0]]) / DX).T
        values = self.value[i0[:, None] + [0, 1, 0, 1], j0[:, None] + [0, 0, 1, 1]]
        weights = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), axis=1)
        valid = np.isfinite(values)
        support = (weights * valid).sum(axis=1)
        if np.any(support == 0.):
            raise ValueError('antenna probe lies inside a solid wall')
        return np.nansum(weights * values, axis=1) / support


def teacher_route(start, goal, wall=None):
    """Clearance-constrained geometric teacher, using privileged coordinates."""
    blocked = np.zeros(POINTS.shape[:2], dtype=bool)
    if wall is not None:
        blocked = wall.contains_xy(POINTS, clearance=1.0)
    ids, graph = lattice(blocked)
    source, target = (ids[grid_index(point)] for point in (start, goal))
    if min(source, target) < 0:
        raise ValueError('route endpoint inside inflated barrier')
    distances, parents = dijkstra(graph, indices=source, unweighted=True, return_predecessors=True)
    if not np.isfinite(distances[target]):
        return None
    chain = [target]
    while chain[-1] != source:
        chain.append(parents[chain[-1]])
    route = POINTS[~blocked][chain[::-1]]
    assert np.all(np.abs(np.diff(route, axis=0)).sum(axis=1) < DX + 1e-9)
    return route
