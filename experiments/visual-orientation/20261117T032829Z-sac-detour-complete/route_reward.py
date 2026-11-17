"""Dense single-box training feedback from signed COM progress on a safe route."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteConfig:
    front_pad: float
    rear_pad: float
    side_pad: float
    gain: float
    cap: float


class BoxRoute:
    """Shortest path around a rectangular planning margin around the solid box."""

    def __init__(self, box, target_xy, config: RouteConfig):
        x, y, _, half_x, half_y, _ = box
        self.lo = np.array([x-half_x-config.front_pad, y-half_y-config.side_pad])
        self.hi = np.array([x+half_x+config.rear_pad, y+half_y+config.side_pad])
        x0, y0 = self.lo
        x1, y1 = self.hi
        self.nodes = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1], list(target_xy)], dtype=float)
        if self._inside(self.nodes[-1]):
            raise ValueError("Target is inside the obstacle planning margin.")
        width = self.hi-self.lo
        # Strong enough that crossing the interior is costlier than walking
        # around its boundary between any two boundary points.
        self.interior_scale = 2. + float(width.max()/width.min())
        graph = np.full((5,5), np.inf)
        np.fill_diagonal(graph, 0.)
        for i, node in enumerate(self.nodes):
            for j in range(i):
                if self._visible(node, self.nodes[j]):
                    graph[i,j] = graph[j,i] = np.linalg.norm(node-self.nodes[j])
        for k in range(len(graph)):
            graph = np.minimum(graph, graph[:,k,None]+graph[None,k,:])
        self.to_goal = graph[:,-1]

    def _inside(self, point):
        return bool(np.all(point > self.lo) and np.all(point < self.hi))

    def _visible(self, a, b):
        # Only an interval in the *open* rectangle blocks the link. Corner
        # tangencies and travel along a margin edge remain valid.
        start, end = 0., 1.
        for p, d, lo, hi in zip(a,b-a,self.lo,self.hi):
            if abs(d) <= 1.e-12:
                if p <= lo+1.e-12 or p >= hi-1.e-12:
                    return True
                continue
            near, far = sorted(((lo-p)/d,(hi-p)/d))
            start, end = max(start,near), min(end,far)
            if end-start <= 1.e-12:
                return True
        return False

    def _outside_distance(self, point):
        return float(min(np.linalg.norm(point-node)+cost
                         for node,cost in zip(self.nodes,self.to_goal,strict=True)
                         if self._visible(point,node)))

    def distance(self, point):
        point = np.asarray(point, dtype=float)
        if not self._inside(point):
            return self._outside_distance(point)
        candidates = [np.array([edge, point[1]]) for edge in (self.lo[0],self.hi[0])]
        candidates.extend(np.array([point[0],edge]) for edge in (self.lo[1],self.hi[1]))
        return min(self._outside_distance(edge)+self.interior_scale*np.linalg.norm(point-edge)
                   for edge in candidates)


def rescore(terms, columns, xy, goal_xy, box, reasons, config: RouteConfig):
    """Replace the two head-based dense terms. Keep physical outcomes and costs."""
    terms = np.asarray(terms, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (len(terms)+1,2) or len(reasons) != len(terms) or not np.all(np.isfinite(xy)):
        raise ValueError("Expected a finite COM position before and after each scoring transition.")
    if box is None:
        return terms.copy(), np.linalg.norm(xy-goal_xy,axis=1)
    route = BoxRoute(box, goal_xy, config)
    remaining = np.fromiter((route.distance(p) for p in xy), dtype=float, count=len(xy))
    progress_coordinate = np.clip(config.gain*(remaining[0]-remaining),0.,config.cap)
    # Weight movement here and discount returns in SAC. Discounting the
    # coordinate itself would charge the fly every tick for earlier progress.
    progress = np.diff(progress_coordinate)
    # No last-step credit for a physical failure.
    unsafe = np.isin(reasons,("obstacle_contact","fall","overshoot"))
    progress[unsafe] = np.minimum(progress[unsafe],0.)
    changed = terms.copy()
    changed[:,columns.index("potential")] = 0.
    changed[:,columns.index("progress")] = progress
    return changed, remaining
