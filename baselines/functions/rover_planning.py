from typing import Tuple, Optional, ClassVar, Dict
from concurrent.futures import ProcessPoolExecutor

import gym
import torch
import numpy as np
import scipy.interpolate as si
import math


class Trajectory:

    def set_params(self, start, goal, params):
        raise NotImplemented

    def get_points(self, t):
        raise NotImplemented

    @property
    def param_size(self):
        raise NotImplemented


class PointBSpline(Trajectory):
    """
    dim : number of dimensions of the state space
    num_points : number of internal points used to represent the trajectory.
                    Note, internal points are not necessarily on the trajectory.
    """

    def __init__(self, dim, num_points):
        self.tck = None
        self.d = dim
        self.npoints = num_points

    """
    Set fit the parameters of the spline from a set of points. If values are given for start or goal,
    the start or endpoint of the trajectory will be forces on those points, respectively.
    """

    def set_params(self, params, start=None, goal=None):

        points = params.reshape((-1, self.d)).T

        if start is not None:
            points = np.hstack((start[:, None], points))

        if goal is not None:
            points = np.hstack((points, goal[:, None]))

        self.tck, u = si.splprep(points, k=3)

        if start is not None:
            for a, sv in zip(self.tck[1], start):
                a[0] = sv

        if goal is not None:
            for a, gv in zip(self.tck[1], goal):
                a[-1] = gv

    def get_points(self, t):
        assert self.tck is not None, "Parameters have to be set with set_params() before points can be queried."
        return np.vstack(si.splev(t, self.tck)).T

    @property
    def param_size(self):
        return self.d * self.npoints


def simple_rbf(x, point):
    return (1 - np.exp(-np.sum(((x - point) / 0.25) ** 2)))

def get_minimum_distance(trajectory, point):
    """
    Calculate the minimum distance between a point and a set of points in an trajectory.
    """
    distances = np.linalg.norm(trajectory - point, axis=1)
    return np.min(distances)


def get_distance_inside_box(point, box_min, box_max):
    """
    상자 내부에 있는 점과 상자의 변들 사이의 최소 거리를 계산합니다.
    점이 상자 외부에 있는 경우 None을 반환합니다.
    
    :param point: (x, y) 형태의 점 (예: (x1, y1))
    :param box_min: 상자의 최소 경계 (x_min, y_min)
    :param box_max: 상자의 최대 경계 (x_max, y_max)
    
    :return: 점이 상자 내부에 있을 때 변까지의 최소 거리, 외부에 있으면 None
    """
    # 점이 상자 내부에 있는지 확인
    if (box_min[0] <= point[0] <= box_max[0] and 
        box_min[1] <= point[1] <= box_max[1]):
        
        # 각 변까지의 거리 계산
        dist_to_left = point[0] - box_min[0]
        dist_to_right = box_max[0] - point[0]
        dist_to_bottom = point[1] - box_min[1]
        dist_to_top = box_max[1] - point[1]
        
        # 가장 가까운 변까지의 거리 반환
        return min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
    else:
        # 상자 외부에 있으면 None 반환
        assert False

def get_max_box_distance(trajectory, point, size):
    """
    상자 안에 존재하는 trajectory 점들에 대해 각 점과 상자의 변들 사이의 최소 거리를 구한 뒤,
    그 중 가장 큰 거리를 반환합니다.
    
    :param trajectory: 경로 상의 점들 (예: [(x1, y1), (x2, y2), ...])
    :param point: 제약 상자의 중심점 (x_center, y_center)
    :param size: 제약 상자의 절반 크기 (가로/세로)
    
    :return: 상자 안에 있는 점들과 상자의 변들 사이의 최대 최소 거리
    """
    # 상자 경계 정의
    box_min = np.array([point[0] - size, point[1] - size])
    box_max = np.array([point[0] + size, point[1] + size])
    
    max_distance = -float('inf')  # 초기 최대 거리 값 (무한대로 설정)
    
    # 경로의 각 점에 대해
    for p in trajectory:
        # 상자 안에 있는 점들만 고려
        if np.all(p >= box_min) and np.all(p <= box_max):
            
            # 상자 안에 있으면 해당 점과 상자 변들 사이의 최소 거리 계산
            distance_to_boundary = get_distance_inside_box(p, box_min, box_max)
            max_distance = max(max_distance, distance_to_boundary)
    
    return max_distance
class RoverDomain:
    """
    Rover domain defined on R^d
    cost_fn : vectorized function giving a scalar cost to states
    start : a start state for the rover
    goal : a goal state
    traj : a parameterized trajectory object offering an interface
            to interpolate point on the trajectory
    s_range : the min and max of the state with s_range[0] in R^d are
                the mins and s_range[1] in R^d are the maxs
    """

    def __init__(self, cost_fn,
                 cost_fn_viz,
                 start,
                 goal,
                 traj,
                 s_range,
                 start_miss_cost=None,
                 goal_miss_cost=None,
                 force_start=True,
                 force_goal=True,
                 constraints = None,
                 constraint_location = None,
                 constraint_size = None,
                 only_add_start_goal=True,
                 rnd_stream=None):
        self.cost_fn = cost_fn
        self.cost_fn_viz = cost_fn_viz
        self.start = start
        self.goal = goal
        self.traj = traj
        self.s_range = s_range
        self.rnd_stream = rnd_stream
        self.force_start = force_start
        self.force_goal = force_goal
        self.constraints = constraints
        self.constraint_location = constraint_location
        self.constraint_size = constraint_size
        self.goal_miss_cost = goal_miss_cost
        self.start_miss_cost = start_miss_cost

        if self.start_miss_cost is None:
            self.start_miss_cost = simple_rbf
        if self.goal_miss_cost is None:
            self.goal_miss_cost = simple_rbf

        if self.rnd_stream is None:
            self.rnd_stream = np.random.RandomState(np.random.randint(0, 2 ** 32 - 1))

    # return the negative cost which need to be optimized
    def __call__(self, params, n_samples=1000):
        self.set_params(params)

        return self.estimate_cost(n_samples=n_samples)

    def set_params(self, params):
        self.traj.set_params(params + self.rnd_stream.normal(0, 1e-4, params.shape),
                             self.start if self.force_start else None,
                             self.goal if self.force_goal else None)
    
    # def eval_constraints(self, points):
    #     """
    #     Evaluate the constraints at the given points.
    #     :param points: The points to evaluate the constraints at.
    #     :return: The evaluated constraints.
    #     """
    #     # Check if any point is within the constraint
    #     c_is = []
    #     for i in range(len(self.constraint_location)):
    #         constraint_point = self.constraint_location[i]
    #         constraint_x_h, constraint_y_h = constraint_point + self.constraint_size
    #         constraint_x_l, constraint_y_l = constraint_point - self.constraint_size
    #         points = np.array(points)
    #         constraint_violated = (
    #             (constraint_x_l <= points[:, 0]) & (points[:, 0] <= constraint_x_h) &
    #             (constraint_y_l <= points[:, 1]) & (points[:, 1] <= constraint_y_h)
    #         )

    #         if constraint_violated.any():
    #             c_is.append(get_max_box_distance(trajectory=points, point = constraint_point, size = self.constraint_size))
    #         else:
    #             c_is.append(-get_minimum_distance(trajectory=points, point = constraint_point))        
        
    #     # stack all the constraints
    #     constraints = np.vstack(c_is)
        
    #     return constraints
    
    def eval_constraints(self, points):
        points = np.asarray(points)  # (M, 2)
        locations = np.asarray(self.constraint_location)  # (N, 2)
        size = self.constraint_size  # scalar or (2,)
        box_min = locations - size  # (N, 2)
        box_max = locations + size  # (N, 2)

        # 각 박스마다 points가 안에 있는지 검사 → (N, M) boolean mask
        in_box = (
            (points[None, :, 0] >= box_min[:, 0:1]) &
            (points[None, :, 0] <= box_max[:, 0:1]) &
            (points[None, :, 1] >= box_min[:, 1:2]) &
            (points[None, :, 1] <= box_max[:, 1:2])
        )  # shape: (N, M)

        # 결과 담을 배열
        result = np.zeros(len(locations))

        for i in range(len(locations)):
            points_in_box = points[in_box[i]]
            if len(points_in_box) > 0:
                dists = np.minimum.reduce([
                    points_in_box[:, 0] - box_min[i, 0],  # left
                    box_max[i, 0] - points_in_box[:, 0],  # right
                    points_in_box[:, 1] - box_min[i, 1],  # bottom
                    box_max[i, 1] - points_in_box[:, 1]   # top
                ])
                result[i] = np.max(dists)
            else:
                distances = np.linalg.norm(points - locations[i], axis=1)
                result[i] = -np.min(distances)

        return result[:, None]  # shape: (N, 1)
    
    
    def estimate_cost(self, n_samples=1000):
        # get points on the trajectory
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        # compute cost at each point
        costs = self.cost_fn(points)
        # check any point within the constraint
        constraints = self.eval_constraints(points)

        # estimate (trapezoidal) the integral of the cost along traj
        avg_cost = 0.5 * (costs[:-1] + costs[1:])
        l = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_cost = np.sum(l * avg_cost)

        if not self.force_start:
            total_cost += self.start_miss_cost(points[0], self.start)
        if not self.force_goal:
            total_cost += self.goal_miss_cost(points[-1], self.goal)

        return -total_cost, constraints

    @property
    def input_size(self):
        return self.traj.param_size


class AABoxes:
    def __init__(self, lows, highs):
        self.l = lows
        self.h = highs

    def contains(self, X):
        if X.ndim == 1:
            X = X[None, :]

        lX = self.l.T[None, :, :] <= X[:, :, None]
        hX = self.h.T[None, :, :] > X[:, :, None]

        return (lX.all(axis=1) & hX.all(axis=1))


class NegGeom:
    def __init__(self, geometry):
        self.geom = geometry

    def contains(self, X):
        return ~self.geom.contains(X)


class UnionGeom:
    def __init__(self, geometries):
        self.geoms = geometries

    def contains(self, X):
        return np.any(np.hstack([g.contains(X) for g in self.geoms]), axis=1, keepdims=True)


class ConstObstacleCost:
    def __init__(self, geometry, cost):
        self.geom = geometry
        self.c = cost

    def __call__(self, X):
        return self.c * self.geom.contains(X)

class ImpassibleObstacleCost:
    def __init__(self, geometry):
        self.geom = geometry
        
    def __call__(self, X):
        # if anything contains X return inf
        return self.geom.contains(X)


class ConstCost:
    def __init__(self, cost):
        self.c = cost

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]
        return np.ones((X.shape[0], 1)) * self.c


class AdditiveCosts:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, X):
        return np.sum(np.hstack([f(X) for f in self.fns]), axis=1)


class GMCost:
    def __init__(self, centers, sigmas, weights=None):
        self.c = centers
        self.s = sigmas
        if self.s.ndim == 1:
            self.s = self.s[:, None]
        self.w = weights
        if weights is None:
            self.w = np.ones(centers.shape[0])

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]

        return np.exp(-np.sum(((X[:, :, None] - self.c.T[None, :, :]) / self.s.T[None, :, :]) ** 2, axis=1)).dot(self.w)


def plot_2d_rover(roverdomain, ngrid_points=100, ntraj_points=100, colormap='Spectral', draw_colorbar=False):
    import matplotlib.pyplot as plt
    # get a grid of points over the state space
    points = [np.linspace(mi, ma, ngrid_points, endpoint=True) for mi, ma in zip(*roverdomain.s_range)]
    grid_points = np.meshgrid(*points)
    points = np.hstack([g.reshape((-1, 1)) for g in grid_points])

    # compute the cost at each point on the grid
    costs = roverdomain.cost_fn_viz(points)
    # get the cost of the current trajectory
    traj_cost = roverdomain.estimate_cost()

    # get points on the current trajectory
    traj_points = roverdomain.traj.get_points(np.linspace(0., 1.0, ntraj_points, endpoint=True))
    plt.figure(figsize = (10, 10))

    # set title to be the total cost
    plt.title('traj cost: {0}'.format(traj_cost))
    print('traj cost: {0}'.format(traj_cost))
    # plot cost function
    cmesh = plt.pcolormesh(grid_points[0], grid_points[1], costs.reshape((ngrid_points, -1)), cmap=colormap)
    if draw_colorbar:
        plt.gcf().colorbar(cmesh)
    # plot traj
    plt.plot(traj_points[:, 0], traj_points[:, 1], 'g')
    # plot start and goal
    plt.plot([roverdomain.start[0], roverdomain.goal[0]], (roverdomain.start[1], roverdomain.goal[1]), 'ok')
    return cmesh


def generate_verts(rectangles):
    poly3d = []
    all_faces = []
    vertices = []
    for l, h in zip(rectangles.l, rectangles.h):
        verts = [[l[0], l[1], l[2]], [l[0], h[1], l[2]], [h[0], h[1], l[2]], [h[0], l[1], l[2]],
                 [l[0], l[1], h[2]], [l[0], h[1], h[2]], [h[0], h[1], h[2]], [h[0], l[1], h[2]]]

        faces = [[0, 1, 2, 3], [0, 3, 7, 4], [3, 2, 6, 7], [7, 6, 5, 4], [1, 5, 6, 2], [0, 4, 5, 1]]

        vert_ind = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 3, 7], [7, 3, 2], [2, 6, 7],
                    [7, 5, 4], [7, 6, 5], [2, 5, 6], [2, 1, 5], [0, 1, 4], [1, 4, 5]]

        plist = [[verts[vert_ind[ix][iy]] for iy in range(len(vert_ind[0]))] for ix in range(len(vert_ind))]
        faces = [[verts[faces[ix][iy]] for iy in range(len(faces[0]))] for ix in range(len(faces))]

        poly3d = poly3d + plist
        vertices = vertices + verts
        all_faces = all_faces + faces

    return poly3d, vertices, all_faces


def plot_3d_forest_rover(roverdomain, rectangles, ntraj_points=100):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    # get the cost of the current trajectory
    traj_cost = roverdomain.estimate_cost()

    # get points on the current trajectory
    traj_points = roverdomain.traj.get_points(np.linspace(0., 1.0, ntraj_points, endpoint=True))

    # convert the rectangles into lists of vertices for matplotlib
    poly3d, verts, faces = generate_verts(rectangles)

    ax = plt.gcf().add_subplot(111, projection='3d')

    # plot start and goal
    ax.scatter((roverdomain.start[0], roverdomain.goal[0]),
               (roverdomain.start[1], roverdomain.goal[1]),
               (roverdomain.start[2], roverdomain.goal[2]), c='k')

    # plot traj
    seg = (zip(traj_points[:-1, :], traj_points[1:, :]))
    ax.add_collection3d(Line3DCollection(seg, colors=[(0, 1., 0, 1.)] * len(seg)))

    # plot rectangles
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=(0.7, 0.7, 0.7, 1.), linewidth=0.5))

    # set limits of axis to be the same as domain
    s_range = roverdomain.s_range
    ax.set_xlim(s_range[0][0], s_range[1][0])
    ax.set_ylim(s_range[0][1], s_range[1][1])
    ax.set_zlim(s_range[0][2], s_range[1][2])



class BaseFunc():
    def __init__(self, dims: int, lb: np.ndarray, ub: np.ndarray):
        self._dims = dims
        self._lb = lb
        self._ub = ub

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def dims(self) -> int:
        return self._dims


class NormalizedInputFn:
    def __init__(self, fn_instance, x_range):
        self.fn_instance = fn_instance
        self.x_range = x_range

    def __call__(self, x):
        return self.fn_instance(self.project_input(x))

    def project_input(self, x):
        return x * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

    def inv_project_input(self, x):
        return (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

    def get_range(self):
        return np.array([np.zeros(self.x_range[0].shape[0]), np.ones(self.x_range[0].shape[0])])


class ConstantOffsetFn:
    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        return self.fn_instance(x)[0] + self.offset, self.fn_instance(x)[1]

    def get_range(self):
        return self.fn_instance.get_range()


def create_cost_small():
    c = np.array([[0.11353145, 0.17251116],
                  [0.4849413, 0.7684513],
                  [0.38840863, 0.10730809],
                  [0.32968556, 0.37542275],
                  [0.64342773, 0.32438415],
                  [0.42, 0.35],
                  [0.38745546, 0.0688907],
                  [0.05771529, 0.1670573],
                  [0.48750001, 0.67864249],
                  [0.5294646, 0.66245226],
                  [0.88495861, 0.76770809],
                  [0.71132462, 0.46580745],
                  [0.02038182, 0.32146063],
                  [0.34077448, 0.70446464],
                  [0.61490175, 0.79081785],
                  [0.37367616, 0.6720441],
                  [0.14711569, 0.57060365],
                  [0.76084188, 0.65168123],
                  [0.51038721, 0.78655373],
                  [0.50396508, 0.90299952],
                  [0.23763956, 0.38260748],
                  [0.40169679, 0.72553068],
                  [0.59670114, 0.08541569],
                  [0.5514408, 0.62855134],
                  [0.84606733, 0.94264543],
                  [0.8, 0.19590218],
                  [0.39181603, 0.46357532],
                  [0.44800403, 0.27380116],
                  [0.5681913, 0.1468706],
                  [0.37418262, 0.69210095]])

    l = c - 0.05
    h = c + 0.05


    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5
    r_h = r_box + 0.5

    trees = AABoxes(l, h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])

    start = np.zeros(2) + 0.05
    goal = np.array([0.95, 0.95])

    costs = [ConstObstacleCost(obstacles, cost=20.),   ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    return cost_fn, start, goal


def create_small_domain():
    cost_fn, start, goal = create_cost_small()

    n_points = 10
    traj = PointBSpline(dim=2, num_points=n_points)
    n_params = traj.param_size
    domain = RoverDomain(cost_fn,
                         cost_fn_viz,
                         start=start,
                         goal=goal,
                         traj=traj,
                         s_range=np.array([[-0.1, -0.1], [1.1, 1.1]]))

    return domain


def create_cost_large():
    c = np.array([[0.43143755, 0.20876147],
                  [0.38485367, 0.39183579],
                  [0.02985961, 0.22328303],
                  [0.7803707, 0.3447003],
                  [0.93685657, 0.56297285],
                  [0.04194252, 0.23598362],
                  [0.28049582, 0.40984475],
                  [0.6756053, 0.70939481],
                  [0.01926493, 0.86972335],
                  [0.5993437, 0.63347932],
                  [0.57807619, 0.40180792],
                  [0.56824287, 0.75486851],
                  [0.35403502, 0.38591056],
                  [0.72492026, 0.59969313],
                  [0.27618746, 0.64322757],
                  [0.54029566, 0.25492943],
                  [0.30903526, 0.60166842],
                  [0.2913432, 0.29636879],
                  [0.78512072, 0.62340245],
                  [0.29592116, 0.08400595],
                  [0.87548394, 0.04877622],
                  [0.21714791, 0.9607346],
                  [0.92624074, 0.53441687],
                  [0.53639253, 0.45127928],
                  [0.99892031, 0.79537837],
                  [0.84621631, 0.41891986],
                  [0.39432819, 0.06768617],
                  [0.92365693, 0.72217512],
                  [0.95520914, 0.73956575],
                  [0.820383, 0.53880139],
                  [0.22378049, 0.9971974],
                  [0.34023233, 0.91014706],
                  [0.64960636, 0.35661133],
                  [0.29976464, 0.33578931],
                  [0.43202238, 0.11563227],
                  [0.66764947, 0.52086962],
                  [0.45431078, 0.94582745],
                  [0.12819915, 0.33555344],
                  [0.19287232, 0.8112075],
                  [0.61214791, 0.71940626],
                  [0.4522542, 0.47352186],
                  [0.95623345, 0.74174186],
                  [0.17340293, 0.89136853],
                  [0.04600255, 0.53040724],
                  [0.42493468, 0.41006649],
                  [0.37631485, 0.88033853],
                  [0.66951947, 0.29905739],
                  [0.4151516, 0.77308712],
                  [0.55762991, 0.26400156],
                  [0.6280609, 0.53201974],
                  [0.92727447, 0.61054975],
                  [0.93206587, 0.42107549],
                  [0.63885574, 0.37540613],
                  [0.15303425, 0.57377797],
                  [0.8208471, 0.16566631],
                  [0.14889043, 0.35157346],
                  [0.71724622, 0.57110725],
                  [0.32866327, 0.8929578],
                  [0.74435871, 0.47464421],
                  [0.9252026, 0.21034329],
                  [0.57039306, 0.54356078],
                  [0.56611551, 0.02531317],
                  [0.84830056, 0.01180542],
                  [0.51282028, 0.73916524],
                  [0.58795481, 0.46527371],
                  [0.83259048, 0.98598188],
                  [0.00242488, 0.83734691],
                  [0.72505789, 0.04846931],
                  [0.07312971, 0.30147979],
                  [0.55250344, 0.23891255],
                  [0.51161315, 0.46466442],
                  [0.802125, 0.93440495],
                  [0.9157825, 0.32441602],
                  [0.44927665, 0.53380074],
                  [0.67708372, 0.67527231],
                  [0.81868924, 0.88356194],
                  [0.48228814, 0.88668497],
                  [0.39805433, 0.99341196],
                  [0.86671752, 0.79016975],
                  [0.01115417, 0.6924913],
                  [0.34272199, 0.89543756],
                  [0.40721675, 0.86164495],
                  [0.26317679, 0.37334193],
                  [0.74446787, 0.84782643],
                  [0.55560143, 0.46405104],
                  [0.73567977, 0.12776233],
                  [0.28080322, 0.26036748],
                  [0.17507419, 0.95540673],
                  [0.54233783, 0.1196808],
                  [0.76670967, 0.88396285],
                  [0.61297539, 0.79057776],
                  [0.9344029, 0.86252764],
                  [0.48746839, 0.74942784],
                  [0.18657635, 0.58127321],
                  [0.10377802, 0.71463978],
                  [0.7771771, 0.01463505],
                  [0.7635042, 0.45498358],
                  [0.83345861, 0.34749363],
                  [0.38273809, 0.51890558],
                  [0.33887574, 0.82842507],
                  [0.02073685, 0.41776737],
                  [0.68754547, 0.96430979],
                  [0.4704215, 0.92717361],
                  [0.72666234, 0.63241306],
                  [0.48494401, 0.72003268],
                  [0.52601215, 0.81641253],
                  [0.71426732, 0.47077212],
                  [0.00258906, 0.30377501],
                  [0.35495269, 0.98585155],
                  [0.65507544, 0.03458909],
                  [0.10550588, 0.62032937],
                  [0.60259145, 0.87110846],
                  [0.04959159, 0.535785]])
    
    
    additional_C = np.array([[0.65294118, 1 - 0.86328872],
            [0.20784314, 1 - 0.80879541],
            [0.81960784, 1 - 0.75525813],
            [0.3745098 , 1 - 0.70172084],
            [0.54117647, 1 - 0.64818356],
            [0.70784314, 1 - 0.59369025],
            [0.09607843, 1 - 0.54015296],
            [0.48627451, 1 - 0.48661568],
            [0.8745098 , 1 - 0.43212237],
            [0.2627451 , 1 - 0.37858509],
            [0.42941176, 1 - 0.3250478 ],
            [0.59607843, 1 - 0.27151052],
            [0.15294118, 1 - 0.21701721],
            [0.7627451 , 1 - 0.16347992],
            [0.31960784, 1 - 0.10994264]])


    l = c - 0.025
    h = c + 0.025
    
    constraint_size = 0.05
    additional_C_l = additional_C - constraint_size
    additional_C_h = additional_C + constraint_size
    

    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5 #Define the Big box range
    r_h = r_box + 0.5

    trees = AABoxes(l, h)
    additional_trees = AABoxes(additional_C_l, additional_C_h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])
    obstacles_2 = UnionGeom([additional_trees, r_box])

    start = np.zeros(2) + constraint_size
    goal = np.array([0.95, 0.95])

    costs = [ConstObstacleCost(obstacles, cost=20.), ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    costs_viz = [ConstObstacleCost(obstacles, cost=20.), ConstCost(0.05), ConstObstacleCost(obstacles_2, cost=200.)]
    cost_fn_viz = AdditiveCosts(costs_viz)
    constraints = ImpassibleObstacleCost(obstacles_2)
    return cost_fn, cost_fn_viz, start, goal, constraints, additional_C, constraint_size


def create_large_domain(force_start=False,
                        force_goal=False,
                        start_miss_cost=None,
                        goal_miss_cost=None):
    cost_fn, cost_fn_viz, start, goal, constraints, constraint_location, constraint_size = create_cost_large()

    n_points = 30
    traj = PointBSpline(dim=2, num_points=n_points)
    n_params = traj.param_size
    domain = RoverDomain(cost_fn,
                         cost_fn_viz,
                         start=start,
                         goal=goal,
                         traj=traj,
                         start_miss_cost=start_miss_cost,
                         goal_miss_cost=goal_miss_cost,
                         force_start=force_start,
                         force_goal=force_goal,
                         constraints = constraints,
                         constraint_location = constraint_location,
                         constraint_size = constraint_size,
                         s_range=np.array([[-0.1, -0.1], [1.1, 1.1]]))
    return domain

def create_large_domain_50(force_start=False,
                        force_goal=False,
                        start_miss_cost=None,
                        goal_miss_cost=None):
    cost_fn, cost_fn_viz, start, goal, constraints, constraint_location, constraint_size = create_cost_large()

    n_points = 50
    traj = PointBSpline(dim=2, num_points=n_points)
    n_params = traj.param_size
    domain = RoverDomain(cost_fn,
                         cost_fn_viz,
                         start=start,
                         goal=goal,
                         traj=traj,
                         start_miss_cost=start_miss_cost,
                         goal_miss_cost=goal_miss_cost,
                         force_start=force_start,
                         force_goal=force_goal,
                         constraints = constraints,
                         constraint_location = constraint_location,
                         constraint_size = constraint_size,
                         s_range=np.array([[-0.1, -0.1], [1.1, 1.1]]))
    return domain



def l2cost(x, point):
    return 10 * np.linalg.norm(x - point, 1)


class Square:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb = -2 * np.ones(dims)
        self.ub = 2 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        result = -1 * np.power(x, 2) + 5
        return float(result)


class Rover(BaseFunc):
    def __init__(self, dim, dtype, device, force_initial_end=False):
        if dim==60:
            super().__init__(60, np.zeros(60), np.ones(60))
            domain = create_large_domain(force_start=False,
                                            force_goal=False,
                                            start_miss_cost=l2cost,
                                            goal_miss_cost=l2cost)
            self.bounds = torch.zeros((2, 60), dtype=dtype, device=device)
        elif dim==100:
            super().__init__(100, np.zeros(100), np.ones(100))
            domain = create_large_domain_50(force_start=False,
                                                force_goal=False,
                                                start_miss_cost=l2cost,
                                                goal_miss_cost=l2cost)
            self.bounds = torch.zeros((2, 100), dtype=dtype, device=device)
        else:
            # error
            raise ValueError(f"Unknown dim: {dim}")

        self.dtype = dtype
        self.device = device
        raw_x_range = np.repeat(domain.s_range, domain.traj.npoints, axis=1)

        # maximum value of f
        f_max = 5.0
        self._func = NormalizedInputFn(ConstantOffsetFn(domain, f_max), raw_x_range)
        # x_range = self._func.get_range()
        #
        # x = np.random.uniform(x_range[0], x_range[1])
        # print("type:", type(self.evaluate(x)))
        # print("x_range[0]:", x_range[0], len(x_range[0]))
        # print("x_range[1]:", x_range[1], len(x_range[1]))
        # print('Input = {}'.format(x))
        # print('Output = {}'.format(self.evaluate(x)))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = x.cpu().detach().numpy().reshape(1, -1)
        return torch.tensor(self._func(x)[0], dtype=self.dtype, device = self.device), torch.tensor(self._func(x)[1].reshape(1, -1), dtype=self.dtype, device = self.device)
    
    
if __name__ == "__main__":
    # Draw the rover planning with plt
    import matplotlib.pyplot as plt
    # Create a small domain
    domain = create_large_domain_50(force_start=True,
                                        force_goal=True,
                                        start_miss_cost=l2cost,
                                        goal_miss_cost=l2cost)
    domain.set_params(torch.randn((100, 2))/1000)
    
    # Plot the 2D rover domain
    # plot_2d_rover(domain, ngrid_points=100, ntraj_points=100, colormap='Spectral', draw_colorbar=True)

    # create a small domain
    # domain = create_small_domain()
    # domain.set_params(torch.randn((60, 2))/1000)
    
    plot_2d_rover(domain, ngrid_points=100, ntraj_points=100, colormap='Spectral', draw_colorbar=True)

    # save the plot image
    plt.savefig("rover_domain_plot.png")