import sys
sys.path.append('./src')

import time
import numpy as np
import random
from rrt.rrt_star import RRTStar
from rrt.rrt_connect import RRTConnect
from rrt.rrt_star_bid_lstm_3d import RRTStarBidirectional_LSTM
from search_space.search_space import SearchSpace
from utilities.load_dataset import LoadDataset_3D
from utilities.plotting import Plot

def rrt_star(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count):
    # create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    start_time = time.time()
    path = rrt.rrt_star()
    end_time = time.time()

    if path == None:
        path = 0
        step = 0
    else:
        step = len(path)

    cost_time = end_time - start_time

    return rrt, path, step, cost_time

def rrt_connect(X, Q, x_init, x_goal, max_samples, r, prc):
    # create rrt_search
    rrt = RRTConnect(X, Q, x_init, x_goal, max_samples, r, prc)
    start_time = time.time()
    path = rrt.rrt_connect()
    end_time = time.time()

    if path == None:
        path = 0
        step = 0
    else:
        step = len(path)

    cost_time = end_time - start_time

    return rrt, path, step, cost_time

def nn_rrt_star_bid(X, Q, x_init, x_goal, encoded_obs, max_samples, num_attempt, r, model_path, prc, rewire_count):
    # create rrt_search
    rrt = RRTStarBidirectional_LSTM(X, Q, x_init, x_goal, encoded_obs, max_samples, num_attempt, r, model_path, prc, rewire_count)
    start_time = time.time()
    path = rrt.nn_rrt_star_bidirectional()
    end_time = time.time()

    if path == None:
        path = 0
        step = 0
    else:
        step = len(path)

    cost_time = end_time - start_time

    return rrt, path, step, cost_time

if __name__ == '__main__':
    # dimensions of Search Space
    X_dimensions = np.array([(-20, 20), (-20, 20), (-20, 20)])

    # initialize dataloader
    LoadDataset = LoadDataset_3D()

    obs_index = 100
    print('#obs:', obs_index)

    Q = np.array([(8, 1)])     # length of tree edges
    Q_rt = np.array([2])       # length of rrt-connect tree edges
    r = 1                      # length of smallest edge to check for intersection with obstacles
    max_samples = 2048         # max number of samples to take before timing out
    num_attempt = 3            # number of sampling with nn
    rewire_count = 32          # optional, number of nearby branches to rewire
    prc = 0.1                  # probability of checking for a connection to goal
    model_path = './model_2d.pkl'

    # obstacles
    raw_obs, vertices_obs, encoded_obs = LoadDataset.load_obs_data(obs_index)

    Obstacles = vertices_obs

    x_init, x_goal = LoadDataset.load_init_goal(obs_index)

    print('init: ({:+.10f}, {:+.10f}, {:+.10f})'.format(x_init[0], x_init[1], x_init[2]))
    print('goal: ({:+.10f}, {:+.10f}, {:+.10f})'.format(x_goal[0], x_goal[1], x_goal[2]))

    # create search space
    X = SearchSpace(X_dimensions, Obstacles)
    nn_rrt, nn_path, nn_step, nn_cost_time = nn_rrt_star_bid(X, Q, x_init, x_goal, encoded_obs, max_samples, num_attempt, r, model_path, prc, rewire_count)
    rc_rrt, rc_path, rc_step, rc_cost_time = rrt_connect(X, Q_rt, x_init, x_goal, max_samples, r, prc)
    rs_rrt, rs_path, rs_step, rs_cost_time = rrt_star(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)

    print('\nStep: nn: {}, rc: {}, rs: {}'.format(nn_step, rc_step, rs_step))
    print('Time: nn: {:.10f}, rc: {:.10f}, rs: {:.10f}'.format(nn_cost_time, rc_cost_time, rs_cost_time))

    # plot
    paths = [nn_path, rc_path, rs_path]
    plot = Plot('ws')
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open = True)

    plot = Plot('path')
    if paths is not None:
        plot.plot_path(X, paths)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open = True)


