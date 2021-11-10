import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

N = 36
sigma = 1
epsilon = 1
m = 1
M = 40 * m
L = 100 * sigma
rho = 10 * sigma
number_of_timesteps = 2000
v0 = np.sqrt(2 * epsilon / m)
velocity_factor = 20
t0 = sigma / v0
timestep_factor = 0.00005
timestep = t0 * timestep_factor
fps = 1000

position_M_array = np.zeros([1, 2, number_of_timesteps])
position_M_array[:, :, 0] = np.array([L / 2, L / 2])
position_m_array = np.zeros([N, 2, number_of_timesteps])
velocity_m_array = np.zeros([N, 2, number_of_timesteps])


def initialize_particles(_position_M, _sigma, _v0, _L, _N, _velocity_factor, _rho):

    _angle = np.random.rand(_N) * 2 * np.pi
    _velocity_list = 2 * _v0 * np.reshape((np.cos(_angle), np.sin(_angle)), (_N, 2))
    _position_list = np.zeros([_N, 2])
    for i in range(_N):
        _placing = True
        while _placing:
            _position_candidate = np.expand_dims(np.random.rand(2) * _L, axis=0)
            if not inside_disc(_position_candidate, _position_M, _rho):
                _position_list[i, :] = _position_candidate
                _placing = False
            else:
                print("Too close! Placing a new particle!")
    return _position_list, _velocity_list


def inside_disc(_position_m, _position_M, _rho):
    _center_distance = get_distance(_position_m, _position_M)
    if _center_distance < _rho + sigma:
        return True
    else:
        return False


def outside_box(_position_list, _velocity_list, _L, _N):
    for i in range(_N):
        if _position_list[i, 0] > _L:
            _position_list[i, 0] = 2 * _L - _position_list[i, 0]
            _velocity_list[i, 0] *= -1
        elif _position_list[i, 0] < 0:
            _position_list[i, 0] *= -1
            _velocity_list[i, 0] *= -1

        if _position_list[i, 1] > _L:
            _position_list[i, 1] = 2 * _L - _position_list[i, 1]
            _velocity_list[i, 1] *= -1
        elif _position_list[i, 1] < 0:
            _position_list[i, 1] *= -1
            _velocity_list[i, 1] *= -1
    return _position_list, _velocity_list


def get_distance(_position_m, _position_M):
    _distances = np.linalg.norm(_position_m - _position_M, axis=1)
    return _distances


def get_force(_position_m, _position_M):
    _distance = get_distance(_position_m, _position_M)
    _direction_m = (_position_M - _position_m) / _distance
    _direction_M = - _direction_m
    _magnitude = 4 * epsilon * (12 * (sigma ** 12 / _distance ** 13)
                                - 6 * (sigma ** 6 / _distance ** 7))
    _force_m = _direction_m * _magnitude
    _force_M = _direction_M * _magnitude
    return _force_m, _force_M


def collide():

    return


def step(_position_m_list, _velocity_list, _N, _L, _m, _M, _timestep):

    return


position_m_array[:, :, 0], velocity_m_array[:, :, 0] = initialize_particles(position_M_array[:, :, 0], sigma,
                                                                            v0, L, N, velocity_factor, rho)

disc_plot = np.linspace(0, 2 * np.pi, 1000)
plt.fill(np.cos(disc_plot) * rho + position_M_array[0, 0, 0], np.sin(disc_plot) * rho + position_M_array[0, 1, 0],
         alpha=0.4, c='C0')
plt.plot(np.cos(disc_plot) * rho + position_M_array[0, 0, 0], np.sin(disc_plot) * rho + position_M_array[0, 1, 0], 'k',
         linewidth='0.8')
plt.scatter(position_m_array[:, 0, 0], position_m_array[:, 1, 0], s=14, c='brown', edgecolor='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('square')

plt.show()