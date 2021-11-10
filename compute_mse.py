import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt('MSE2.csv', delimiter=',')
data = data.T
t0 = 0.00005


def compute_mse(_data, n):
    _length = _data.shape[0] - n
    _mse = 1 / _length * np.sum((_data[:-n, 0] - _data[n:, 0]) ** 2 + (_data[:-n, 1] - _data[n:, 1]) ** 2)
    return _mse


length = data.shape[0]
n_vector = np.arange(5, int(length / 2), 2000)

length_n_vector = len(n_vector)
mse_vector = np.zeros(length_n_vector)

for i in range(length_n_vector):
    mse_vector[i] = compute_mse(data, n_vector[i])

tau = n_vector * t0
k_value = (mse_vector[-2] - mse_vector[10]) / (tau[-2] - tau[10])
diffusion_coefficient = k_value / 4
print(diffusion_coefficient)

fig, ax = plt.subplots()
ax.plot(tau, mse_vector, 'bo')
ax.plot(tau[10:-2], mse_vector[10:-2] * 1.6, 'k--', linewidth=2)
ax.set_xlabel('$\\tau~[t_0]$')
ax.set_ylabel('MSD $[\\sigma_0^2]$')
ax.set_title('Mean squared displacement (MSD)')
ax.loglog()
ax.set_ylim([0.01, 100])
ax.set_xlim([0.01, 10])

plt.tight_layout()
plt.show()
