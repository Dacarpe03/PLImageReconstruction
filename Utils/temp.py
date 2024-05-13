import numpy as np

from scipy.stats import unitary_group

import matplotlib.pyplot as plt

plt.ion()



# M = np.load('19PLmat.npy')



# Get a random unitary matrix

nmodes = 9

M = unitary_group.rvs(nmodes)



# Check its condition number - if it's close to 1 then the matrix is well conditioned

M_condnum = np.linalg.cond(M)

print('Matrix condition number: %.16f' % M_condnum)



# Description of forward propagation, g(phi):

def g(phi):

    I = np.abs(M @ np.exp(1j * phi))**2

    return I



# Example of propagating some random phase inputs

philim = [-1, 1]  # [-5, 5] # Min and max phase coeffs (rad)

npts = 10

phis = np.random.uniform(philim[0], philim[1], (nmodes, npts))

I = np.abs(M @ np.exp(1j * phis))**2 # Output intensities

plt.plot(I.T)