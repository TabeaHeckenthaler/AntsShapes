

"""
Parameters I chose??
"""

# ant force [N]
f_0 = 1

"""
Parameters from the paper
"""
# detachment rate from a moving cargo, detachment from a non moving cargo [1/s]
k1_off, k2_off = 0.035, 0.01
# attachment independent of cargo velocity [1/s]
k_on = 0.0017
# Radius of the object
radius = 0.57
# Number of available sites
N_max = 20
#
k_c = 0.7
# beta, I am not sure, what this is
beta = 1.65
# connected to the ants decision making process
f_ind = 10 * f_0
# kinetic friction linear coefficient [N * sec/cm]
gamma = 25 * f_0
# kinetic rotational friction coefficient [N * sec/c]
gamma_rot = 0.4 * gamma

f_kinx, f_kiny = 0, 0
n_av = 100  # number of available ants
phi_max = 0.9075712110370514  # rad in either direction from the normal