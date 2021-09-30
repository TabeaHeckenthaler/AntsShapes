# similar to the Gillespie Code described in the SI of the nature communications paper
# titled 'Ants optimally amplify... '
import numpy as np
from Setup.Load import Gillespie_sites_angels


def rot(angle: float):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


"""
Parameters not yet confirmed
"""
# ant force [N]
f_0 = 1

"""
Parameters from the paper
"""
# detachment rate from a moving cargo, detachment from a non moving cargo [1/s]
k1_off, k2_off = 0.035, 0.01  # TODO: If its still in one points, its still in all points?
# attachment independent of cargo velocity [1/s]
k_on = 0.0017
# Radius of the object
radius = 0.57
# Number of available sites
# N_max = 20 # TODO
N_max = 4
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

"""
Parameters I chose
"""


class Gillespie:
    def __init__(self, my_load, x=None):
        self.n_p = [0 for _ in range(N_max)]  # array with 0 and 1 depending on whether is puller or not
        self.n_l = [0 for _ in range(N_max)]  # array with 0 and 1 depending on whether is lifter of not

        self._attachment_sites, self._phi_default_load_coord = Gillespie_sites_angels(my_load, N_max, x=x)
        # vector to ith attachment site in load coordinates
        # angle of normal vector from ith attachment site to the x axis of the world, when my_load.angle = 0

        # TODO: I misunderstood until now. At every step, the phi to the world coordinates stays the same.
        self.phi = np.empty(N_max)  # angle of the ant to world coordinates!
        self.phi[:] = np.nan  # (NaN if not occupied)

        self.r_att = None
        self.r_det = None
        self.r_con = None
        self.r_orient = None
        self.r_tot = None

    @property
    def attachment_sites(self):
        return self._attachment_sites

    @attachment_sites.setter
    def attachment_sites(self, value):
        raise AttributeError('Denied')

    @property
    def phi_default_load_coord(self):
        return self._phi_default_load_coord

    @phi_default_load_coord.setter
    def phi_default_load_coord(self, value):
        raise AttributeError('Denied')

    def is_occupied(self, *i):
        if len(i) > 0:
            return self.n_p[i[0]] or self.n_l[i[0]]
        else:
            return np.any([self.n_p, self.n_l], axis=0)

    def f_loc(self, my_load, i: int):
        """
        :param my_load: b2Body, which is carried by ants
        :param i: ith attachment position
        :return: Linear velocity of my_load at the attachment site (b2Vec) in world coordinates so that they will oppose
        rotations
        """
        f_x, f_y = gamma * np.array(my_load.linearVelocity) \
                   + 0.7 * \
                   np.cross(np.hstack([self.attachment_site_world_coord(my_load, i), [0]]),
                            np.array([0, 0, my_load.angularVelocity]))[:2]
        # return my_load.GetLinearVelocityFromLocalPoint(self.attachment_position(my_load, i))
        return f_x, f_y

    def attachment(self, i: int, my_load, ant_type: str):
        if ant_type == 'puller':
            f_x, f_y = self.f_loc(my_load, i)
            self.n_p[i] = 1
            self.phi[i] = np.arctan2(f_y, f_x)
            # When a puller ant is attached to the cargo she contributes
            # to the cargo’s velocity by applying a force, and gets aligned as much as possible with the
            # direction of the local force at its point of attachment.

        else:
            self.n_l[i] = 1
            self.phi[i] = self.phi_default_load_coord[i] + my_load.angle
            # If a lifter ant attaches, she aligns with the outgoing normal of her attachment site

    def detachment(self, i: int):
        if not self.is_occupied(i):
            raise ValueError('Detachment at empty site')
        self.n_p[i] = 0
        self.n_l[i] = 0
        self.phi[i] = np.NaN

    def number_attached(self):
        return np.sum(self.is_occupied())

    def number_empty(self):
        return N_max - np.sum(self.is_occupied())

    def attachment_site_world_coord(self, my_load, i: int):
        """
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: the ith attachment site on the object
        :return: the attachment position in world coordinates.
        """
        return my_load.position + np.dot(rot(my_load.angle), self.attachment_sites[i])

    def normal_site_vector(self, angle: float, i: int):
        """
        :param i: ith position, counted counter clockwise
        :param angle: angle of the shape to the world coordinate system (my_load.angle)
        :return: ant vector pointing in the direction that the ant is pointing in world coordinate system
        """
        vector = np.array([np.cos(self.phi_default_load_coord[i]), np.sin(self.phi_default_load_coord[i])])
        return np.dot(rot(angle), vector)

    def ant_vector(self, angle: float, i: int):
        """
        :param i: ith position, counted counter clockwise
        :param angle: angle of the shape to the world coordinate system (my_load.angle)
        :return: ant vector pointing in the direction that the ant is pointing in world coordinates
        """
        vector = np.array([np.cos(self.phi[i]), np.sin(self.phi[i])])
        return np.dot(rot(angle), vector)

    # TODO: Check the code from here onward.
    def ant_force(self, my_load, i: int, pause=False):
        """
        Updates my_loads linear and angular velocity and returns the force vector in world coordinate system
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: ith position, counted counter clockwise
        :param pause: boolean. Whether the force should be applied, or we just return current force vector
        :return: force vector (np.array) pointing along the body axis of the ant at the ith position
        """
        if not self.n_p[i]:
            raise ValueError('force of a non puller site')

        force = np.array(f_0 / gamma * self.ant_vector(my_load.angle, i))

        # equations (4) and (5) from the SI
        if not pause:
            vectors = (self.attachment_site_world_coord(my_load, i),
                       self.ant_vector(my_load.angle, i))
            my_load.linearVelocity = my_load.linearVelocity + f_0 / gamma * np.inner(*vectors)
            my_load.angularVelocity = my_load.angularVelocity + f_0 / gamma_rot * np.cross(*vectors)
        # TODO: update angular velocity
        return force

    def whatsNext(self, my_load):
        """
        Decides the new change in the ant configuration. Randomly according to calculated probabilities one of the
        following events occur:
        (1) attachment of a new ant
        (2) detachment of an attached ant
        (3) conversion of a puller to lifter, or lifter to puller
        (4) reorientation of an ant to pull in the current direction of motion
        :param my_load: b2Body
        :return:
        """
        lot = np.random.uniform(0, 1)
        self.update_rates(my_load)

        if lot < self.r_att / self.r_tot:
            i = np.random.choice(np.where([not occ for occ in self.is_occupied()])[0])
            self.new_attachment(i, my_load)

        elif lot < (self.r_att + self.r_det) / self.r_tot:
            i = np.random.choice(np.where(self.is_occupied())[0])
            self.detachment(i)

        elif lot < (self.r_att + self.r_det + self.r_con) / self.r_tot:
            def rl_p(ii):
                return k_c * np.exp(np.inner(self.normal_site_vector(my_load.angle, ii),
                                             self.f_loc(my_load, ii)) / f_ind)

            def rp_l(ii):
                rp_l = k_c * np.exp(-np.inner(self.normal_site_vector(my_load.angle, ii),
                                              self.f_loc(my_load, ii)) / f_ind)
                if np.isnan(rp_l):
                    k_c * np.exp(-np.inner(self.normal_site_vector(my_load.angle, ii),
                                           self.f_loc(my_load, ii)) / f_ind)
                return rp_l

            prob_unnorm = [self.n_p[ii] * rp_l(ii) + self.n_l[ii] * rl_p(ii) for ii in np.where(self.is_occupied())[0]]
            i = np.random.choice([ii for ii in np.where(self.is_occupied())[0]],
                                 p=prob_unnorm * 1 / np.sum(prob_unnorm))
            if not (self.n_p[i] or self.n_l[i]) or (self.n_p[i] and self.n_l[i]):
                raise ValueError('Switching is messed up!')
            self.n_p[i], self.n_l[i] = self.n_l[i], self.n_p[i]

        else:
            i = np.random.choice(np.where(self.n_p)[0])
            f_x, f_y = self.f_loc(my_load, i)
            self.phi[i] = np.arctan2(f_y, f_x)
        return self.dt()

    def populate(self, my_load):
        self.new_attachment(0, my_load, ant_type='puller')
        self.new_attachment(2, my_load, ant_type='puller')
        self.new_attachment(4, my_load, ant_type='lifter')

    def new_attachment(self, i: int, my_load, ant_type=None):
        """
        A new ant attaches, and becomes either puller or lifter

        :param i: ith attachment position
        :param my_load: Box2D body
        :param ant_type: str, 'puller' or 'lifter' or None
        :return:
        """
        if self.is_occupied(i):
            raise ValueError('Ant tried to attach to occupied site')
        if ant_type is not None and ant_type not in ['puller', 'lifter']:
            raise ValueError('unknown type of ant... not puller or lifter')
        if ant_type is None:
            puller = 1 / (1 + np.exp(
                -np.inner(self.normal_site_vector(my_load.angle, i), self.f_loc(my_load, i)) / f_ind))
            if np.random.uniform(0, 1) < puller:
                ant_type = 'puller'
            else:
                ant_type = 'lifter'

        self.attachment(i, my_load, ant_type)
        return

    def update_rates(self, my_load):
        """

        :param my_load: Box2D body
        :return:
        """
        self.r_att = k_on * n_av * self.number_empty()
        self.r_det = np.sum([np.sum(self.is_occupied()) * (k1_off * np.heaviside(self.f_loc(my_load, i), 0)
                                                           + k2_off * (1 - np.heaviside(self.f_loc(my_load, i), 0)))
                             for i in range(N_max) if self.is_occupied(i)])
        # TODO: The rates of attachment and detachment are independent of the orientation
        #  with respect to the local force. (??)
        self.r_con = k_c * np.sum([self.n_p[i] * np.exp(-np.inner(self.normal_site_vector(my_load.angle, i),
                                                                  self.f_loc(my_load, i)) / f_ind)
                                   + self.n_l[i] * np.exp(-np.inner(self.normal_site_vector(my_load.angle, i),
                                                                    self.f_loc(my_load, i)) / f_ind)
                                   for i in np.where(self.is_occupied())[0]])
        self.r_orient = k_c * np.sum(self.n_p)
        self.r_tot = self.r_att + self.r_det + self.r_con + self.r_orient

    def dt(self):
        return -1 / self.r_tot * np.log(np.random.uniform(0, 1))
