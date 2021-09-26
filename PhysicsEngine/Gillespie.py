# similar to the Gillespie Code described in the SI of the nature communications paper
# titled 'Ants optimally amplify... '
import numpy as np

f0 = 1
k_c, f_0, gamma, f_ind = 0.7, 1, 2000, 10 * f0
gamma_rot = 0.4 * gamma
k_on = 0.0017  # attachment independent of cargo velocity
k2_off, k1_off = 0.01, 0.035  # detachment rate from a moving cargo, detachment from a non moving cargo
f_kinx, f_kiny = 0, 0
sites = 20
n_av = 10  # number of available ants


class Gillespie:
    def __init__(self):
        self.n_p = [0 for _ in range(sites)]  # array with 0 and 1 depending on whether is puller or not
        self.n_l = [0 for _ in range(sites)]  # array with 0 and 1 depending on whether is lifter of not

        self._theta = np.linspace(0, 2 * np.pi, sites)  # angle of the attachment site to the 0th attachment site
        # only correct for circular objects

        self.phi = np.empty(sites)  # angle of the ant to the normal of her attachment site
        self.phi[:] = np.nan  # (NaN if not occupied)

        self.r_att = None
        self.r_det = None
        self.r_con = None
        self.r_orient = None
        self.r_tot = None

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        raise AttributeError('Denied')

    def f_loc(self, my_load, i):
        """
        :param my_load: b2Body, which is carried by ants
        :param i: ith attachment position
        :return: Linear velocity of my_load at the attachment site (b2Vec)
        """
        return my_load.GetLinearVelocityFromLocalPoint(self.attachment_position(my_load, i))

    def new_attachment(self, i, my_load, type=None):
        """
        A new ant attaches, and becomes either puller or lifter
        :param i: ith attachment position
        :param angle: my_load.angle during the run gives the current angle of the carried shape
        :return:
        """
        if self.is_occupied(i):
            raise ValueError('Ant tried to attach to occupied site')
        if type is not None and type not in ['puller', 'lifter']:
            raise ValueError('unknown type of ant... not puller or lifter')
        if type is None:
            puller = 1 / (1 + np.exp(-np.inner(self.ant_vector(i, my_load.angle), self.f_loc(my_load, i)) / f_ind))
            if np.random.uniform(0, 1) < puller:
                type = 'puller'
            else:
                type = 'lifter'

        if type == 'puller':
            self.n_p[i] = 1
            self.phi[i] = 0
        else:
            self.n_l[i] = 1
            self.phi[i] = 0
        return

    def detachment(self, i):
        if not self.is_occupied(i):
            raise ValueError('Detachment at empty site')
        self.n_p[i] = 0
        self.n_l[i] = 0
        self.theta[i] = np.NaN

    def number_attached(self):
        return np.sum(self.is_occupied())

    def number_empty(self):
        return sites - np.sum(self.is_occupied())

    def is_occupied(self, *i):
        if len(i) > 0:
            return self.n_p[i[0]] or self.n_l[i[0]]
        else:
            return np.any([self.n_p, self.n_l], axis=0)

    def attachment_position(self, my_load, i):
        """
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: the ith attachment site on the object
        :return: the attachment position in world coordinates.
        """
        from Box2D import b2CircleShape
        for fixture in my_load.fixtures:
            if isinstance(fixture.shape, b2CircleShape):
                return fixture.shape.radius * np.array([np.cos(self.theta[i] + my_load.angle),
                                                        np.sin(self.theta[i] + my_load.angle)]) \
                       + my_load.position

    def ant_vector(self, i, angle):
        """
        :param i: ith position, counted counter clockwise
        :param angle: angle of the shape to the world coordinate system (my_load.angle)
        :return: ant vector pointing in the direction that the ant is pointing in
        """
        if not isinstance(i, int):
            i = int(i)
        angle2 = self.phi[i] + self.theta[i] + angle
        return np.array([np.cos(angle2), np.sin(angle2)])

    def ant_force(self, my_load, i):
        """
        Updates my_loads linear and angular velocity and returns the force vector
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: ith position, counted counter clockwise
        :return: force vector (np.array) pointing along the body axis of the ant at the ith position
        """
        if not self.n_p[i]:
            raise ValueError('force of a non puller site')

        force = np.array(f_0 / gamma * self.ant_vector(i, my_load.angle))

        # equations (4) and (5) from the SI

        my_load.linearVelocity = my_load.linearVelocity \
                                 + [f_0 / gamma * np.cos(self.theta[i] + self.phi[i] + my_load.angle),
                                    f_0 / gamma * np.sin(self.theta[i] + self.phi[i] + my_load.angle)]
        my_load.angularVelocity = my_load.angularVelocity + f_0 / gamma_rot * np.sin(self.phi[i])  # TODO: update angular velocity
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
                return k_c * np.exp(np.inner(self.ant_vector(ii, my_load.angle),
                                             self.f_loc(my_load, ii)) / f_ind)

            def rp_l(ii):
                return k_c * np.exp(-np.inner(self.ant_vector(my_load.angle, ii),
                                              self.f_loc(my_load, ii)) / f_ind)

            prob_unnorm = [self.n_p[ii] * rp_l(ii) + self.n_l[ii] * rl_p(ii) for ii in np.where(self.is_occupied())[0]]
            if np.any([np.isnan(p) for p in prob_unnorm]):
                k = 1
            i = np.random.choice([ii for ii in np.where(self.is_occupied())[0]], p=prob_unnorm * 1/np.sum(prob_unnorm))
            if not (self.n_p[i] or self.n_l[i]) or (self.n_p[i] and self.n_l[i]):
                raise ValueError('Switching is messed up!')
            self.n_p[i], self.n_l[i] = self.n_l[i], self.n_p[i]

        else:
            i = np.random.choice(np.where(self.n_p)[0])
            self.phi[i] = np.arctan2(*self.f_loc(my_load, i)) - self.theta[i]
        return self.dt()

    def update_rates(self, my_load):
        """

        :param my_load: Box2D body
        :return:
        """
        self.r_att = k_on * n_av * self.number_empty()
        self.r_det = np.sum([np.sum(self.is_occupied()) * (k1_off * np.heaviside(self.f_loc(my_load, i), 0)
                                                           + k2_off * (1 - np.heaviside(self.f_loc(my_load, i), 0)))
                             for i in range(sites) if self.is_occupied(i)])
        self.r_con = k_c * np.sum([self.n_p[i] * np.exp(-np.inner(self.ant_vector(i, my_load.angle),
                                                                  self.f_loc(my_load, i)) / f_ind)
                                   + self.n_l[i] * np.exp(-np.inner(self.ant_vector(i, my_load.angle),
                                                                    self.f_loc(my_load, i)) / f_ind)
                                   for i in np.where(self.is_occupied())[0]])
        self.r_orient = k_c * np.sum(self.n_p)
        self.r_tot = self.r_att + self.r_det + self.r_con + self.r_orient

    def dt(self):
        return -1 / self.r_tot * np.log(np.random.uniform(0, 1))
