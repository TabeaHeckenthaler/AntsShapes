from StateMachine import *


class RotemsStateMachine(StateMachine):
    def __init__(self, numPoints=5):
        super().__init__(numPoints=numPoints, final_node=numPoints-1)
        self.seeds, self.gen = None, None
        self.points = None, None
        self.markovianMatrix = self.get_markovianMatrix()

    def update_state(self):
        """
        Update the state of the system
        """
        # find next node to go to according to gradient descent in graph
        global curr_i, last_i
        p = np.copy(self.markovianMatrix[curr_i])
        temp = self.gen.choice(self.markovianMatrix.shape[1], p=p)
        last_i = curr_i
        curr_i = temp
        # print(p)
        # print(states[last_i + 1], ' -> ', states[curr_i + 1])
        return curr_i, last_i

    def animate(self):
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        ax.plot(self.seeds[:, 0], self.seeds[:, 1], 'ro')
        points_plot = ax.plot(self.points[:, 0], self.points[:, 1], 'o')[0]
        dude_plot = ax.plot([self.points[0, 0]], [self.points[0, 1]], 'o', markersize=10)[0]

        ax.set_xbound(-1, 1)
        ax.set_ybound(-1, 1)

        plt.tight_layout()

    def getPoints(self, numPoints, rndSeed) -> list:
        """
        Get points for the state machine
        :param numPoints: number of points to generate
        :param rndSeed: random seed
        """
        def calc_score(point, seeds):
            x, y = (self.seeds - point).T
            return np.sum(np.exp(-10 * np.hypot(x, y)))
        points = []

        for i in range(numPoints):
            best_score = -np.inf
            best_point = None
            for j in range(numPoints):
                point = self.gen.random((2,)) * 2 - 1
                score = calc_score(point, self.seeds)
                if score >= best_score:
                    best_score = score
                    best_point = point
            points.append(best_point)

        points = np.array(points, float)  # these are the states
        return points

    # def get_markovianMatrix(self):
    #     move_mat = np.array(initial_Markovian_Matrix)
    #     move_mat = move_mat / np.sum(move_mat, 1)[:, np.newaxis]
    #     return move_mat

    def get_markovianMatrix(self, numPoints=5):
        """
        Get the markovian matrix
        """
        self.points = self.getPoints(numPoints, 42)

        l = self.points.shape[0]
        deltas = np.repeat(self.points, l, 0).reshape((l, l, 2))
        deltas -= deltas.swapaxes(0, 1)
        dists = self.distances(self.points)

        h = np.hypot(deltas[..., 0], deltas[..., 1])[..., np.newaxis]
        h[h == 0] = np.inf
        deltas = deltas / h

        move_mat = np.reciprocal(dists)
        move_mat *= np.sqrt(1 + deltas[..., 0])
        move_mat[move_mat == np.inf] = 0
        move_mat /= np.sum(move_mat, 1)[:, np.newaxis]
        return move_mat

    # def save_movie(self, animation):
    #     plt.rcParams['animation.ffmpeg_path'] = u'C:\\ffmpeg\\bin\\ffmpeg.exe'
    #     # Writer = animation.writers['ffmpeg'] #
    #     writer = animation.FFMpegWriter()
    #     writer = Writer(fps=30, metadata=dict(artist='Rotem Shalev'), bitrate=1500)
    #
    #     print("Saving")
    #     anim.save("Langevin_Vid_Thingy.mp4", writer=writer)
    #     print("Done.")

# rotemsStateMachine = RotemsStateMachine()
# rotemsStateMachine.run()
# anim = rotemsStateMachine.animate()
# rotemsStateMachine.save_movie(anim)
