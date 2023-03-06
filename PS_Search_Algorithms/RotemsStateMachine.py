import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import networkx as nx

curr_i = 0
last_i = 0

states = {0: 'ab', 1: 'ac', 2: 'b',
          3: 'be', 4: 'b1', 5: 'b2', 6: 'c',
          7: 'cg', 8: 'e', 9: 'eb',
          10: 'eg', 11: 'f',
          12: 'g', 13: 'h', 14: 'i'}

# initial_Markovian_Matrix = pd.read_excel("initialMarkovianMatrix.xlsx", index_col=0)
connectionMatrix = pd.read_excel("ConnectionMatrix.xlsx", index_col=0).to_numpy(dtype=bool)
percievedConnectionMatrix = pd.read_excel("PercievedConnectionMatrix.xlsx", index_col=0).to_numpy(dtype=bool)
# check whether the matrix is symmetric
# assert (initial_Markovian_Matrix == initial_Markovian_Matrix.T).all().all()
DEBUG = 0

class StateMachine:
    def __init__(self, numPoints=len(states), final_node=14):
        self.seeds, self.gen = self.getSeeds(numPoints, 42)
        self.points = self.getPoints()
        self.distanceMatrix = None
        self.G = self.create_graph()
        self.final_node = 14
        self.paths = []
        self.anim = None

    def create_graph(self):
        self.distanceMatrix = self.distances(self.points)
        self.distanceMatrix[~percievedConnectionMatrix] = np.inf

        G = nx.Graph()
        G.add_nodes_from(states.keys())
        G.add_weighted_edges_from([(i, j, self.distanceMatrix[i, j])
                                   for i in range(len(states)) for j in range(i + 1, len(states))])
        edge_costs = {}
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                edge_costs[(i, j)] = self.distanceMatrix[i, j]
        nx.set_edge_attributes(G, edge_costs, 'cost')
        return G


    def getPoints(self, *args) -> list:
        # read exel file
        df = pd.read_excel("points_states_in_image.xlsx", index_col=0)
        return np.array(df[['x', 'y']])

    @staticmethod
    def getSeeds(numPoints, rndSeed):
        gen = np.random.default_rng(rndSeed)
        seeds = gen.random((numPoints, 2)) * 2 - 1
        return seeds, gen

    def update_state(self):
        """
        Update the state of the system
        """
        # find next node to go to according to gradient descent in graph
        global curr_i, last_i
        shortest_path = nx.shortest_path(self.G, source=curr_i, target=self.final_node, weight='cost')
        last_i = curr_i

        if len(shortest_path) > 1:
            if connectionMatrix[shortest_path[0], shortest_path[1]]:
                curr_i = shortest_path[1]
                self.paths[-1].append(curr_i)
            else:
                self.G.remove_edge(shortest_path[0], shortest_path[1])
                self.distanceMatrix[shortest_path[0], shortest_path[1]] = np.inf
                self.distanceMatrix[shortest_path[1], shortest_path[0]] = np.inf
                print('updated edges', states[shortest_path[0]], states[shortest_path[1]])
        else:
            curr_i = shortest_path[0]
            if self.anim is not None:
                self.anim.event_source.stop()
        return curr_i, last_i

    def run(self, num_traj=3):
        for i in range(num_traj):
            print(i)
            global curr_i, last_i
            curr_i = 0
            last_i = 0
            self.paths.append([curr_i])
            self.G = self.create_graph()
            while curr_i != self.final_node:
                curr_i, last_i = self.update_state()
    def save(self):
        with open("states.txt", 'w') as f:
            f.write("\n".join(",".join(str(x) for x in statelis) for statelis in self.paths))
            f.write("\n\n")
            f.write("\n".join(",".join(str(states[x]) for x in statelis) for statelis in self.paths))

    def animate(self):
        img = plt.imread("CS_image.png")
        # plt.style.use('dark_background')
        fig, ax = plt.subplots()

        # show img in background of fig
        im = ax.imshow(img)

        # ax.plot(self.seeds[:, 0], self.seeds[:, 1], 'ro')
        points_plot = ax.plot(self.points[:, 0], self.points[:, 1], 'o')[0]
        dude_plot = ax.plot([self.points[0, 0]], [self.points[0, 1]], 'o', markersize=10)[0]

        # ax.set_xbound(-1, 1)
        # ax.set_ybound(-1, 1)

        plt.tight_layout()
        def other_update_anim(frame):
            print(states[curr_i])
            self.update_state()
            dude_plot.set_data([self.points[curr_i, 0]],
                               [self.points[curr_i, 1]])
            return dude_plot,

        self.anim = animation.FuncAnimation(fig, other_update_anim, interval=100, repeat=False, frames=range(20))
        plt.show()

    @staticmethod
    def distances(points):
        l = points.shape[0]
        deltas = np.repeat(points, l, 0).reshape((l, l, 2))
        deltas -= deltas.swapaxes(0, 1)
        dists = np.hypot(deltas[..., 0], deltas[..., 1])[..., np.newaxis]
        dists = dists[:, :, 0]  # find distances between points
        return dists

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

stateMachine = StateMachine()
stateMachine.run()
stateMachine.save()
# stateMachine.animate()
print(stateMachine.paths)

# TODO: Add a check, whether the nodes are connected or not.
# TODO: Allow the network to update itself when it tries to move along a false connection


DEBUG = 1