import numpy as np
from Analysis.PathPy.network_functions import Network
from DataFrame.dataFrame import myDataFrame


class FailedAttempt:
    def __init__(self, name, path_length, time, final_state):
        self.name = name
        self.path_length = path_length
        self.time = time
        self.final_state = final_state

    def mean_speed(self) -> float:
        return self.path_length/self.time


class FailedAttemptPathLengthExtender:
    """
    Markovian assumption: Extend the trajectory path length
    """
    def __init__(self, failedAttempt: FailedAttempt, diffusion_time: np.array):
        """
        fundamental matrix must contain self_loops, or I have to define mean time spent in a certain state.
        """
        self.failedAttempt = failedAttempt
        self.diffusion_time = diffusion_time

    def expected_solving_time(self) -> float:
        return self.diffusion_time[self.failedAttempt.final_state] * self.failedAttempt.mean_speed()

    def expected_additional_path_length(self) -> float:
        return self.failedAttempt.mean_speed() * self.expected_solving_time()


if __name__ == '__main__':
    filename = 'L_SPT_4650007_LSpecialT_1_ants (part 1)'
    exp = myDataFrame.loc[myDataFrame.filename == filename].squeeze()

    my_failed_attempt = FailedAttempt(exp.filename, exp.path_length, exp.time, exp.final_state)
    my_network = Network(exp.solver, exp.size, exp.shape)
    my_network.get_results()

    extender = FailedAttemptPathLengthExtender(my_failed_attempt, my_network.t)
    print(extender.expected_solving_time())
