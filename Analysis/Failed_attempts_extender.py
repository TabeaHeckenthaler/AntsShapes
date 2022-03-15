import numpy as np


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
    def __init__(self, failedAttempt: FailedAttempt, expected_solving_times: np.array):
        """
        fundamental matrix must contain self_loops, or I have to define mean time spent in a certain state.
        """
        self.failedAttempt = failedAttempt
        self.expected_solving_times = expected_solving_times

    def expected_solving_time(self) -> float:
        return self.expected_solving_times[self.failedAttempt.final_state] * self.failedAttempt.mean_speed()

    def expected_additional_path_length(self) -> float:
        return self.failedAttempt.mean_speed() * self.expected_solving_time()
