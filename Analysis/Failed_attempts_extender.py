
class FailedAttempt:
    def __init__(self, name, path_length, time, final_state):
        self.name = name
        self.path_length = path_length
        self.time = time
        self.final_state = final_state

    def mean_speed(self) -> float:
        return self.path_length/self.time


class FailedAttemptPathLengthExtender:
    def __init__(self, failedAttempt, fundamental_matrix):
        """
        fundamental matrix must contain self_loops, or I have to define mean time spent in a certain state.
        """
        self.failedAttempt = failedAttempt
        self.fundamental_matrix = fundamental_matrix  # TODO

    def expected_solving_time(self) -> float:
        self.fundamental_matrix
        return float()

    def expected_additional_path_length(self) -> float:
        return self.failedAttempt.mean_speed() * self.expected_solving_time()
