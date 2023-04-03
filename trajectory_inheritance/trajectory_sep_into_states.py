from trajectory_inheritance.trajectory import Trajectory_part
import numpy as np

states = {'ab', 'ac', 'b', 'be', 'b1', 'b2', 'c', 'cg', 'e', 'eb', 'eg', 'f', 'g', 'h'}

columns = ['filename', 'size', 'solver', 'state', 'frames', 'pL', 'time']


class Traj_sep_by_state:
    def __init__(self, traj, ts):
        self.traj = traj
        self.ts = ts

        state_change_indices, states = self.find_state_change_indices()
        self.traj_parts = [Trajectory_part(self.traj, indices=range(*inds), VideoChain=[], tracked_frames=[],
                                           parent_states=ts)
                           for inds in state_change_indices]
        self.states = states

    def get_state(self, wanted_state: str, number=None):
        """
        Get the trajectory part with the given state
        :param state: state of the trajectory part
        :param number: number of the trajectory part with the given state
        :return: Trajectory_part
        """
        traj_parts = [traj_part for traj_part, state in
                      zip(self.traj_parts, self.states) if state == wanted_state]
        if number is not None:
            return traj_parts[number]
        return traj_parts

    def get_states(self, wanted_states: list) -> list:
        """
        Get the trajectory part with the given states
        :param state: state of the trajectory part
        :param number: number of the trajectory part with the given state
        :return: Trajectory_part
        """
        traj_p = []
        for i, (traj_part, state) in enumerate(zip(self.traj_parts, self.states)):
            if state in wanted_states:
                if i > 0:
                    if traj_p[-1][-1][1] + 1 == i:
                        traj_p[-1].append((traj_part, i))
                    else:
                        traj_p.append([(traj_part, i)])
                else:
                    traj_p.append([(traj_part, i)])

        list_of_traj_parts = []
        for t in traj_p:
            list_of_traj_parts.append(t[0][0])
            for traj_part, _ in t[1:]:
                list_of_traj_parts[-1] = list_of_traj_parts[-1] + traj_part
        return list_of_traj_parts

    def percent_of_succession1_ended_like_succession2(self, succession1, succession2):
        """
        Get the percentage of trajectory parts with the given succesion of states that ended like the given succession
        :param succesion1: succesion of states
        :param succesion2: succesion of states
        :return: percentage
        """
        if succession1 != succession2[:len(succession1)]:
            raise ValueError('The first succession should be the first part of the second succession')
        traj_parts1 = self.get_successions_of_states(succession1)
        traj_parts2 = self.get_successions_of_states(succession2)
        if len(traj_parts1) == 0:
            return None
        return len(traj_parts2) / len(traj_parts1)

    def get_successions_of_states(self, succession: list) -> list:
        """
        Get the trajectory parts with the given succession of states
        :param succession: succession of states
        :return: list of Trajectory_parts
        """
        result = []
        beginnings = [i for i, state in enumerate(self.states) if state in succession[0]]
        for i in beginnings:
            traj_parts, states = [], []
            succession_copy = succession.copy()
            for state_name, t in zip(self.states[i:], self.traj_parts[i:]):
                if len(succession_copy) > 0:
                    if state_name in succession_copy[0]:
                        traj_parts.append(t)
                        states.append(state_name)
                    elif len(succession_copy) > 1 and state_name in succession_copy[1]:
                        traj_parts.append(t)
                        states.append(state_name)
                        succession_copy.pop(0)
                    elif len(succession_copy) == 1 and state_name not in succession_copy[0]:
                        result.append((traj_parts, states))
                        succession_copy.pop(0)
                else:
                    break
        return result

    def find_state_change_indices(self):
        states = []
        state_change_indices = [[0]]

        for i in range(1, len(self.ts)):
            if self.ts[i] != self.ts[i - 1]:
                state_change_indices[-1].append(i)
                state_change_indices.append([i])
                states.append(self.ts[i - 1])
        state_change_indices[-1].append(len(self.ts)-1)
        states.append(self.ts[-2])
        return state_change_indices, states

    @staticmethod
    def extend_time_series_to_match_frames(ts, traj):
        indices_to_ts_to_frames = np.cumsum([1 / (int(len(traj.frames) / len(ts) * 10) / 10)
                                             for _ in range(len(traj.frames))]).astype(int)
        ts_extended = [ts[min(i, len(ts) - 1)] for i in indices_to_ts_to_frames]
        return ts_extended



