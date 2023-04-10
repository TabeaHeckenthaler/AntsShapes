from PS_Search_Algorithms.human_network_simulation.StateMachine import HumanStateMachine
import json
from Directories import home
import numpy as np
import pandas as pd


class ManyStateMachines:
    def __init__(self, n):
        self.n = n
        self.stateMachines = [HumanStateMachine(seed=i) for i in range(self.n)]
        self.initial_state = 'ab'
        self.path = [self.initial_state]

    def run_most_commonChoice(self):
        curr_i = 'ab'
        while curr_i != 'i':
            first_choices = [stateMachine.shortest_path(curr_i)[1] for stateMachine in self.stateMachines]
            communal_choice = max(set(first_choices), key=first_choices.count)
            print(communal_choice)
            print(first_choices)
            if HumanStateMachine.is_connected(curr_i, communal_choice):
                curr_i = communal_choice
                self.path.append(communal_choice)
            else:
                [stateMachine.weaken_connection(curr_i, communal_choice, False) for stateMachine in self.stateMachines]

    def run_randomChoice(self):
        curr_i = 'ab'
        while curr_i != 'i':
            first_choices = [stateMachine.shortest_path(curr_i)[1] for stateMachine in self.stateMachines]
            randomChoice = np.random.choice(first_choices)
            print(randomChoice)
            print(first_choices)
            if HumanStateMachine.is_connected(curr_i, randomChoice):
                curr_i = randomChoice
                self.path.append(randomChoice)
            else:
                [stateMachine.weaken_connection(curr_i, randomChoice, False) for stateMachine in self.stateMachines]

    @staticmethod
    def save(paths, string="states_many.json"):
        with open(string, 'w') as f:
            json.dump(paths, f)

    @classmethod
    def df(cls, size, name):
        with open(home + "\\PS_Search_Algorithms\\human_network_simulation\\" + name + ".json", 'r') as f:
            paths = json.load(f)
        # create dataframe with columns: name,
        tuples = [(name, size, True, path) for name, path in enumerate(paths)]

        df = pd.DataFrame(tuples, columns=['name', 'size', 'winner', 'states_series'])
        return df


if __name__ == '__main__':
    paths = []
    part = 20
    for i in range(20):
        many = ManyStateMachines(part)
        many.run_most_commonChoice()
        paths.append(many.path)
    ManyStateMachines.save(paths, string="states_randomChoice_" + str(part) + '.json')
    DEBUG = 1