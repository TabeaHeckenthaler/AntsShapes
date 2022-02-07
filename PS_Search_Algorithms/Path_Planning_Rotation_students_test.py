import unittest
import os

from PS_Search_Algorithms.Path_Planning_Rotation_students import *
import pandas as pd
from Directories import home
from ConfigSpace.ConfigSpace_Maze import ConfigSpace

dir = os.path.join(home, 'PS_Search_Algorithms', 'path_planning_test.xlsx')
space = pd.read_excel(io=dir, sheet_name='space').to_numpy()
binned_space = pd.read_excel(io=dir, sheet_name='binned_space').to_numpy()
resolution = 2


class Path_PlanningTest(unittest.TestCase):
    def test_calculate_binned_space(self):
        cs = ConfigSpace(space)
        binned_cs = Binned_ConfigSpace(config_space=cs, resolution=resolution)
        self.assertEqual(binned_cs.binned_space, binned_space)

    def test_bins_connected(self):
        self.assertEqual(

        )

    def test_find_path(self):
        self.assertEqual(

        )

    def test_step_to(self):
        self.assertEqual(

        )

    def test_initialize_speed(self):
        self.assertEqual(

        )

    def test_add_knowledge(self):
        self.assertEqual(

        )
