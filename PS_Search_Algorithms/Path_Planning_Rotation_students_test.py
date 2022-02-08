import unittest

from PS_Search_Algorithms.Path_Planning_Rotation_students import *
import pandas as pd
from ConfigSpace.ConfigSpace_Maze import ConfigSpace


# Zeile, Spalte
binned_space = pd.read_excel(io=dir, sheet_name='binned_space').to_numpy()
cs = ConfigSpace(conf_space)
binned_cs = Binned_ConfigSpace(config_space=cs, resolution=resolution)
Planner = Path_Planning_Rotation_students(conf_space=ConfigSpace(space=conf_space),
                                          start=Node2D(0, 0, conf_space),
                                          end=Node2D(8, 8, conf_space),
                                          resolution=resolution)


class Path_PlanningTest(unittest.TestCase):
    def test_calculate_binned_space(self):
        self.assertEqual(binned_cs.binned_space, binned_space)

    def test_bin_cut_out(self):
        self.assertEqual(binned_cs.bin_cut_out([(0, 0), (2, 3)]), binned_space[0:4, 1:3])

    def test_find_path(self):
        self.assertEqual(binned_cs.find_path((3, 0), (0, 5))[0], True)

    def test_initialize_speed(self):
        self.assertEqual(Planner.speed, binned_cs.binned_space)

    def test_add_knowledge(self):
        self.assertEqual(

        )
