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


class BinningTest(unittest.TestCase):
    def test_calculate_binned_space(self):
        self.assertEqual(binned_cs.binned_space,
                         binned_space)

    def test_space_ind_to_bin_ind(self):
        self.assertEqual(binned_cs.space_ind_to_bin_ind((4, 2)),
                         (2, 1))

    def test_bin_cut_out(self):
        self.assertEqual(binned_cs.bin_cut_out([(0, 0), (2, 1)]),
                         ((0, 0), binned_space[0:4, 1:3]))

    def test_find_path(self):
        self.assertTrue(binned_cs.find_path((3, 0), (0, 5))[0])


class Path_PlanningTest(unittest.TestCase):
    def test_initialize_speed(self):
        self.assertEqual(Planner.speed, binned_cs.binned_space)

    def test_add_knowledge(self):
        start = (4, 2)
        end = (4, 5)
        self.assertFalse(binned_cs.find_path(start, end)[0])
        initial_speed = Planner.speed[binned_cs.space_ind_to_bin_ind(start)]
        Planner.add_knowledge(Node2D(*start, cs))
        final_speed = Planner.speed[binned_cs.space_ind_to_bin_ind(start)]
        self.assertLess(initial_speed, final_speed)

