from trajectory_inheritance.trajectory import get

interpolate_list = ['M_SPT_4690009_MSpecialT_1_ants',
                    'M_SPT_4700022_MSpecialT_2_ants (part 1)',
                    'S_SPT_4720005_SSpecialT_1_ants (part 1)',
                    'S_SPT_4720014_SSpecialT_1_ants',
                    'S_SPT_4750005_SSpecialT_1_ants (part 1)',
                    'S_SPT_4750014_SSpecialT_1_ants (part 1)',
                    'S_SPT_4750016_SSpecialT_1_ants',
                    'S_SPT_4770012_SSpecialT_1_ants (part 1)',
                    'S_SPT_4780002_SSpecialT_1_ants',
                    'S_SPT_4790005_SSpecialT_1_ants (part 1)']


for filename in interpolate_list:
    x = get(filename)
    exclude = [7702 - (6895 - 1592), 7763 - (6895 - 1592)]
    new = x.interpolate(exclude)
