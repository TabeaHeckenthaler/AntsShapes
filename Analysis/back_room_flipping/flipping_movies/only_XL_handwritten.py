from trajectory_inheritance.get import get
# TEST the hinging
#
# ALL turned in the same direction!!!!
# Maybe the paper was not quite flat?
# Maybe there was a strong pheromone scent in the experiment?
# Maybe they are attracted to boundaries? ...
# that would facilitate turning around if the long side moves towards a boundary

# bias_direction = {'463': 'right', '464': 'right', '504': 'left',   # XL
#                   '442': 'right', '465': 'right', '466': 'right', '467': 'right', '501': 'right', '503':
#                   'right'  # L
#                   }

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640001_XLSpecialT_1_ants (part 1)')
traj.play(bias='right')
traj.play(frames=[33000, 35000], bias='right')

# looks like they understand, what turning around means & hinging
traj = get('XL_SPT_4640005_XLSpecialT_1_ants (part 1)')
traj.play(frames=[35000, 37000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640007_XLSpecialT_1_ants (part 1)')
traj.play(frames=[44027, 48027])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640009_XLSpecialT_1_ants (part 1)')
traj.play(frames=[18442, 22442])
traj.play(frames=[24887, 28887])
traj.play(frames=[78777, 82777])

# looks like they understand, what turning around means & hinging
traj = get('XL_SPT_4640012_XLSpecialT_1_ants')
traj.play(frames=[49000, 52000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640013_XLSpecialT_1_ants')
traj.play(frames=[9000, 12000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640014_XLSpecialT_1_ants')
traj.play(frames=[30000, 32000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640015_XLSpecialT_1_ants')
traj.play(frames=[38009, 46000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640018_XLSpecialT_1_ants')
traj.play(frames=[18000, 23000])

# looks like they understand, what turning around means (hinging?)
traj = get('XL_SPT_4640020_XLSpecialT_1_ants')
traj.play(frames=[13000, 15000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640021_XLSpecialT_1_ants (part 1)')
traj.play(frames=[32000, 38000])

# looks like they are attracted to specific part on the wall
traj = get('XL_SPT_4640023_XLSpecialT_1_ants')
traj.play(frames=[47000, 51000])

# looks like they understand, what turning around means (no hinging)
traj = get('XL_SPT_4640024_XLSpecialT_1_ants (part 1)')
traj.play(frames=[55701, 59701])

# ## ANOTHER DAY, OPPOSITE PART ON WALL!!!
# looks like they are attracted to specific part on the wall
traj = get('XL_SPT_5040003_XLSpecialT_1_ants (part 1)')
traj.play(frames=[23234 - 1000, 23234 + 2000])

# looks like they are attracted to specific part on the wall
traj = get('XL_SPT_5040006_XLSpecialT_1_ants (part 1)')
traj.play(frames=[21613 - 1000, 21613 + 2000])

# looks like they are attracted to specific part on the wall
traj = get('XL_SPT_5040008_XLSpecialT_1_ants (part 1)')
traj.play(frames=[3601 - 3000, 3601 + 2000])

# looks like they are attracted to specific part on the wall
traj = get('XL_SPT_5040010_XLSpecialT_1_ants (part 1)')
traj.play(frames=[3601 - 3000, 3601 + 2000])

traj = get('XL_SPT_5040012_XLSpecialT_1_ants')
traj.play(frames=[12296 - 1000, 12296 + 2000])

traj = get('XL_SPT_5040013_XLSpecialT_1_ants')
traj.play(frames=[12237 - 1000, 12237 + 3000])

DEBUG = True