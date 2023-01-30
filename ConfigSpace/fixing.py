from ConfigSpace.ConfigSpace_Maze import *

# ____________________________________________________________________________________________
# ps = ConfigSpace_Labeled(solver='human', size='Small Far', shape='SPT',
#                          geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
# ps.load_eroded_labeled_space()
#
# mask = np.zeros_like(ps.space, dtype=bool)
# mask[242, 199, 84] = True
# mask = ps.dilate(mask, 100)
# new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'ba', mask),
#                          np.logical_and(ps.space_labeled == 'b', mask))
# ps.space_labeled[new_mask] = 'be'
#
# mask = np.zeros_like(ps.space, dtype=bool)
# mask[242, 131, 669] = True
# mask = ps.dilate(mask, 100)
# new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'be', mask),
#                          np.logical_and(ps.space_labeled == 'b', mask))
# ps.space_labeled[new_mask] = 'bd'
#
# # ____ add db
# mask = np.zeros_like(ps.space, dtype=bool)
# # indices_eb = np.where(ps.space_labeled == 'eb')
# # np.stack(indices_eb)[:, indices_eb[2] == max(indices_eb[2])]
# mask[243, 202, 105] = True
# mask = ps.dilate(mask, 150)
# new_mask = np.logical_and(ps.space_labeled == 'e', mask)
# ps.space_labeled[new_mask] = 'eb'
#
# mask = np.zeros_like(ps.space, dtype=bool)
# # indices_d = np.where(ps.space_labeled == 'd')
# # np.stack(indices_d)[:, indices_d[2] == max(indices_d[2])]
# mask[244, 129, 667] = True
# mask = ps.dilate(mask, 150)
# new_mask = np.logical_and(ps.space_labeled == 'd', mask)
# ps.space_labeled[new_mask] = 'db'
#
# ps.visualize_transitions(reduction=4, only_states=['be', 'eb', 'db', 'bd'])
# labels = np.unique(ps.space_labeled)
# ps.ps_states = {label: PS_Area(ps, np.bool_(ps.space_labeled == label), label)
#                 for label in labels}
# ps.save_labeled()


# ____________________________________________________________________________________________
ps = ConfigSpace_Labeled(solver='humanhand', size='', shape='SPT',
                         geometry=('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx'))
ps.load_eroded_labeled_space()
# # ps.visualize_transitions(reduction=4, only_states=['b', 'ab'])
# DEBUG = 1
mask = np.zeros_like(ps.space, dtype=bool)
# # indices_b = np.where(ps.space_labeled == 'b')
# # np.stack(indices_b)[:, indices_b[2] == max(indices_b[2])]
# indices = np.where(ps.space_labeled == 'd')
# indices = max(np.where(ps.space_labeled[:, :, 540] == 'd')[0])
# np.stack(indices)[:, indices[2] == max(indices[2])]
mask[183, 93, 555] = True
mask = ps.dilate(mask, 75)
new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'be', mask),
                         np.logical_and(ps.space_labeled == 'b', mask),
                         np.logical_and(ps.space_labeled == 'ba', mask))
ps.space_labeled[new_mask] = 'bd'

mask = np.zeros_like(ps.space, dtype=bool)
# indices_d = np.where(ps.space_labeled == 'd')
# np.stack(indices_d)[:, indices_d[2] == max(indices_d[2])]
mask[189, 85, 540] = True
mask = ps.dilate(mask, 100)
new_mask = np.logical_and(ps.space_labeled == 'd', mask)
ps.space_labeled[new_mask] = 'db'

mask = np.zeros_like(ps.space, dtype=bool)
# in_be = np.where(ps.space_labeled == 'be')
# np.stack(in_be)[:, in_be[0] == max(in_be[0])]
mask[183, 140, 69] = True
mask = ps.dilate(mask, 75)
new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'b', mask),
                         np.logical_and(ps.space_labeled == 'ba', mask))
ps.space_labeled[new_mask] = 'be'

mask = np.zeros_like(ps.space, dtype=bool)
# in_eb = np.where(ps.space_labeled == 'eb')
# np.stack(in_eb)[:, in_eb[2] == min(in_eb[2])]
mask[189, 154, 84] = True
mask = ps.dilate(mask, 100)
new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'eb', mask),
                         np.logical_and(ps.space_labeled == 'e', mask))
ps.space_labeled[new_mask] = 'eb'
DEBUG = 1
# ps.save_labeled()


# ____________________________________________________________________________________________
# ps = ConfigSpace_Labeled(solver='human', size='Large', shape='SPT',
#                          geometry=('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'))
# ps.load_eroded_labeled_space()
#
# mask = np.zeros_like(ps.space, dtype=bool)
# # # ind = np.where(ps.space_labeled == 'b')
# # # np.stack(ind)[:, ind[1] == min(ind[1])]
# mask[194, 103, 553] = True  # smallest in point in b in y
# mask = ps.dilate(mask, 75)
# new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'b', mask),
#                          np.logical_and(ps.space_labeled == 'ba', mask))
# ps.space_labeled[new_mask] = 'bd'
#
# mask = np.zeros_like(ps.space, dtype=bool)
# # # ind = np.where(ps.space_labeled == 'd')
# # # np.stack(ind)[:, ind[2] == max(ind[2])]
# mask[201, 94, 537] = True  # highest point in d
# mask = ps.dilate(mask, 100)
# new_mask = np.logical_and(ps.space_labeled == 'd', mask)
# ps.space_labeled[new_mask] = 'db'
#
#
# mask = np.zeros_like(ps.space, dtype=bool)
# # # ind = np.where(ps.space_labeled == 'be')
# # # np.stack(ind)[:, ind[1] == max(ind[1])]
# mask[194, 149, 63] = True
# mask = ps.dilate(mask, 75)
# new_mask = np.logical_or(np.logical_and(ps.space_labeled == 'b', mask),
#                          np.logical_and(ps.space_labeled == 'ba', mask))
# ps.space_labeled[new_mask] = 'be'
#
# mask = np.zeros_like(ps.space, dtype=bool)
# # # ind = np.where(ps.space_labeled == 'eb')
# # # np.stack(ind)[:, ind[2] == min(ind[2])]
# mask[201, 157, 79] = True
# mask = ps.dilate(mask, 100)
# new_mask = np.logical_and(ps.space_labeled == 'e', mask)
# ps.space_labeled[new_mask] = 'eb'