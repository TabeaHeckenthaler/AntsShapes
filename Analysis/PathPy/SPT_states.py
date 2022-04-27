import string

cc_to_keep = 8
final_state = string.ascii_lowercase[cc_to_keep-1]
pre_final_state = string.ascii_lowercase[cc_to_keep-2]

states = ['0'] + list(string.ascii_lowercase[:cc_to_keep])

short_forbidden = ['bd', 'be', 'bf', 'cg', 'dg', 'eg']
forbidden_transition_attempts = short_forbidden + [s[::-1] for s in short_forbidden]
# forbidden_transition_attempts = ['be', 'bd', 'bf',
#                                  'cg',
#                                  'eb', 'eg',
#                                  'db', 'dg',
#                                  'gc', 'ge', 'gd'
#                                  'fb']

allowed_transition_attempts = ['ab', 'ac',
                               'ba',
                               'ce', 'cd', 'ca',
                               'ec', 'ef',
                               'dc', 'df',
                               'fd', 'fe', 'fh',
                               'gh',
                               'hf', 'hg']
