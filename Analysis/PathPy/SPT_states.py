import string

cc_to_keep = 8
final_state = string.ascii_lowercase[cc_to_keep]
pre_final_state = string.ascii_lowercase[cc_to_keep - 1]

states = ['0'] + list(string.ascii_lowercase[:cc_to_keep]) + ['i']

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

all_states = states + forbidden_transition_attempts + allowed_transition_attempts
