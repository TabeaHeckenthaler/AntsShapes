import string

cc_to_keep = 8
final_state = string.ascii_lowercase[cc_to_keep-1]
pre_final_state = string.ascii_lowercase[cc_to_keep-2]

states = ['0'] + list(string.ascii_lowercase[:cc_to_keep])

forbidden_transition_attempts = ['be', 'bd', 'bf',
                                 'cg',
                                 'eb', 'eg',
                                 'db', 'dg',
                                 'gc',
                                 'fb']

allowed_transition_attempts = ['ab', 'ac',
                               'ba',
                               'ce', 'cd', 'ca',
                               'ec', 'ef',
                               'dc', 'df',
                               'fd', 'fe', 'fh',
                               'gh',
                               'hf', 'hg']
