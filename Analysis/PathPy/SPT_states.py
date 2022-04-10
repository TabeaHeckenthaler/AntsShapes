import string

states = ['0', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'j']

forbidden_transition_attempts = ['be', 'bf', 'bg',
                                 'di',
                                 'eb', 'ei',
                                 'fb', 'fi',
                                 'gb',
                                 'id', 'ie', 'if']

allowed_transition_attempts = ['ab', 'ad',
                               'ba',
                               'de', 'df', 'da',
                               'ed', 'eg',
                               'fd', 'fg',
                               'gf', 'ge', 'gj',
                               'ij',
                               'jg', 'ji']

cc_to_keep = 8
final_state = string.ascii_lowercase[cc_to_keep-1]
pre_final_state = string.ascii_lowercase[cc_to_keep-2]
