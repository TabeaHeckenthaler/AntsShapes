
a = {1: {3: 4}}


def flatten_dict(dict1):
    new = {}
    for key1, dict2 in dict1.items():
        for key2, value in dict2.items():
            new[str(key1) + '_' + str(key2)] = value
    return new


print(flatten_dict(a))