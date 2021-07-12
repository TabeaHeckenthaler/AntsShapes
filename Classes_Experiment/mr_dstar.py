import re


def find_number(text, c):
    return re.findall(r"%s(\d+)" % c, text)


def sensing_radius(filename):
    l = find_number(filename, 'sensing')
    if len(l) == 0:
        return None
    return int(find_number(filename, 'sensing')[0])


def dil_radius(filename):
    l = find_number(filename, 'dil')
    if len(l) == 0:
        return None
    return int(find_number(filename, 'dil')[0])


def filename_dstar(size, shape, dil_radius, sensing_radius):
    return size + '_' + shape + '_' + 'dil' + str(dil_radius) + '_sensing' + str(sensing_radius)


class Mr_dstar:
    def __init__(self, filename):
        self.filename = filename
        self.sensing_radius = sensing_radius(filename)
        self.dil_radius = dil_radius(filename)
        return

    @staticmethod
    def averageCarrierNumber():
        return 1
