from trajectory_inheritance.get import get

if __name__ == '__main__':
    filename = "S_SPT_4800006_SSpecialT_1_ants (part 1)"
    x = get(filename)
    x.stuck()
