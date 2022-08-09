from trajectory_inheritance.get import get

if __name__ == '__main__':
    # filename = "L_SPT_4670008_LSpecialT_1_ants (part 1)"
    # x = get(filename)
    # x.play(frames=[int(144262-500), -1], wait=200)
    d = 1

filename = 'XL_SPT_4640023_XLSpecialT_1_ants'
x = get(filename)
print(x.winner)
x.play(step=5)
x.play(frames=[-2000, -1], step=5, wait=10)
