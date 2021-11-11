from trajectory_inheritance.trajectory import get


if __name__ == '__main__':
    x = get('large_20210805171741_20210805172610')
    x.load_participants()
    x.smooth()
    x.play(step=5)
