

def mainGame(x, my_maze, interval=1, display=None):
    i = 0
    while i < len(x.frames) - 1 - interval:
        x.step(my_maze, i, display)
        i += interval
        if display is not None:
            display.renew_screen(frame=x.frames[i], movie_name=x.filename)
            end = display.update_screen(x, i)
            if end:
                display.end_screen()
                x.frames = x.frames[:i]
                break
    return x
