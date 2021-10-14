"""self written functions"""
from PhysicsEngine.Display import Display


def mainGame(x, my_maze, interval=1, display=False, wait=0):
    """
    Start instantiating the World and the load...
    """
    my_load = my_maze.bodies[-1]

    if display:
        display = Display(x, my_maze, wait=wait)

    """
    --- main game loop ---
    """
    i = 0
    while i < len(x.frames) - 1 - interval:
        display.renew_screen(frame=x.frames[i], movie_name=x.filename)
        x.step(my_load, i, my_maze=my_maze, display=display)
        i += interval

        if display:
            end = display.update_screen(x, i)
            if end:
                display.end_screen()
                x.frames = x.frames[:i]
                break
    return x
