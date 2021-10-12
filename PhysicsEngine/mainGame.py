"""self written functions"""
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Display import Display


def mainGame(x, interval=1, display=False, PhaseSpace=None, ps_figure=None, wait=0, free=False, **kwargs):
    """
    Start instantiating the World and the load...
    """
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver, free=free)
    my_load = Load(my_maze)
    if display:
        display = Display(x, my_maze, wait=wait)

    """
    --- main game loop ---
    """
    i = 0
    while i < len(x.frames) - 1 - interval:
        x.step(my_load, i, my_maze=my_maze, display=display, **kwargs)
        i += interval

        if display:
            end = display.update_screen(my_load, x, i)
            if end:
                display.end_screen()
                x.frames = x.frames[:i]
                break

    return x
