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
        display = Display(my_maze, x.filename, wait=wait)

    """
    --- main game loop ---
    """
    end = False
    i = 0  # This tells us which movie the experiment is part of, 'i' is loop counter

    # Loop that runs the simulation...
    while True:
        x.step(my_load, i, my_maze=my_maze, display=display, **kwargs)
        i += interval

        if i >= len(x.frames) - 1 - interval or end:
            display.end_screen()
            break

        """ Display the frame """
        if display:
            end = display.update_screen(my_load, x, i)
    return x
