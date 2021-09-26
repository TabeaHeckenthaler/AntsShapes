"""self written functions"""
from Setup.Maze import Maze
from Setup.Load import Load
from PhysicsEngine.Display_Pygame import Display_screen, Pygame_EventManager, Display_end, Display_renew


def MainGameLoop(x, *args, interval=1, display=False, PhaseSpace=None, ps_figure=None, wait=0, free=False, **kwargs):
    """
    Start instantiating the World and the load...
    """
    pause = False
    my_maze = Maze(*args, size=x.size, shape=x.shape, solver=x.solver, free=free)
    my_load = Load(my_maze)
    screen = None
    if display:
        screen = Display_screen(my_maze, caption=x.filename)

    """
    --- main game loop ---
    """
    running = True  # this tells us, if the simulation is still running
    i = 0  # This tells us which movie the experiment is part of, 'i' is loop counter

    # Loop that runs the simulation... 
    while running:
        arrows = x.step(my_load, i, my_maze=my_maze, pause=pause, **kwargs)

        """ Display the frame """
        if display:
            running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=pause, interval=interval,
                                                    arrows=arrows, PhaseSpace=PhaseSpace, ps_figure=ps_figure,
                                                    wait=wait, **kwargs)

        if not pause:
            i += interval  # we start a new iteration

        if i >= len(x.frames) - 1 - interval and not pause:
            running = False  # break the loop, if we are at the end of the experimental data.
            if display:
                if len(x.frames) < 4:  # just to check the error.
                    running, i, pause = Pygame_EventManager(x, i, my_load, my_maze, screen, pause=True)
                running = Display_end()
    return x
