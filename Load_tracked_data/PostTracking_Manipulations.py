import trajectory
import numpy as np
from Load_tracked_data.Load_Experiment import MainGameLoop
from scipy.ndimage import gaussian_filter


def SmoothConnector(file1, file2):
    # We use this function, when we want to smoothly connect two datasets, where there was an pause between the tracking
    # Find the velocity at which the shape moved before the pause
    from Analysis_Functions.Velocity import velocity
    from Setup.Load import periodicity
    velocity1 = velocity(file1.position, file1.angle, file1.fps, file1.size, file1.shape, 1, 'x', 'y')
    velocity2 = velocity(file2.position, file2.angle, file2.fps, file2.size, file2.shape, 1, 'x', 'y')

    con_vel = max(0.01, np.mean(np.hstack([velocity1[:, -200:-1],velocity2[:, 1:200]])))
    # connector velocity in distance (cm) per frame

    # Instantiate a new load which has the right starting position
    connector_load = trajectory.Trajectory(size=file1.size, shape=file1.shape, solver=file1.solver,
                                          filename=file1.VideoChain[-1] + '_CONNECTOR_' + file2.VideoChain[0],
                                          free=file1.free, fps=file1.fps,
                                          winner=False,
                                          )
    # Find the end position
    dx = file1.position[-1][0] - file2.position[1][0]
    dy = file1.position[-1][1] - file2.position[1][1]

    per = 2 * np.pi / periodicity[file1.shape]
    con_frames = int(np.sqrt(dx**2 + dy**2) / con_vel)

    connector_load.position, connector_load.angle, connector_load.frames = np.ndarray([con_frames, 2]), np.ndarray(
        [con_frames]), np.ndarray([con_frames])
    connector_load.position[:, 0] = np.linspace(file1.position[-1][0], file2.position[1][0], num=con_frames)
    connector_load.position[:, 1] = np.linspace(file1.position[-1][1], file2.position[1][1], num=con_frames)

    # find the right angle to connect to
    final_angle = np.floor(file1.angle[-1] / per) * per + np.mod(file2.angle[1], per)
    connector_load.angle = np.linspace(file1.angle[-1], final_angle, num=con_frames)

    # all the other stuff
    connector_load.frames = np.int0(np.linspace(1, con_frames, num=con_frames))
    # connector_load.contact = [[] for i in range(len(connector_load.frames))]
    from Load_tracked_data.Load_Experiment import step
    connector_load = MainGameLoop(connector_load, step=step)
    return connector_load


def PostTracking_Manipulations_shell(filename):
    def FalseTracking_Smooth():
        x = trajectory.Get(filename)
        if not (hasattr(x, 'falseTracking')):
            x.falseTracking = []

        trajectory.Save(x)

        x.position = np.transpose(
            np.vstack([gaussian_filter(x.position[:, 0], sigma=51), gaussian_filter(x.position[:, 1], sigma=51)]))

        # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
        x.angle = gaussian_filter(x.angle, sigma=51)
        # all the other stuff
        trajectory.Save(x)

    def CutFalseTracking_Free():
        x = trajectory.Get(filename)
        breakpoint()
        if not (hasattr(x, 'falseTracking')):
            x.falseTracking = []
        print(str(x.falseTracking))
        if bool(int(input('Cut off the end '))):
            frame2 = int(input('EndFrame '))
            frame1 = x.frames[0]
        if bool(int(input('Cut off the start '))):
            frame1 = int(input('StartFrame '))
            frame2 = x.frames[-1]
        trajectory.Save(x)

        if not (hasattr(x, 'free')):
            print('Why no free attribute?')
        index1 = np.where(x.frames == frame1)[0][0]
        index2 = np.where(x.frames == frame2)[0][0]

        x.position = x.position[index1:index2, :]

        # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
        x.angle = x.angle[index1:index2]
        x.frames = x.frames[index1:index2]
        x.contact = x.contact[index1:index2]
        breakpoint()
        # all the other stuff
        trajectory.Save(x)
        print(x)

    print('0 = no corrections')
    print('1 = Cut False tracking (if there are issues in the beginning or the end in the free motion)')
    print('3 = Smooth out the trajectory')
    print('Connecting to another movie requieres x + y')
    manipulation = int(input('Want to correct something??  '))
    while manipulation != 0:
        if manipulation == 1:
            CutFalseTracking_Free()

        if manipulation == 3:
            FalseTracking_Smooth()

        print('\n 0 = no corrections')
        print('1 = Cut False tracking (if there are issues in the beginning or the end in the free motion)')
        print('3 = Smooth out the trajectory')
        print('Connecting to another movie requires x + y')
        manipulation = int(input('Want to correct something else??'))
