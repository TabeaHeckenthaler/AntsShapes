from progressbar import progressbar
from Analysis.GeneralFunctions import graph_dir
from Analysis.Velocity import velocity_x
from os import path
import plotly.express as px
import datetime
from DataFrame.dataFrame import myDataFrame
from Setup.Maze import Maze
import numpy as np
from trajectory_inheritance.trajectory import get
import pandas as pd

DELTA_T = 2

# TODO: I still want to restructure  the modules related to contact in the directories PhysicsEngine and Analysis.
# TODO: Test this module. There are probably still many mistakes.

def theta(r):
    [x, y] = r
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0 and y != 0:
        return np.sign(y) * np.pi / 2


class Contact:

    def __init__(self, filename, impact_frame, contact_points, x):
        self.x = x
        self.filename = filename
        self.impact_frame = impact_frame
        self.start_frame = int(max(0, self.impact_frame - x.fps * DELTA_T))
        self.end_frame = int(min(len(x.frames) - 1, self.impact_frame + x.fps * DELTA_T))
        self.contact_points = self.reduce_contact_points(contact_points)
        self.my_maze = Maze(x)

    @staticmethod
    def reduce_contact_points(contact_points: list) -> list:
        """
        Reduce the contact points when a single contact is has many consecutive contact points
        :return: the shortened list of contact points
        """
        a = np.array(contact_points)
        a = a[a[:, 1].argsort()]
        contact_points = [[a[0]]]
        for a0, a1 in zip(a[:-1], a[1:]):
            if a0[1] - a1[1] > 1:
                contact_points.append([a1])
            else:
                contact_points[-1].append(a1)
        contact_points = [np.array(c).mean(axis=0) for c in np.array(contact_points)]
        return contact_points

    def contacts_bottom_slit(self) -> bool:
        """
        :return: boolean, whether contact is with bottom slit
        """

        return np.mean(np.array(self.contact_points)[:, 1]) < \
               self.my_maze.arena_height / 2 - self.my_maze.exit_size / 2 + 0.1

    def torque(self) -> float:
        """
        :return: torque on the object according to the velocity and the points of contact with the wall
        """
        rhos = rho_cross = torque_i = []
        for contact_point in self.contact_points:
            rhos.append(contact_point - self.x.position[self.impact_frame])
            # rho_cross.append(np.cross(np.hstack([rhos[-1], 1]), [0, 0, 1]))
            v0 = np.mean(velocity_x(self.x, 1, 'x', 'y')[:, self.start_frame:self.impact_frame], axis=1)
            torque_i.append(np.cross(rhos[-1], v0))

        torque = np.sum(torque_i)
        if self.contacts_bottom_slit():
            torque[-1], = -torque[-1],
        return torque

    def theta_dot(self) -> float:
        # Characterize rotation
        r_impact = self.x.position[self.impact_frame] - self.contact_points[0]
        r_end = self.x.position[self.end_frame] - self.contact_points[0]

        theta_dot = (theta(r_end) - theta(r_impact))/DELTA_T

        # I want to flip the ones contacting the bottom corner...
        if self.contacts_bottom_slit():
            theta_dot = -theta_dot
        return theta_dot


class Contact_analyzer(pd.DataFrame):

    def __init__(self, filenames):
        super().__init__(filenames, columns=['filename'])
        self['delta_frames'] = df['fps'] * DELTA_T  # minimum time that a contact has to be apart from each other
        self.contacts = self.find_contacts()

    def find_contacts(self) -> list:
        """
        Search in every trajectory for frames in which the shape has a contact with the wall, add to new dataframe
        start and end frame and position of contact (after reducing to the central point)
        :return: pd.DataFrame with the following columns = ['start frame', 'end frame', 'contact point']
        """

        for filename in self['filename']:
            x = get(filename[0])
            contacts = x.find_contact()
            wall_contacts = np.where([len(contact) > 0 and contact[0][0] > Maze(x).slits[0] - 1
                                      #  and (abs(con[0][1] - my_maze.arena_height / 2 - my_maze.exit_size / 2) < 2
                                      #       or abs(con[0][1] - my_maze.arena_height / 2 + my_maze.exit_size / 2) < 2)
                                      for contact in contacts])[0]

            # only if its not a to short contact!
            # wall_contacts = [c for i, c in enumerate(contact_frames) if abs(c - contact_frames[i - 1]) < 2
            impact_frames = list(wall_contacts[0:1]) + [c for i, c in enumerate(wall_contacts)
                                                        if c - wall_contacts[i - 1] > int(x.fps * 2)]

            for impact_frame in impact_frames:
                contacts.append(Contact(filename, impact_frame, contacts[impact_frame], x))
            return contacts

    @staticmethod
    def plot(torques, theta_dots, information=''):
        titles = ['torque_vs_omega_' + size]
        titles_new = ['parallel_force_vs_rho_dot' + size]
        fig = px.scatter(x=torques, y=theta_dots, text=information)
        fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
        name = graph_dir() + path.sep + titles[0] + datetime.datetime.now().strftime("%H_%M")
        fig.write_html(name + '.html')

        fig = px.scatter(x=torques, y=theta_dots)
        fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
        fig.write_image(name + '.svg')
        fig.write_image(name + '.pdf')
        fig.show()


if __name__ == '__main__':
    solver = 'ant'
    shape = 'I'
    df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape)

    for size in df.groupby('size').groups.keys():
        contact_analyzer = Contact_analyzer(df.loc[df.groupby('size').groups[size]]['filename'])
        theta_dots = [contact.theta_dot() for contact in contact_analyzer.contacts]
        torques = [contact.torque() for contact in contact_analyzer.contacts]
        contact_analyzer.plot(torques, theta_dots)
