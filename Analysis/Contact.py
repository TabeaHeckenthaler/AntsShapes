from tqdm import tqdm
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
import os
from Directories import data_home

DELTA_T = 2


# TODO: Test this module. There are probably still many mistakes.

def theta(r):
    [x, y] = r
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0 and y != 0:
        return np.sign(y) * np.pi / 2


class Contact(pd.Series):

    def __init__(self, filename, impact_frame, contact_points):
        x = get(filename)
        my_maze = Maze(x)
        super().__init__(pd.Series({'filename': filename,
                                    'impact_frame': impact_frame,
                                    'contact_points': contact_points,
                                    'start_frame': int(max(0, impact_frame - x.fps * DELTA_T)),
                                    'end_frame': int(min(len(x.frames) - 1, impact_frame + x.fps * DELTA_T)),
                                    'arena_height': my_maze.arena_height,
                                    'exit_size': my_maze.exit_size
                                    }))

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

        return np.mean(np.array(self.contact_points)[:, 1]) < self.arena_height / 2 - self.exit_size / 2 + 0.1

    def torque(self) -> float:
        """
        :return: torque on the object according to the velocity and the points of contact with the wall
        """
        x = get(self.filename)
        rhos = rho_cross = torque_i = []
        for contact_point in self.contact_points:
            rhos.append(contact_point - x.position[self.impact_frame])
            # rho_cross.append(np.cross(np.hstack([rhos[-1], 1]), [0, 0, 1]))
            v0 = np.mean(velocity_x(x, 1, 'x', 'y')[:, self.start_frame:self.impact_frame], axis=1)
            torque_i.append(np.cross(rhos[-1], v0))

        torque = np.sum(torque_i)
        if self.contacts_bottom_slit():
            torque[-1], = -torque[-1],
        return torque

    def theta_dot(self) -> float:
        """
        :return: first derivative of angle theta
        """
        x = get(self.filename)
        # Characterize rotation
        r_impact = x.position[self.impact_frame] - self.contact_points[0]
        r_end = x.position[self.end_frame] - self.contact_points[0]

        theta_dot = (theta(r_end) - theta(r_impact)) / DELTA_T

        # I want to flip the ones contacting the bottom corner...
        if self.contacts_bottom_slit():
            theta_dot = -theta_dot
        return theta_dot

    def __str__(self):
        return self.filename + '_' + str(self.impact_frame)


class Contact_analyzer(pd.DataFrame):

    def __init__(self, df):
        super().__init__(df)
        self['delta_frames'] = self['fps'] * DELTA_T  # minimum time that a contact has to be apart from each other
        self.contacts = pd.DataFrame([])

    def address(self):
        return data_home + 'Contacts' + path.sep + 'ant' + path.sep + \
               self['size'].iloc[0] + '_' + self['shape'].iloc[0] + '_contact_list.json'

    def find_contacts(self) -> pd.DataFrame:
        """
        Search in every trajectory for frames in which the shape has a contact with the wall, add to new dataframe
        start and end frame and position of contact (after reducing to the central point)
        :return: pd.DataFrame
        """
        print('Finding contacts for ' + self.address())
        for filename in tqdm(self['filename']):
            x = get(filename)
            traj_contacts = x.find_contact()
            wall_contacts = np.where([len(contact) > 0 and contact[0][0] > Maze(x).slits[0] - 1
                                      #  and (abs(con[0][1] - maze.arena_height / 2 - maze.exit_size / 2) < 2
                                      #       or abs(con[0][1] - maze.arena_height / 2 + maze.exit_size / 2) < 2)
                                      for contact in traj_contacts])[0]

            # only if its not a to short contact!
            # wall_contacts = [c for i, c in enumerate(contact_frames) if abs(c - contact_frames[i - 1]) < 2
            impact_frames = list(wall_contacts[0:1]) + [c for i, c in enumerate(wall_contacts)
                                                        if c - wall_contacts[i - 1] > int(x.fps * 2)]

            for impact_frame in impact_frames:
                con = Contact(filename, impact_frame, traj_contacts[impact_frame])
                self.contacts = pd.concat([self.contacts, con], axis=1)
        return self.contacts.transpose().reset_index(drop=True)

    # @staticmethod
    # def plot(torques, theta_dots, information=''):
    #     titles = ['torque_vs_omega_' + size]
    #     titles_new = ['parallel_force_vs_rho_dot' + size]
    #     fig = px.scatter(x=torques, y=theta_dots, text=information)
    #     fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
    #     name = graph_dir() + path.sep + titles[0] + datetime.datetime.now().strftime("%H_%M")
    #     fig.write_html(name + '.html')
    #
    #     fig = px.scatter(x=torques, y=theta_dots)
    #     fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
    #     fig.write_image(name + '.svg')
    #     fig.write_image(name + '.pdf')
    #     fig.show()

    def save(self) -> None:
        """
        save a pickle of the list of Contact objects
        """
        self.contacts.to_json(self.address())

    def load_contacts(self, save=True) -> None:
        """
        load calculated contacts to self.contacts
        """
        if os.path.exists(self.address()):
            self.contacts = pd.read_json(self.address())
        else:
            self.contacts = self.find_contacts()
            if save:
                contact_analyzer.save()


if __name__ == '__main__':
    solver = 'ant'
    shapes = ['I', 'T', 'H']

    for shape in shapes:
        df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape)
        for size in df.groupby('size').groups.keys():
            contact_analyzer = Contact_analyzer(df.loc[df.groupby('size').groups[size]][['filename', 'size', 'solver', 'shape', 'fps']])
            contact_analyzer.load_contacts()
            # theta_dots = [contact.theta_dot() for contact in contact_analyzer.contacts]
            # torques = [contact.torque() for contact in contact_analyzer.contacts]
            # contact_analyzer.plot(torques, theta_dots)
