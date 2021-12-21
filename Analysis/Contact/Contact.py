from tqdm import tqdm
from DataFrame.dataFrame import myDataFrame
from Setup.Maze import Maze
import numpy as np
from trajectory_inheritance.trajectory import get
import pandas as pd
import os
from Directories import contacts_dir
import plotly.express as px
import datetime
from DataFrame.plot_dataframe import save_fig
from matplotlib import pyplot as plt

DELTA_T = 2


def theta(r: list) -> float:
    """
    :param r: Position vector
    :return: angle to the x axis
    """
    [x, y] = r
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0 and y != 0:
        return np.sign(y) * np.pi / 2


class Contact(pd.Series):
    def __init__(self, filename=str(), impact_frame=int(), contact_points=None, ds: pd.Series = pd.Series([])):
        """
        ds: DataSeries
        """
        if len(ds) > 0:
            super().__init__(ds)
        else:
            x = get(filename)
            my_maze = Maze(x)
            super().__init__(pd.Series({'filename': filename,
                                        'impact_frame': impact_frame,
                                        'contact_points': contact_points,
                                        'start_frame': int(max(0, impact_frame - x.fps * DELTA_T)),
                                        'end_frame': int(min(len(x.frames) - 1, impact_frame + x.fps * DELTA_T)),
                                        'arena_height': my_maze.arena_height,
                                        'exit_size': my_maze.exit_size,
                                        'fps': x.fps
                                        }))
        k = 1

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

    def pre_velocity(self):
        x = get(self.filename)
        return np.mean(x.velocity(1, 'x', 'y', 'angle')[:, self.start_frame:self.impact_frame], axis=1)

    def post_velocity(self):
        x = get(self.filename)
        return np.mean(x.velocity(1, 'x', 'y', 'angle')[:, self.impact_frame:self.end_frame], axis=1)

    def retraction_distance(self) -> float:
        return 0.

    def torque(self) -> float:
        """
        :return: torque on the object according to the velocity and the points of contact with the wall
        """
        x = get(self.filename)
        torque = 0

        for contact_point in self.contact_points:
            rho = x.position[self.impact_frame] - contact_point
            v0 = self.pre_velocity()
            torque += np.cross(np.hstack([rho, 0]), np.hstack([v0, 0]))[2]

        # I want to flip the ones contacting the top corner...
        if not self.contacts_bottom_slit():
            return torque
        return -torque

    def theta_dot(self) -> float:
        """
        :return: first derivative of angle theta
        """
        x = get(self.filename)
        r_impact = x.position[self.impact_frame] - self.contact_points[0]
        r_end = x.position[self.end_frame] - self.contact_points[0]

        theta_dot = (theta(r_end) - theta(r_impact)) / DELTA_T

        # I want to flip the ones contacting the top corner...
        if not self.contacts_bottom_slit():
            return theta_dot
        return -theta_dot

    def __str__(self):
        return self.filename + '_' + str(self.impact_frame)

    def play(self, end_frame: int = None):
        if end_frame is None:
            end_frame = self.end_frame
        x = get(self.filename)
        x.play(indices=[self.start_frame, min(end_frame + 200, len(x.frames))])

    def redecided(self, walking_speed=0.05) -> float:
        """
        Find the time after impact,  at which the shape resumes walking_speed.
        :param walking_speed: At what speed, can we say, that the object is moving?
        :return: Time passed before redecision
        """
        x = get(self.filename)
        speed = np.linalg.norm(x.velocity(0.5, 'x', 'y', 'angle'), axis=0)
        test_sec_distance_start = 6
        test_frame_distance = min(int(x.fps*test_sec_distance_start), len(x.frames) - self.impact_frame-5)

        while len(x.frames) > self.impact_frame+test_frame_distance and \
                speed[self.impact_frame+test_frame_distance] < walking_speed:
            test_frame_distance = test_frame_distance + x.fps

        speed = speed[self.impact_frame:self.impact_frame+test_frame_distance]
        frames = len(np.where(walking_speed > speed)[0])

        if frames/x.fps > 10:
            k = 1

        return frames/x.fps


class Contact_analyzer(pd.DataFrame):

    def __init__(self, df):
        super().__init__(df)
        self['delta_frames'] = self['fps'] * DELTA_T  # minimum time that a contact has to be apart from each other
        self.contacts = pd.DataFrame([])

    def address(self):
        return os.path.join(contacts_dir, self['size'].iloc[0] + '_' + self['shape'].iloc[0] + '_contact_list.json')

    def add_column(self):
        fps = [get(filename).fps for filename in self.contacts['filename']]
        self.contacts['fps'] = fps
        self.save()
        return

    def find_contacts(self) -> pd.DataFrame:
        """
        Search in every trajectory for frames in which the shape has a contact with the wall, add to new dataframe
        start and end frame and position of contact (after reducing to the central point)
        :return: pd.DataFrame
        """
        print('Finding contacts for ' + self.address())
        for filename in tqdm(self['filename']):
            x = get(filename)
            contact_points = x.find_contact()
            wall_contacts = np.where([len(contact) > 0 and contact[0][0] > Maze(x).slits[0] - 1
                                      #  and (abs(con[0][1] - maze.arena_height / 2 - maze.exit_size / 2) < 2
                                      #       or abs(con[0][1] - maze.arena_height / 2 + maze.exit_size / 2) < 2)
                                      for contact in contact_points])[0]

            # only if its not a to short contact!
            # wall_contacts = [c for i, c in enumerate(contact_frames) if abs(c - contact_frames[i - 1]) < 2
            impact_frames = list(wall_contacts[0:1]) + [c for i, c in enumerate(wall_contacts)
                                                        if c - wall_contacts[i - 1] > int(x.fps * 2)]

            for impact_frame in impact_frames:
                con = Contact(filename=filename, impact_frame=impact_frame, contact_points=contact_points[impact_frame])
                self.contacts = pd.concat([self.contacts, con], axis=1)
        return self.contacts.transpose().reset_index(drop=True)

    @staticmethod
    def plot(torques, theta_dots, size, information=''):
        # this is to out single points :)
        fig = px.scatter(x=torques, y=theta_dots, text=information)
        fig.update_layout(xaxis_range=[-10, 10], yaxis_range=[-1, 1])
        fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
        fig.write_html('torque_vs_omega_' + size + '.html')
        fig.show()
        fig = px.scatter(x=torques, y=theta_dots)
        fig.update_layout(xaxis_range=[-10, 10], yaxis_range=[-5, 5])
        fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
        save_fig(fig, 'torque_vs_omega_' + size + '_' + shape + '_' + datetime.datetime.now().strftime("%H_%M"))
        fig.show()

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
                self.save()


if __name__ == '__main__':
    solver = 'ant'
    shapes = ['I', 'T', 'H']
    #
    # for shape in shapes:
    #     df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape)
    #
    #     for size in ['XL']:
    #     # for size in df.groupby('size').groups.keys():
    #         contact_analyzer = Contact_analyzer(df.loc[df.groupby('size').groups[size]][['filename', 'size', 'solver', 'shape', 'fps']])
    #         contact_analyzer.load_contacts()
    #         contacts = [Contact(ds=contact[1]) for contact in contact_analyzer.contacts.iterrows()]
    #         contacts[4].theta_dot()
    #
    #         torques = [contact.torque() for contact in contacts]
    #         theta_dots = [contact.theta_dot() for contact in contacts]
    #         k = [i for i, (torque, theta_dot) in enumerate((zip(torques, theta_dots))) if torque > 0 and theta_dot > 0]
    #         contact_analyzer.plot(torques, theta_dots, information=contact_analyzer.contacts['filename'].to_list())
    #         k = 1

    shape = 'I'
    df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape)
    contact_analyzer = Contact_analyzer(df[['filename', 'size', 'solver', 'shape', 'fps']])
    contact_analyzer.load_contacts()
    contacts = [Contact(ds=contact[1]) for contact in contact_analyzer.contacts.iterrows()]  # contacts[1] because iterrows returns a tuple
    contact1 = contacts[1]

    contacts_df = pd.concat(contacts, axis=1).transpose()
    contacts_df['torque'] = [contact.torque() for contact in contacts]
    contacts_df['theta_dot'] = [contact.theta_dot() for contact in contacts]

    k = 1
