from DataFrame.dataFrame import myDataFrame
from Analysis.Contact.Contact import Contact, Contact_analyzer
import pandas as pd
from trajectory_inheritance.trajectory import get
from Directories import contacts_dir
import numpy as np
import plotly.express as px
from DataFrame.plot_dataframe import save_fig
from tqdm import tqdm


if __name__ == '__main__':
    shapes = ['I', 'T', 'H']
    sizes = ['M', 'L', 'SL', 'XL']

    shape = 'H'
    solver = 'ant'
    redecision_times = {}

    for size in tqdm(sizes):
        df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape).groupby('size').get_group(size)
        contact_analyzer = Contact_analyzer(df[['filename', 'size', 'solver', 'shape', 'fps']])
        contact_analyzer.load_contacts()

        if 'fps' not in contact_analyzer.contacts.keys():
            contact_analyzer.add_column()

        contacts = [Contact(ds=contact[1]) for contact in contact_analyzer.contacts.iterrows()]
        # contacts[1] because iterrows returns a tuple

        contacts_df = pd.concat(contacts, axis=1).transpose()

        redecision_times[size] = [contact.redecided() for contact in contacts]
        contacts_df['redecision_time'] = redecision_times[size]

        # # display contacts
        # [c.play(end_frame=(c.impact_frame + c.fps * time + c['fps'] * 2))
        #  for c, time in zip(contacts, contacts_df['redecision_time']) if time > 0]

    k = 1
