from DataFrame.dataFrame import myDataFrame
from Analysis.Contact.Contact import Contact, Contact_analyzer
import numpy as np
import pandas as pd
from Directories import contacts_dir
import plotly.express as px
from DataFrame.plot_dataframe import save_fig

if __name__ == '__main__':
    solver = 'ant'
    shapes = ['I', 'T', 'H']

    shape = 'I'
    df = myDataFrame.groupby('solver').get_group(solver).groupby('shape').get_group(shape)
    contact_analyzer = Contact_analyzer(df[['filename', 'size', 'solver', 'shape', 'fps']])
    contact_analyzer.load_contacts()
    contacts = [Contact(ds=contact[1]) for contact in contact_analyzer.contacts.iterrows()]
    # contacts[1] because iterrows returns a tuple

    contact1 = contacts[1]

    contacts_df = pd.concat(contacts, axis=1).transpose()
    contacts_df['vel_correlation'] = [np.dot(contact.pre_velocity(), contact.post_velocity()) for contact in contacts]

    k = 1
