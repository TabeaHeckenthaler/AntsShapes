import pandas as pd
from Directories import df_dir
from Analysis.GeneralFunctions import graph_dir
from os import path
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_json(df_dir)


def reduce_legend(ax):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def save_fig(fig, name, svg=False):
    name = "".join(x for x in name if x.isalnum())
    if fig.__module__ == 'plotly.graph_objs._figure':
        fig.write_image(graph_dir() + path.sep + name + '.pdf')
        if svg:
            fig.write_image(graph_dir() + path.sep + name + '.svg')
    else:
        fig.savefig(graph_dir() + path.sep + name + '.pdf', format='pdf', pad_inches=0.5, bbox_inches='tight')
        if svg:
            fig.savefig(graph_dir() + path.sep + name + '.svg', format='svg', pad_inches=0.5, bbox_inches='tight')


def Carrier_Number_Binning(df, solver, number_of_bins=5):
    bin_name = 'bin_name'
    if solver == 'human' or solver == 'ant':
        sorter_dict = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
        df[bin_name] = df['maze size'].map(sorter_dict)

        df.loc[(df['average Carrier Number'] < 3) & (df['maze size'] == 'M'), bin_name] = 0.5

        return df.sort_values(by=bin_name).reset_index(drop=True).copy()
    else:
        bin_content = int(np.ceil(len(df) / number_of_bins))
        sorted_df = df.sort_values(by='average Carrier Number').reset_index(drop=True).copy()
        sorted_df[bin_name] = [ii for ii in range(number_of_bins) for _ in range(bin_content)][:len(df)]

        # check bin boundaries
        def set_boundary_group_indices(sorted_df, i):
            aCN = int(sorted_df[['average Carrier Number']].iloc[i])
            indices = sorted_df.groupby(by='average Carrier Number').get_group(aCN).index
            in_group = sorted_df[bin_name].iloc[indices[0]]

            sorted_df.loc[indices, bin_name] = in_group
            return sorted_df

        for i in range(bin_content, len(df), bin_content):
            sorted_df = set_boundary_group_indices(sorted_df, i)
        return sorted_df
