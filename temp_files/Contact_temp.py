from PhysicsEngine.Contact import Contact_analyzer_2
from Classes_Experiment.bundle import Bundle
from progressbar import progressbar
from Analysis_Functions.GeneralFunctions import graph_dir
from os import path
import plotly.express as px
import datetime

solver = 'ant'
shape = 'I'
# fig = px.figure()

for size in ['XL']:
    # my_bundle = Bundle(size='M') - Bundle(size='M', shape='SPT')
    my_bundle = Bundle(solver, size=size, shape=shape) - Bundle(solver, shape='SPT')
    titles = ['torque_vs_omega_' + size]
    titles_new = ['parallel_force_vs_rho_dot' + size]

    li = [Contact_analyzer_2(x) for x in progressbar(list(my_bundle)[10:])]

    theta_dot, torque, information = [], [], []
    for i in li:
        torque = torque + i[1]
        theta_dot = theta_dot + i[0]
        information = information + i[2]

    fig = px.scatter(x=torque, y=theta_dot, text=information)
    fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
    name = graph_dir() + path.sep + titles[0] + datetime.datetime.now().strftime("%H_%M")
    fig.write_html(name + '.html')

    fig = px.scatter(x=torque, y=theta_dot)
    fig.update_layout(xaxis_title="torque [N] ", yaxis_title="theta_dot [rad/s]", )
    fig.write_image(name + '.svg')
    fig.write_image(name + '.pdf')
    fig.show()

