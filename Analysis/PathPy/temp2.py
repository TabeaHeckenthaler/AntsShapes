import pathpy as pp
from pathpy.visualisation.html import export_html

n = pp.Network(directed=True)
n.add_edge('a', 'c')
n.add_edge('b', 'c')
n.add_edge('c', 'd')
n.add_edge('c', 'e')
print(n)

export_html(n, r'C:\Users\tabea\PycharmProjects\AntsShapes\a.html', )
