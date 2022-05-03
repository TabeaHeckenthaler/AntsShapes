import subprocess

import numpy as np
import pyvista as pv

tetgen_dir = r'C:\Users\Rotem Shalev\Desktop\semester A3 baby!\לומדים ניסויים עם אנשים נוספים\אצל עופר\tetgen\built_code\Debug'
file_dir   = r'C:\Users\Rotem Shalev\Desktop\semester A3 baby!\לומדים ניסויים עם אנשים נוספים\אצל עופר\tetgen\node files'


def is_feasible(pos, former_found=(0, 0)):
    # The boolean function we're interested in describing

    # first check the bounding box
    x, y, z = pos
    theta = z/z_scale
    maze_bb.set_configuration([x, y], float(theta))
    res = possible_configuration(load_bb, maze_corners, former_found)
    if res[0]:
        return res
    else:
        maze.set_configuration([x, y], float(theta))
        return possible_configuration(load, maze_corners, former_found)
        # flipped?? "True, if there is collision. False, if it is an allowed configuration"
        # ..Seems fine.

# get_signs = np.vectorize(is_feasible, excluded=['former_found'], signature='(3)->()()')
# doesn't work when returning two arrays

def calculate_volume(tet, points):
    # calculate volume of a tetrahedron
    a = points[tet[0]]
    b = points[tet[1]] - a
    c = points[tet[2]] - b
    d = points[tet[3]] - c

    # return abs(np.linalg.det(np.stack((b, c, d))))
    return abs(np.dot(b, np.cross(c, d)))


def write_node_file(fname, points):
    numpoints, dim = points.shape
    assert dim == 3

    header = f"{numpoints} 3 0 0"  # save attributes and such yourself

    linelis = [header]

    for i, point in enumerate(points):
        linelis.append(" ".join(str(x) for x in
                                (i, float(point[0]), float(point[1]), float(point[2]))
                                ))

    with open(fname, 'w') as file:
        file.write("\n".join(linelis))


def load_tetgen_file(fname, max_index=None, is_float=True):
    # loads a tetgen file into an array, or returns None if that failed.
    # ignores rows starting in "#", and the first row.
    try:
        with open(fname, 'r') as file:
            data = file.read()

        f = float if is_float else int
        arr = tuple(tuple(f(y) for y in x.split()[1:max_index])  # splits on multiple spaces ("whitespace")
                    for x in data.split("\n")[1:] if len(x) > 0 and x[0] != "#")
        # maybe it'll be nicer to use a generator version of str.split instead of building two O(n) arrays:
        # https://stackoverflow.com/questions/3862010/is-there-a-generator-version-of-string-split-in-python
        # or we could build the array itself in place, but that won't necessarily be faster

        return np.array(arr, float if is_float else int)
    except FileNotFoundError:
        print(f"Failed loading {fname}")
        return None


def make_init_tet(fname):
    subprocess.run(f'tetgen -zQBF "{fname}"', shell=True, check=True, cwd=tetgen_dir, stdout=subprocess.PIPE)


def refine_tet(fname, max_volume, quality, points_to_add=None):
    s = f'tetgen -QBFrq{quality}'  # silent, without boundary info, without face/edge files, refined, of specific ~quality~
    if points_to_add is not None:
        points_to_add = np.array(points_to_add, float)
        write_node_file(f'{fname}.a.node', points_to_add)  # file to add from needs a specific name
        s += 'i'
    else:
        s += f'a{float(max_volume)}'  # add max volume constraint only once
    subprocess.run(s + f' "{fname}"\n', shell=True, check=True, cwd=tetgen_dir, stdout=subprocess.PIPE)


def get_tetrahedralization(func, init_points, max_volume, min_volume, quality=1.3, max_iterations=15, fname='temp'):
    # main function - takes a boolean function, a set of initial points, and volume constraints, and
    # tetrahedralizes the domain of the function
    # fname is the name of the files to be written while refining

    fname = fr"{file_dir}\{fname}"
    init_tet_fname = fname + ".1"
    write_node_file(init_tet_fname + ".node", init_points)

    def tot_func(points, former_found_arr=None):
        if former_found_arr is None:
            former_found_arr = np.zeros((len(points), 2), int)
        res = tuple(func(p, tuple(f)) for p, f in zip(points, former_found_arr))
        return (tuple(r[0] for r in res),
                tuple(r[1] for r in res))

    signs, former_found_arr = tot_func(init_points)

    make_init_tet(init_tet_fname)

    new_points = None

    if max_iterations <= 2: raise Exception(f"{max_iterations=} should be 3 or more")

    for i in range(2, max_iterations):  # i is the index of current tetrahedralization
        refine_tet(fr"{fname}.{i}", max_volume, quality, new_points)
        i += 1

        points = load_tetgen_file(fr"{fname}.{i}.node", 4)
        tets = load_tetgen_file(fr"{fname}.{i}.ele", 5, False)


        if (len_s := len(signs)) != points.shape[0]:
            extra_points = points[len_s:]  # points added in current refinement
            neighs = {}
            for tet in tets:
                for edge in combinations(tet, 2):
                    a, b = edge
                    if a >= len_s and b < len_s:
                        neighs[a] = neighs.get(a, tuple()) + (b,)
                    if b >= len_s and a < len_s:
                        neighs[b] = neighs.get(b, tuple()) + (a,)
            extra_former_found = []
            for p in range(len_s, len(points)):
                extra_former_found.append(max(former_found_arr[n] for n in neighs.get(p, (0,))))

            new_signs, new_former_found_arr = tot_func(extra_points, extra_former_found)
            signs = signs + new_signs
            former_found_arr = former_found_arr + new_former_found_arr

        new_points = []

        for tet in tets:
            if (1 <= sum(signs[s] for s in tet) <= 3) and\
                    calculate_volume(tet, points) >= min_volume:
                new_points.append(sum(points[s] for s in tet)/4)

        if not new_points: break
        print(str(i).ljust(4), str(len(new_points)).ljust(6))
    return points, tets, signs, i, former_found_arr


class EdgeDict:
    def __init__(self, points, former_found_arr):
        self._dict = {}
        self._new_points = []
        self._new_former_found = []
        #self._num_old_points = len(points)
        self._points = points
        self._former_found_arr = former_found_arr

    def __call__(self, start, end):
        key = tuple(sorted((start, end)))
        res = self._dict.get(key, None)
        if res is not None: return res

        res = len(self._new_points)  # + self._num_old_points
        new_p = (self._points[start] + self._points[end])/2
        new_f = max(self._former_found_arr[start], self._former_found_arr[end])   # doesn't take (0, 0) unless needed
        self._new_points.append(new_p)
        self._new_former_found.append(new_f)
        self._dict[key] = res
        return res

    def get_new_points(self):
        return self._new_points

    def get_new_former_found(self):
        return self._new_former_found


def plot_tetrahedrons(points, ele, signs, former_found_arr):
    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.enable_anti_aliasing()

    faces = []
    edge_dict = EdgeDict(points, former_found_arr)
    for tet in ele:
        ele_signs = tuple(signs[point] for point in tet)
        s = sum(ele_signs)
        if   s == 3:
            i = np.argmin(ele_signs)
            res = np.roll(tet, 3 - i)
            faces.extend((3,) + tuple(edge_dict(x, res[3]) for x in res[:3]))
        elif s == 1:
            i = np.argmax(ele_signs)
            res = np.roll(tet, 3 - i)
            faces.extend((3,) + tuple(edge_dict(x, res[3]) for x in res[:3]))
        elif s == 2:
            # do a zigzag type thing - moving between on and off
            old_i = 0
            new_i = 1
            if ele_signs[old_i] == ele_signs[new_i]: new_i += 1  # could be while, but s == 2..
            verts = []
            for i in range(4):  # we seek 4 vertices
                verts.append(edge_dict(tet[old_i], tet[new_i]))
                for temp in range(4):
                    if temp != old_i and ele_signs[temp]^ele_signs[new_i]: break
                old_i = new_i; new_i = temp
            faces.extend((4,) + tuple(verts))

    # tot_points = np.concatenate((points, edge_dict.get_new_points()), 0)
    tot_points = edge_dict.get_new_points()
    mesh = pv.PolyData(tot_points, faces)

    # plotter.add_mesh(mesh, 'gray', lighting=True)#, show_edges=True)
    c = np.array(edge_dict.get_new_former_found(), float)[..., 1]
    plotter.add_mesh(mesh, scalars=c, cmap='gist_rainbow', interpolate_before_map=False,
                     lighting=True, show_scalar_bar=False)

    plotter.show_bounds(mesh, grid='back', location='front',
                        xlabel='x', ylabel='y', zlabel='z',
                        all_edges=True
                        )

    plotter.show()


from PhysicsEngine.Contact import possible_configuration
from Setup.Maze import Maze  # BTW, I use box2d-py instead of box2d + box2d-kengz

# set up objects for phase space calculator
shape = 'SPT'
solver = 'human'
geometry = ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx')
size = 'Large'

maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)
load = maze.bodies[-1]

maze_bb = Maze(size=size, shape=shape, solver=solver, geometry=geometry, bb=True)
load_bb = maze_bb.bodies[-1]

maze_corners = np.array_split(maze.corners(), maze.corners().shape[0]//4)



from itertools import product, combinations

x = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
y = (0, maze.arena_height)
z = (0, 2*np.pi)
points = tuple(product(*(x, y, z))) + ((20, 10, np.pi),)
points = np.array(points, float)

# rescale z axis - it's rescaled back in is_feasible
z_scale = maze.average_radius()
points[..., 2] *= z_scale

max_vol, min_vol, quality = 5, 1., 1.3  # 15 minutes, 100mb, and it... Kinda misses tunnels, damnit
points, tet, signs, final_index, former_found_arr = get_tetrahedralization(is_feasible, points, max_vol, min_vol, quality, max_iterations=10)

plot_tetrahedrons(points, tet, signs, former_found_arr)