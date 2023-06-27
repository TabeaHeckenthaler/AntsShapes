import os.path

from Analysis.bottleneck_c_to_e.correlation_edge_walk_decision_c_e_ac import *
from Setup.Maze import Maze
import cv2
from PhysicsEngine.Display import Display
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from DataFrame.import_excel_dfs import dfs_ant
from trajectory_inheritance.get import get
from copy import deepcopy
import time

def save_coords():
    pass
    # this function is found in decay_of_prob_distribution_edge.py


def stack_of_images(xs, ys, thetas, maze, d) -> np.array:
    imgs = []
    for i in range(len(xs)):
        maze.set_configuration([xs[i], ys[i]], thetas[i])
        maze.draw(display=d)
        img = d.get_image()
        # only keep pixels that are red [255, 0, 0]
        img_load = np.logical_and(img[:, :, 0] == 250, img[:, :, 1] == 0)

        binary_image = np.uint8(img_load)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_load = np.zeros_like(binary_image)
        img_load = cv2.drawContours(img_load.copy(), contours, 0, 1, thickness=cv2.FILLED)
        imgs.append(img_load)
        d.renew_screen()
    return imgs


def filename_frame(key):
    filename = key.split(':')[0]
    frame = int(key.split(':')[1][1:])
    return filename, frame

def find_percentages_maximum(percentages_image: np.array, masks) -> list:
    percentages = []
    for mask in masks:
        cut_out = percentages_image[mask]
        percentages.append(np.round(np.max(cut_out), 2).astype(float))
    return percentages


def find_percentages_mean(percentages_image: np.array, masks) -> list:
    percentages = []
    for mask in masks:
        cut_out = percentages_image[mask]
        percentages.append(np.round(np.mean(cut_out), 2).astype(float))
    return percentages


def shift_image(image, x, y):
    image = np.roll(image, y, axis=0)
    image = np.roll(image, x, axis=1)
    return image


def get_maze_boundary_mask(d, maze):
    d.renew_screen()
    maze.set_configuration([-1, -1], 0)
    maze.draw(display=d)
    img = d.get_image()
    maze_boundary = img[:, :, 0] == 0
    return maze_boundary


def plot_entire_maze_percentages(percentages_image, maze_boundary, key, text: str, masks, interactive=False):
    average_image = 1 - np.stack([percentages_image for _ in range(3)], axis=2)  # extend to 3 dimensions for plotting
    average_image_scaled = average_image / np.max(average_image)

    average_image_scaled[maze_boundary] = [1, 0, 0]  # where maze_boundary is true, draw red on average_image

    for mask in masks:
        average_image_scaled[mask] = [0, 1, 0]

    plt.imshow(average_image_scaled)

    if interactive:
        x, y = 0, 0
        plt.show()
        while input('continue? ') != 'q':
            # shift the average_image to the right
            x = int(input('x   '))
            y = -int(input('y   '))

            average_image = 1 - np.stack([percentages_image for _ in range(3)],
                                         axis=2)  # extend to 3 dimensions for plotting
            average_image_scaled = average_image / np.max(average_image)
            average_image_scaled = shift_image(average_image_scaled, x, y)
            for mask in masks:
                average_image_scaled[mask] = [0, 1, 0]
            average_image_scaled[maze_boundary] = [1, 0, 0]

            plt.imshow(average_image_scaled)
            plt.show()

        # open an excel sheet
        shifts = pd.read_excel(folder + 'shifts.xlsx', index_col=0)
        shifts.loc[key, 'x'] = x
        shifts.loc[key, 'y'] = y
        shifts.to_excel(folder + 'shifts.xlsx')

    plt.gca().text(40, 40, text)
    plt.tight_layout()
    plt.savefig(folder + '\\' + size + '\\' + key.replace(': ', '_') + '_entire_maze.png')
    plt.close()


def plot_histogram(percentages_image, key):
    plt.hist(percentages_image.flatten(), bins=20)
    plt.yscale('log')
    plt.xlim([0, 1])
    plt.savefig(folder + '\\' + size + '\\' + + key.replace(': ', '_') + '_values.png')
    plt.close()


def get_masks_corners(percentages_image, d, maze):
    y2_up, y2_down, x2 = {'XL': [273, 410, 678],
                          'L': [277, 413, 695],
                          'M': [262, 390, 655],
                          'S': [174, 276, 624],
                          }[size]
    # cut out square from average_image around (y1, x1) with size 5
    cut_out_size = int(d.ppm * np.diff(maze.slits) / 20)
    masks, contour_masks = [], []
    for y2 in [y2_up, y2_down]:
        mask = np.zeros_like(percentages_image)
        mask[y2 - cut_out_size: y2 + cut_out_size, x2 - cut_out_size: x2 + cut_out_size] = 1
        masks.append(mask.astype(bool))

        contour = plt.contour(mask, levels=[0.5], colors='red')

        # Extract contour coordinates
        contour_coords = contour.collections[0].get_paths()[0].vertices
        contour_coords = np.round(contour_coords).astype(int)

        # Create a blank mask of the same size
        contour_mask = np.zeros_like(mask)
        # Set contour coordinates to 1 in the mask
        contour_mask[contour_coords[:, 1], contour_coords[:, 0]] = 1
        contour_masks.append(contour_mask.astype(bool))
    return masks, contour_masks


def get_mask_slit(percentages_image, d, maze):
    y2_up, y2_down, x2 = {'XL': [273, 410, 678],
                          'L': [277, 413, 695],
                          'M': [262, 390, 655],
                          'S': [174, 276, 624],
                          }[size]
    # cut out square from average_image around (y1, x1) with size 5
    cut_out_size = int(d.ppm * np.diff(maze.slits) / 20)
    mask = np.zeros_like(percentages_image)
    mask[:, x2 - cut_out_size: x2 + cut_out_size] = 1

    contour = plt.contour(mask, levels=[0.5], colors='red')
    plt.close()

    # Extract contour coordinates
    contour_coords = contour.collections[0].get_paths()[0].vertices
    contour_coords = np.round(contour_coords).astype(int)

    # Create a blank mask of the same size
    contour_mask = np.zeros_like(mask)
    # Set contour coordinates to 1 in the mask
    contour_mask[contour_coords[:, 1], contour_coords[:, 0]] = 1
    return mask.astype(bool), contour_mask


def density_around_corner():
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    d = Display('', 1, maze)

    with open(folder + '\\' + size + '_percentages_mean.json', 'r') as file:
        percentages = json.load(file)

    masks, contour_masks = None, None
    for i, key in tqdm(enumerate(all_coords_x.keys()), desc=size, total=len(all_coords_x.keys())):
        xs = all_coords_x[key]
        ys = all_coords_y[key]
        thetas = all_coords_theta[key]
        frames = all_coords_frames[key]

        if len(frames) > 60:

            directory = folder + '\\percentages_numpy\\' + key.replace(': ', '_') + '.npy'
            if os.path.exists(directory):
                percentages_image = np.load(directory)
            else:
                imgs = stack_of_images(xs, ys, thetas, maze, d)
                percentages_image = np.sum(imgs, axis=0) / imgs.shape[0]
                np.save(directory, percentages_image)

            if key in shifts.index:  # check whether the shift is already in the excel sheet
                x, y = shifts.loc[key, 'x'], shifts.loc[key, 'y']
                percentages_image = shift_image(percentages_image, x, y)

            if masks is None:
                masks, contour_masks = get_masks_corners(percentages_image, d, maze)
            percentages[key] = find_percentages_mean(percentages_image, masks)

            # if key not in shifts.index:
            #     plot_entire_maze_percentages(percentages_image, d, maze, key, str(percentages[key]), contour_masks,
            #                                  interactive=True)
            #     percentages[key] = find_percentages_mean(percentages_image, masks)

            # plot_histogram(percentages_image, key)
            maze_boundary = get_maze_boundary_mask(d, maze)
            plot_entire_maze_percentages(percentages_image, maze_boundary, key, 'mean: ' + str(percentages[key]), contour_masks)

    # save percentages in json file with name 'folder + 'single_exp_shift' + condition + '\\size'
    with open(folder + '\\' + size + '_percentages_mean.json', 'w') as file:
        json.dump(percentages, file)

    print(size)
    print(percentages.values())


def find_experiments_to_shift():
    with open(folder + '\\' + size + '_percentages.json', 'r') as file:
        percentages = json.load(file)

    # find experiments to shift
    experiments_to_shift = []

    for key in percentages.keys():
        if np.sum(percentages[key]) < 0.3:
            experiments_to_shift.append(key)

    # save in text_file
    with open(folder + '\\' + size + '_experiments_to_shift.txt', 'w') as file:
        file.write('\n'.join(experiments_to_shift))


def play_x(x, boolean_array, wait=0) -> None:
    x = deepcopy(x)
    my_maze = Maze(x)
    display = Display(x.filename, x.fps, my_maze, wait=10)

    i = 0

    while i < len(x.frames):
        if boolean_array[i]:
            color_background = (200, 250, 250)
        else:
            color_background = (250, 250, 250)
        display.renew_screen(movie_name=x.filename,
                             frame_index=str(x.frames[display.i]),
                             color_background=color_background)
        x.step(my_maze, i, display=display)
        if display is not None:
            end = display.update_screen(x, i)
            if end:
                display.end_screen()
                x.frames = x.frames[:i]
                break
        i += 1
        time.sleep(wait/1000)

    if display is not None:
        display.end_screen()


def find_stretches_in_which_in_contact_with_wall():
    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    d = Display('', 1, maze)
    edge_contact, edge_transport, moving = {}, {}, {}

    for i, key in tqdm(enumerate(all_coords_x.keys()), desc=size, total=len(all_coords_x.keys())):
        xs = all_coords_x[key]
        ys = all_coords_y[key]
        thetas = all_coords_theta[key]
        frames = all_coords_frames[key]

        if len(frames) > 60:
            imgs = stack_of_images(xs, ys, thetas, maze, d)

            if key in shifts.index:  # check whether the shift is already in the excel sheet
                x, y = shifts.loc[key, 'x'], shifts.loc[key, 'y']
                imgs = [shift_image(img, x, y) for img in imgs]

            # masks, contour_masks = get_masks_corners(imgs[0], d, maze)
            # upper = np.stack([img[masks[0]] for img in imgs]).mean(axis=1)
            # lower = np.stack([img[masks[1]] for img in imgs]).mean(axis=1)
            # edge_contact = (upper + lower) > 0.1
            # edge_transport1 = gaussian_filter1d(edge_transport, sigma=2)
            # upper = gaussian_filter1d(upper, sigma=2)
            # lower = gaussian_filter1d(lower, sigma=2)
            # plt.plot(frames, upper, label='upper')
            # plt.plot(frames, lower, label='lower')

            mask, contour_mask = get_mask_slit(imgs[0], d, maze)
            slit = np.stack([img[mask] for img in imgs]).mean(axis=1)
            slit = gaussian_filter1d(slit, sigma=2)  # smooth speed with gaussian filter
            edge_contact[key] = slit > 0.005

            speed = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + (np.diff(thetas) * maze.average_radius()) ** 2)
            speed = np.hstack((speed, speed[-1]))  # add 0 at the end
            speed = gaussian_filter1d(speed, sigma=2)  # smooth speed with gaussian filter
            moving[key] = speed > {'S': 0.05, 'M': 0.1, 'L': 0.2, 'XL': 0.4}[size] * 3/4

            edge_transport[key] = np.logical_and(moving[key], edge_contact[key])

            plt.close()
            # set xticks positions
            plt.xticks(list(range(0, len(frames), 100)), frames[::100])
            plt.plot(range(len(frames)), slit, label='slit_contact', color='k')
            plt.plot(range(len(frames)), speed, label='speed', color='blue')

            # plt.fill_between(frames, 0, 1, where=edge_transport[key], color='b', alpha=0.2, label='edge transport')
            plt.fill_between(range(len(frames)), 0, 1, where=edge_contact[key], color='r', alpha=0.3, label='edge_contact')
            plt.fill_between(range(len(frames)), 0, 1, where=moving[key], color='blue', alpha=0.3, label='moving')

            plt.legend()
            plt.ylim([0, 1])

            plt.savefig(folder + '\\stretches\\' + size + '\\' + key.replace(': ', '_') + '.png')
            plt.close()

            # filename, frame = filename_frame(key)
            # x = get(filename)
            # x = x.cut_off(frames=[frames[0], frames[-1]])
            # x.adapt_fps(new_fps=x.fps/(frames[1]-frames[0]))
            # play_x(x, edge_transport[key], wait=20)
            # play_x(x, edge_contact[key], wait=40)

    # save edge_contact, edge_transport, moving
    with open(folder + '\\stretches\\' + size + '\\edge_contact.json', 'w') as file:
        if type(list(edge_contact.values())[0]) != list:
            edge_contact = {key: edge_contact[key].tolist() for key in edge_contact.keys()}
        json.dump(edge_contact, file)
    with open(folder + '\\stretches\\' + size + '\\moving.json', 'w') as file:
        if type(list(moving.values())[0]) != list:
            moving = {key: moving[key].tolist() for key in edge_contact.keys()}
        json.dump(moving, file)


def plot_statistics_edge_transport():
    fig, ax = plt.subplots()
    for size in ['S', 'M', 'L', 'XL']:
        with open(folder + '\\stretches\\' + size + '\\edge_contact.json', 'r') as file:
            edge_contact = json.load(file)
        with open(folder + '\\stretches\\' + size + '\\moving.json', 'r') as file:
            moving = json.load(file)

        edge_transport = {key: np.logical_and(moving[key], edge_contact[key]) for key in edge_contact.keys()}

        to_plot = moving
        to_plot = {key: np.sum(to_plot[key])/len(to_plot[key]) for key in to_plot.keys()}
        # plot histogram of edge_transport
        values, bins = np.histogram(list(to_plot.values()), bins=10, range=(0, 1), density=True)
        ax.plot(bins[:-1], values, label=size)
    plt.legend()
    plt.xlabel('moving')


def FourierTransformOfSpeedDirection():
    # TODO: Somehow show, that upon collision the direction changes slowly...
    #  maybe I can get the angle of the speed and the change of angle of the orientation vector.
    #  I expect to see more high frequencies in the smaller size, as opposed to the larger size.

    # the statement I would like to make is something like:
    # upon contact with the wall, the direction of motion changes slowly for the larger size,
    # but quickly for the smaller size.

    # load edge_contact, edge_transport, moving
    with open(folder + '\\stretches\\' + size + '\\edge_contact.json', 'r') as file:
        edge_contact = json.load(file)
    with open(folder + '\\stretches\\' + size + '\\moving.json', 'r') as file:
        moving = json.load(file)
    edge_transport = {key: np.logical_and(moving[key], edge_contact[key]) for key in edge_contact.keys()}

    for key in all_coords_x.keys():
        filename, frame = filename_frame(key)

        xs = all_coords_x[key]
        ys = all_coords_y[key]
        thetas = all_coords_theta[key]
        frames = all_coords_frames[key]

        traj = get(filename)
        traj = traj.cut_off(frames=[frames[0], frames[-1]])
        traj.adapt_fps(new_fps=traj.fps/(frames[1]-frames[0]))

        x = traj.position[:, 0]
        y = traj.position[:, 1]
        theta = traj.angle

        # get angle of vector (x, y)
        angle = np.arctan2(y, x)
        # get change in angle
        angle = np.unwrap(angle)
        # smooth
        angle = gaussian_filter1d(angle, sigma=2)
        dangle = np.diff(angle)
        # get change in orientation
        theta = np.unwrap(theta)
        # smooth
        theta = gaussian_filter1d(theta, sigma=2)
        dtheta = np.diff(theta)

        plt.plot(dangle, label='dangle')
        plt.plot(dtheta, label='dtheta')
        # draw a vertical line where the np.diff(edge_contact[key].astype(int)) is 1
        into_contact = np.where(np.diff(np.array(edge_contact[key]).astype(int)) == 1)[0]
        plt.vlines(into_contact, ymin=-np.max(dtheta), ymax=np.max(dtheta), color='r', label='into_contact')
        plt.legend()
        plt.show()
        plt.close()

        # get the fourier transform
        ft_dangle = np.fft.fft(dangle)
        ft_dtheta = np.fft.fft(dtheta)

        # plot the fourier transform
        plt.plot(ft_dangle, label='ft_dangle')
        plt.plot(ft_dtheta, label='ft_dtheta')
        plt.legend()
        plt.show()

        DEBUG = 1




    DEBUG = 1


def plot_percentages_histogram():
    fig, ax = plt.subplots()
    for size, df_size in dfs_ant.items():
        DEBUG = 0
        if size in ['S (> 1)', 'Single (1)']:
            sizename = 'S'
        else:
            sizename = size

        with open(folder + '\\' + sizename + '_percentages_mean.json', 'r') as file:
            percentages = json.load(file)

        # only keep the keys that are in the dataframe
        percentages = {key: percentages[key] for key in percentages.keys()
                       if np.any([key.split(':')[0] in f for f in df_size['filename']])}

        if len(percentages):
            percentages = np.array(list(percentages.values())).sum(axis=1)
            # percentages = np.array(list(percentages.values())).flatten()
            # percentages = np.array(list(percentages.values()))[:, 1]

            # plot_histogram of percentages
            values, bins = np.histogram(percentages, bins=10, range=[0, 1], density=True)
            plt.plot(bins[:-1], values, label=size)

    plt.xlim([0, 1])
    plt.xlabel('Fraction of contact time with front corners')
    plt.ylabel('Probability density')
    plt.legend()
    plt.title('')
    plt.savefig(folder + 'percentages_histogram.png')
    DEBUG = 1


def stack_all_images():
    imgs = []
    for i, key in tqdm(enumerate(all_coords_frames.keys()), desc=size, total=len(all_coords_frames.keys())):
        frames = all_coords_frames[key]

        if len(frames) > 60:
            directory = folder + '\\percentages_numpy\\' + key.replace(': ', '_') + '.npy'
            percentages_image = np.load(directory)

            if key in shifts.index:  # check whether the shift is already in the excel sheet
                x, y = shifts.loc[key, 'x'], shifts.loc[key, 'y']
                percentages_image = shift_image(percentages_image, x, y)

            imgs.append(percentages_image)

    imgs = np.array(imgs)
    average_image = np.sum(imgs, axis=0) / imgs.shape[0]
    average_image = 1 - np.stack([average_image for _ in range(3)], axis=2)  # extend to 3 dimensions for plotting

    maze = Maze(solver='ant', geometry=('MazeDimensions_new2021_SPT_ant.xlsx',
                                        'LoadDimensions_new2021_SPT_ant.xlsx'),
                size=size, shape='SPT')
    d = Display('', 1, maze)
    min, max = int(d.ppm * (maze.slits[0] - np.diff(maze.slits))), int(d.ppm * (maze.slits[1] + np.diff(maze.slits)))

    test = average_image[:, min: max, :]
    laplacian = cv2.Laplacian(test, cv2.CV_64F)
    variance = np.var(laplacian)
    print(size, ':    ', variance)
    plt.imshow(test)
    plt.title(size)
    plt.savefig(folder + size + '_contrast.png')

    maze_boundary_mask = get_maze_boundary_mask(d, maze)
    average_image[maze_boundary_mask] = [1, 0, 0]
    plt.imshow(average_image)
    plt.savefig(folder + size + '_average_image.png')
    plt.show()

    DEBUG = 1


if __name__ == '__main__':
    folder = 'results\\percentage_around_corner\\'
    # plot_percentages_histogram()
    # plot_statistics_edge_transport()

    # open the shift excel sheet
    shifts = pd.read_excel(folder + 'shifts.xlsx', index_col=0)

    # # ________________load coordinates _______________________
    for size in ['S', 'XL', 'L', 'M', ]:
        with open(folder + 'coordinates\\' + size + 'all_coords_x.json', 'r') as json_file:
            all_coords_x = json.load(json_file)
            json_file.close()

        with open(folder + 'coordinates\\' + size + 'all_coords_y.json', 'r') as json_file:
            all_coords_y = json.load(json_file)
            json_file.close()

        with open(folder + 'coordinates\\' + size + 'all_coords_theta.json', 'r') as json_file:
            all_coords_theta = json.load(json_file)
            json_file.close()

        with open(folder + 'coordinates\\' + size + 'all_coords_frames.json', 'r') as json_file:
            all_coords_frames = json.load(json_file)
            json_file.close()

        # ________________calculate calc_distances _______________________
        # filenames = {'L_SPT_4660001_LSpecialT_1_ants (part 1)': 50041,
        #              'XL_SPT_4630012_XLSpecialT_1_ants (part 1)': 61275,
        #              'S_SPT_5180005_SSpecialT_1_ants (part 1)': 17436,
        #              'M_SPT_4690010_MSpecialT_1_ants': 5722,
        #              }
        #
        # keys = [filename + ': ' + str(frame) for filename, frame in filenames.items()]
        #
        # # load experiments to shift
        # with open(folder + '\\' + size + '_experiments_to_shift.txt', 'r') as file:
        #     keys = file.read().splitlines()

        # # read file names in folder
        # keys = ['_'.join(exp[:-16].split('_')[:-1]) + ': ' + exp[:-16].split('_')[-1]
        #         for exp in os.listdir(folder + size + '\\to_shift')]

        keys = list(all_coords_x.keys())

        # reduce all_coords to the keys
        all_coords_x = {key: all_coords_x[key] for key in keys if key in all_coords_x.keys()}
        all_coords_y = {key: all_coords_y[key] for key in keys if key in all_coords_y.keys()}
        all_coords_theta = {key: all_coords_theta[key] for key in keys if key in all_coords_theta.keys()}
        all_coords_frames = {key: all_coords_frames[key] for key in keys if key in all_coords_frames.keys()}

        if len(all_coords_x):
            # find_experiments_to_shift()
            # density_around_corner()
            # stack_all_images()
            # find_stretches_in_which_in_contact_with_wall()
            FourierTransformOfSpeedDirection()
            pass


# TODO: sort percentages by smallest values
# TODO: do we have to shift things?
# TODO: save all the percentages in a dictionary and save it in a json file
# TODO: make a histogram of all the percentages for every size
