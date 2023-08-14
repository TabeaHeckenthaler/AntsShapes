from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter


def smooth_array(array, fps, kernel_size=None):
    if kernel_size is None:
        kernel_size = 8 * (fps // 4) + 1
    new_array = medfilt(array, kernel_size=kernel_size)
    new_array = gaussian_filter(new_array, sigma=kernel_size // 5)
    return new_array

    # fig_angle = px.line(x=new_array)
    # fig_angle.show()
