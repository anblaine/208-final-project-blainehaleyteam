def readAll(directory, filetype, block_size=0):
    """
    Returns a list of all images in the directory in the format of readImage()

    Input:  directory where images are stored,
            filetype (suggested to use .png),
            block size for cropping
    """
    import glob

    images = []

    for filename in glob.glob(directory + '/*' + filetype):
        images.append(readImage(filename, block_size))

    return images


def readImage(filename, block_size=0):
    """
    Creates a numpy array for an image

    Input:  filename of image (in same directory),
            block size for cropping
    """
    from scipy import misc
    import numpy as np

    image = misc.imread(filename, mode='F')
    A = np.asarray(image, dtype=np.float_)

    if (block_size):
        A = evenBlocksCrop(A, block_size)

    return A


def evenBlocksCrop(image, block_size):
    """
    Returns a cropped version of the image centered such that there's
    the maximum integer number of blocks

    Input: numpy image array
    """
    x_diff = image.shape[1] - block_size * (image.shape[1] / block_size)
    y_diff = image.shape[0] - block_size * (image.shape[0] / block_size)

    return image[y_diff / 2:-y_diff + (y_diff / 2),
                 x_diff / 2:-x_diff + (x_diff / 2)]
