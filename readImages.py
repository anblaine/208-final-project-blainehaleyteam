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

    return A
