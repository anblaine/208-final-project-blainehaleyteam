def readAll(directory, filetype, color_mode):
    """
    Returns a list of all images in the directory in the format of readImage()

    Input:  directory where images are stored,
            filetype (suggested to use .png),
            color mode (see readImage)
    """
    import glob

    images = []

    for filename in glob.glob(directory + '/*' + filetype):
        images.append(readImage(filename, color_mode))

    return images


def readImage(filename, color_mode):
    """
    Creates a numpy array for an image

    Input:  filename of image (in same directory),
            color mode (suggestions: "RGBA", "RGB", or "greyscale")
    """
    from scipy import misc
    import numpy as np

    if color_mode == "greyscale":
        color_mode = "L"

    image = misc.imread(filename, mode=color_mode)
    A = np.asarray(image)
    return A


def colorDecompose(image):
    """
    Returns the red, green, and blue channels of an image imported using
    readImage with mode 'RGB'

    Input: numpy image array
    """
    import numpy as np

    decomposed = np.transpose(image, (2, 0, 1))

    return decomposed[0], decomposed[1], decomposed[2]
