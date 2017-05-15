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


def readAll(directory, color_mode):
    """
    Returns a list of all images in the directory in the format of readImage()

    Input:  directory where images are stored,
            color mode (see readImage)
    """
    from glob import glob

    images = []
    for filename in glob('./Images/*.png'):
        images.append(readImage.readImage(filename, "RGB"))

    return images
