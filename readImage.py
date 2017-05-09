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
