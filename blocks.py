def blockDecompose(image, block_size):
    blocks = []
    for i in range(0, (image.shape[0] / block_size)):
        for j in range(0, (image.shape[1] / block_size)):
            blocks.append(image[block_size * i:block_size * (i + 1),
                          block_size * j:block_size * (j + 1)])
    return blocks


def createBlocks(images, block_size):
    blocks = []
    for image in images:
        blocks = blocks + blockDecompose(image, block_size)

    return blocks


def vectorizeBlocks(blocks, block_size):
    import numpy as np
    vectors = []
    for block in blocks:
        vectors.append(np.reshape(block, block_size**2, order='C'))
    return vectors


def unvectorizeBlocks(vectors, block_size):
    import numpy as np
    blocks = []
    for vector in vectors:
        block = unvectorizeBlock(vector, block_size)
        blocks.append(np.asarray(block))
    return blocks


def unvectorizeBlock(vector, block_size):
    from numpy import zeros
    block = zeros((block_size, block_size))
    for i in range(0, block_size):
        for j in range(0, block_size):
            block[i, j] = vector[block_size * i + j]
    return block


def createTrain(image, block_size, num_blocks):
    from numpy.random import randint
    x_max = image.shape[1] - block_size
    y_max = image.shape[0] - block_size
    blocks = []
    points_x = []
    points_y = []
    for i in range(0, num_blocks):
        x_i = randint(x_max)
        y_i = randint(y_max)
        points_x.append(x_i)
        points_y.append(y_i)
        blocks.append(image[y_i:y_i + block_size,
                            x_i:x_i + block_size])
    return blocks, points_x, points_y
