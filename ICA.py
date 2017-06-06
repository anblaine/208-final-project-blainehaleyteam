def ICA_basis(samples, n_components, num_iter=2000):
    from sklearn.decomposition import FastICA

    estimator = FastICA(n_components=n_components, whiten=True,
                        max_iter=num_iter)
    estimator.fit(samples)
    return estimator.components_


def train_ICA(images, block_size, num_samples=400, num_iter=2000):
    from blocks import createTrain, vectorizeBlocks
    import numpy as np
    train = []
    points_x = []
    points_y = []
    for image in images:
        ttmp, pxtmp, pytmp = createTrain(image, block_size, num_samples)
        train = train + ttmp
        points_x.append(pxtmp)
        points_y.append(pytmp)
    vectorized_train = vectorizeBlocks(train, block_size)
    basis = ICA_basis(np.matrix(vectorized_train), block_size ** 2, num_iter)
    return basis, train, points_x, points_y


def plot_gallery(title, images, n_col=8, n_row=8, image_shape=(8, 8)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()


def ICA_decompose(image, block_size, basis, tol):
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from blocks import vectorizeBlocks, blockDecompose
    import numpy as np

    omp = OrthogonalMatchingPursuit(tol=tol, normalize=True)

    blocks = blockDecompose(image, block_size)
    vectorized_blocks = vectorizeBlocks(blocks, block_size)

    omp.fit(np.transpose(np.matrix(basis)), np.transpose(vectorized_blocks))

    return omp.coef_, omp.intercept_, omp.n_iter_


def reconstruct(decompositions, intercepts, basis, block_size, image_shape):
    import numpy as np
    from blocks import unvectorizeBlock
    reconstructed_blocks = []
    for i in range(0, len(decompositions)):
        reconstructed_vector = np.full(block_size ** 2, intercepts[i])
        for j in range(0, len(basis)):
            reconstructed_vector = reconstructed_vector \
                + decompositions[i][j] * basis[j]
        reconstructed_blocks.append(unvectorizeBlock(reconstructed_vector, 
                                                     block_size))

    reconstructed_image = np.zeros(image_shape)
    offset = image_shape[0] / block_size
    for i in range(0, offset):
        x_start = block_size * i
        x_end = x_start + block_size
        for j in range(0, image_shape[1] / block_size):
            y_start = block_size * j
            y_end = y_start + block_size
            reconstructed_image[x_start:x_end, y_start:y_end] \
                += reconstructed_blocks[offset * i + j]
    return reconstructed_image
