tsne = numpy.random.rand(5,2)
trc_coords = numpy.random.rand(5,2)

dist_tsne_grid = np.zeros([tsne.shape[0], tsne.shape[0]])
dist_real_grid = np.zeros([tsne.shape[0], tsne.shape[0]])
diag = np.zeros([tsne.shape[0], tsne.shape[0]])

for i in np.arange(tsne.shape[0]):
    for j in np.arange(tsne.shape[0]):
        dist_tsne_grid[i, j] = np.sqrt((tsne[i, 0]-tsne[j,0])**2 + (tsne[i, 1]-tsne[j, 1])**2)
        diag[i,j] = i != j
        dist_real_grid[i, j] = np.sqrt((trc_coords[i, 0]-trc_coords[j,0])**2 + (trc_coords[i, 1]-trc_coords[j, 1])**2)


def dist2D(a):
    N = a.shape[0]
    A = np.matlib.repmat(a[:,0], N,1)
    B = np.matlib.repmat(a[:,0][np.newaxis,:].T, 1,N)

    C = np.matlib.repmat(a[:,1], N,1)
    D = np.matlib.repmat(a[:,1][np.newaxis,:].T, 1,N)

    return np.sqrt((A-B)**2 + (C-D)**2)

diag = 1-np.diag(np.ones((3,)))
