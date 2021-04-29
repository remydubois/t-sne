from tsne.pca import project_pca
from sklearn.decomposition import PCA
import numpy as np


def test_answer():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(1000, 8))

    y_nb = project_pca(x, n_components=2)
    y_sk = PCA(2).fit_transform(x)

    # Adjust signs to match sklearn's
    signs = 2 * (np.sign(y_nb[0]) == np.sign(y_sk[0])) - 1.0
    y_nb = y_nb * signs

    np.testing.assert_allclose(y_nb, y_sk, err_msg="Projection differs")
