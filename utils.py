import torch
import numpy as np
import nibabel
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from nilearn.input_data import NiftiMasker


def prepare_face_data(sigma=0.3):
    """
    Prepares the Olivetti Faces data set
    Input:
        sigma: Float, controls the additive Noise
    Output:
        X_train, y_train: training data
        X_test, y_test: Test data
        masker: defines data graph
    """
    rnd = check_random_state(23)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rnd)
    X, y = dataset['images'], dataset['target']
    X_noisy = X + sigma * np.random.randn(X.shape[0], X.shape[1], X.shape[2])
    X_noisy = X_noisy.reshape(X_noisy.shape[0], X_noisy.shape[1] ** 2)
    n_x, n_y = int(np.sqrt(X_noisy.shape[1])), int(np.sqrt(X_noisy.shape[1]))
    p = np.prod([n_x, n_y, 1])
    mask = np.zeros(p).astype(np.bool)
    mask[:X_noisy.shape[-1]] = 1
    mask = mask.reshape([n_x, n_y, 1])
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(mask.astype(np.float), affine)
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.33, random_state=10)
    return X_train, X_test, y_train, y_test, masker


def set_torch_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)