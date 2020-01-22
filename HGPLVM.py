
import numpy as np
import math

import matplotlib.pyplot as plt
import tensorflow as tf

import gpflow
from gpflow.mean_functions import Zero
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter


class myGPLVM(gpflow.models.BayesianModel):
    def __init__(self, data, latent_data, x_data_mean, kernel):
        super().__init__()
        print("HGPLVM")
        self.iter = 0
        self.kernel0 = kernel[0]
        self.kernel1 = kernel[1]
        self.mean_function = Zero()
        self.likelihood0 = gpflow.likelihoods.Gaussian(1.0)
        self.likelihood1 = gpflow.likelihoods.Gaussian(1.0)

        # make some parameters
        self.data = (gpflow.Parameter(x_data_mean), gpflow.Parameter(latent_data), data)
        print("gpr_data", np.shape(self.data[0]), np.shape(self.data[1]), np.shape(self.data[2]))

    def hierarchy_ll(self):
        x, h, y = self.data
        K = self.kernel0(x)
        num_data = x.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood0.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(x)

        return multivariate_normal(h, m, L)

    def log_likelihood(self):
        """
        Computes the log likelihood.

        .. math::
            \log p(Y | \theta).

        """
        x, h, y = self.data
        K = self.kernel1(h)
        num_data = h.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood1.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(h)

        # m = zeros((np.shape(m)[0], 1))
        log_prob = multivariate_normal(y, m, L)
        log_prob_h = self.hierarchy_ll()
        log_likelihood = tf.reduce_sum(log_prob) + tf.reduce_sum(log_prob_h)
        self.iter += 1
        return log_likelihood


set_default_float(np.float64)

data_type = 'high_five'
# data_type = 'toy_example'
if data_type == 'toy_example':
    # generate 12d classification dataset
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=40, centers=3, n_features=12, random_state=2)
    # X, y = make_blobs(n_samples=199, centers=3, n_features=38, random_state=2)
    Y = tf.convert_to_tensor(X, dtype=default_float())
    labels = tf.convert_to_tensor(y)
elif data_type == 'high_five':
    from reader import *

    file_name = 'high_five.bvh'

    b = MyReader(file_name)
    mocap = b.read()
    Y = np.array(mocap[1])
    print("shape Y", shape(Y))
    # Y = np.reshape(Y, (np.shape(Y)[0], -1))
    Y = Y[:,0,:]
    Y = tf.convert_to_tensor(Y, dtype=default_float())


print("shape Y", np.shape(Y))
print('Number of points: {} and Number of dimensions: {}'.format(Y.shape[0], Y.shape[1]))

latent0_dim = 2  # number of latent dimensions
latent1_dim = 8  # number of latent dimensions from hidden layer
# num_inducing = 20  # number of inducing pts
num_data = Y.shape[0]  # number of data points

x_mean_latent = tf.convert_to_tensor(ops.pca_reduce(Y, latent1_dim), dtype=default_float())
x_mean_init = tf.convert_to_tensor(ops.pca_reduce(x_mean_latent, latent0_dim), dtype=default_float())

lengthscale0 = tf.convert_to_tensor([1.0] * latent0_dim, dtype=default_float())
lengthscale1 = tf.convert_to_tensor([1.0] * latent1_dim, dtype=default_float())
kernel0 = gpflow.kernels.RBF(lengthscale=lengthscale0)
kernel1 = gpflow.kernels.RBF(lengthscale=lengthscale1)

print("shape x_mean_latent", np.shape(x_mean_latent))
print("shape x_mean_init", np.shape(x_mean_init))
model = myGPLVM(Y, latent_data=x_mean_latent,
            x_data_mean=x_mean_init,
            kernel=[kernel0, kernel1])

model.likelihood0.variance.assign(0.01)
model.likelihood1.variance.assign(0.01)

opt = gpflow.optimizers.Scipy()
maxiter = ci_niter(1)

# test = m.hierarchy_ll()
# print("shape test", np.shape(test))


def closure():
    return - model.log_marginal_likelihood()


opt = gpflow.optimizers.Scipy()
_ = opt.minimize(closure, method="bfgs", variables=model.trainable_variables, options=dict(maxiter=maxiter))

print_summary(model)