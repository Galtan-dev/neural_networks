#!/usr/bin/env python3
# file: pca_first.py
import argparse
import os

# uncomment to run on CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import torch

print("Num GPUs Available: ", torch.cuda.device_count())

from mnist import MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use. Zero value for default.")


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    # import random
    # random.seed(args.seed)
    # import numpy as np
    np.random.seed(args.seed)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(len(mnist.train.images), size=args.examples, replace=False)
    data = torch.tensor(mnist.train.images[data_indices] / 255, dtype=torch.float32)

    # TODO: Original data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    #   We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    #   We can do so using `torch.reshape(data, new_shape)` with new shape
    #   `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = data.reshape([data.shape[0], -1])    # -1 dopočítá tu dimenzi, nebo bychom tam mohli dát to v tom TODO

    # TODO: Now compute mean of every feature. Use `torch.mean`, and set
    #   `dim` argument to zero -- therefore, the mean will be
    #   computed across the first dimension, so across examples.
    #   Note that for compatibility with Numpy/TF/Keras, all `dim` arguments
    #   in PyTorch can be also called `axis`.
    mean = torch.mean(data, dim=0)

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    #   where transpose can be computed using `torch.transpose` or `torch.t` and
    #   matrix multiplication using either Python operator @ or `torch.matmul`.
    #   Note: Covariance matrix is feature-by-feature shaped and thus is relatively
    #   independent on amount of samples used.
    #   https://towardsdatascience.com/understanding-the-covariance-matrix-92076554ea44
    dm = data - mean
    cov = torch.matmul(dm.t(), dm)/data.shape[0]

    # TODO: Compute the total variance, which is the sum of the diagonal
    #   of the covariance matrix. To extract the diagonal use `torch.diagonal`,
    #   and to sum a tensor use `torch.sum`.
    total_variance = torch.sum(torch.diagonal(cov))

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    #   Start with a vector of `cov.shape[0]` ones of type `torch.float32` using `torch.ones`.
    v = torch.ones(cov.shape[0], dtype=torch.float32)
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        #   1. v = cov v
        #      The matrix-vector multiplication can be computed as regular matrix multiplication
        #      or using `torch.mv`.
        v = torch.mv(cov, v)
        #   2. s = l2_norm(v)
        #      The l2_norm can be computed using for example `torch.linalg.vector_norm`.
        s = torch.linalg.vector_norm(v)
        #   3. v = v / s
        v = v / s

    # The `v` is now approximately the eigenvector of the largest eigenvalue, `s`.
    # We now compute the explained variance, which is the ratio of `s` and `total_variance`.
    explained_variance_ratio = s / total_variance

    # Return the total and explained variance
    return total_variance, explained_variance_ratio

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance_ratio = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(100*explained_variance_ratio))

    # vyśledek je cca 10%, potvrzuje to že tam jsou rovnoměrně rozložená čísla. Je to pro první komponentu
