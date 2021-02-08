import SEAL as SE
import numpy as np
import torch


def bspline_collocation_matrix(d=1, n_in=128, n_out=512):
    """
    Generate the n_out x n_in collocation matrix of n_in uniform B-spline basis functions of degree d evaluated
    at n_out evenly spaced parameter values.

    :param int d    : Degree of the splines
    :param int n_in : Number of uniform B-spline basis functions
    :param int n_out: Number of evenly spaced parameter values for evaluation
    :return: Univariate collocation matrix as a 2-dimensional array
    """
    # Parameter values
    x = np.linspace(0, 1, n_out)

    # Spline space
    knots_x = SE.create_knots(0, 1, d, n_in)
    T = SE.SplineSpace(d, knots_x)
    fs = T.basis

    # Create a collocation matrix
    col_mat = [[fs[i0](x0)[0] for i0 in range(n_in)] for x0 in x]
    col_mat = np.array(col_mat)
    col_mat = torch.tensor(col_mat, dtype=torch.float).cuda()

    return col_mat


def evaluate_from_col_mat(coefficients, col_mat):
    """
    Given a univariate collocation matrix 'col_mat' and bivariate coefficient array 'coefficients' (with additional
    batch size dimension). Let f be the function given as the linear combination of the coefficients with the bivariate
    tensor-product basis of the univariate basis underlying the collocation matrix. Evaluate f at the pairs of parameter
    values underlying the collocation matrix using a two-stage tensor contraction, which is memory and computationally
    efficient.

    :param array coefficients: Trivariate array of size |batch| x |basis| x |basis|
    :param array col_mat     : Univariate collocation matrix of size |parameters| x |basis|
    :return: Evaluated functions as an array of size |batch| x |parameters| x |parameters|
    """
    z = torch.einsum('ik,mkl->mil', col_mat, coefficients)
    z = torch.einsum('jl,mil->mij', col_mat, z)

    return z


def get_level_set_from_coefficients(coefficients, col_mat):
    """
    Given a univariate collocation matrix 'col_mat' and bivariate coefficient array 'coefficients' (with additional
    batch size dimension). Let f be the function given as the linear combination of the coefficients with the bivariate
    tensor-product basis of the univariate basis underlying the collocation matrix. Compute:
     * |batch| x |parameters| x |parameters| array z of values of f at the pairs of parameter values underlying the
       collocation matrix,
     * |batch| x |parameters| x |parameters| binary array z_treshold representing the sign of z, i.e., with 1 at
       nonnegative values of z and 0 otherwise.

    :param array coefficients: Trivariate array of size |batch| x |basis| x |basis|
    :param array col_mat     : Univariate collocation matrix of size |parameters| x |basis|
    :return: |batch| x |parameters| x |parameters|
    """
    z = evaluate_from_col_mat(coefficients, col_mat)

    # For some reason this threshold is the inverse of the original done in the generation of the data. Not sure why.
    z_treshold = z.cpu().detach().numpy()
    z_treshold[z_treshold > 0] = 1
    z_treshold[z_treshold < 0] = 0

    return z, z_treshold
