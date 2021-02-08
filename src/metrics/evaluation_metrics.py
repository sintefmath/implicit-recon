import numpy as np
import torch
from src.utilities.spline_utils import evaluate_from_col_mat


def intersection_over_union(array1, array2, eps=10**(-6)):
    """
    Computes the intersection over union of the two binary arrays array1 and array2.

    :param array1: must contain 0s and 1s.
    :param array2: must contain 0s and 1s.
    :return: intersection over union
    """

    array1 = np.asarray(array1, dtype=bool).ravel()
    array2 = np.asarray(array2, dtype=bool).ravel()

    intersection = np.sum(np.bitwise_and(array1, array2))
    union = np.sum(np.bitwise_or(array1, array2))

    return intersection / (eps + union)


def intersection_over_union_torch(array1, array2, eps=10**(-6)):
    """
    Computes the intersection over union of the two binary arrays array1 and array2.

    :param array1: must contain 0s and 1s.
    :param array2: must contain 0s and 1s.
    :return: intersection over union
    """

    intersection = torch.sum(array1*array2)
    union = torch.sum(array1 + array2 - array1*array2)

    return intersection / (eps + union)


def dice_index(array1, array2, eps=10**(-6)):
    """
    Computes the dice index of the two binary arrays array1 and array2.

    :param array1: must contain 0s and 1s.
    :param array2: must contain 0s and 1s.
    :return: dice index
    """

    array1 = np.asarray(array1, dtype=bool).ravel()
    array2 = np.asarray(array2, dtype=bool).ravel()

    TP = np.sum(np.bitwise_and(array1, array2))
    FP = np.sum(np.bitwise_and(array1, np.invert(array2)))
    FN = np.sum(np.bitwise_and(array2, np.invert(array1)))

    return 2 * TP / (eps + 2 * TP + FP + FN)


def dice_index_torch(array1, array2, eps=10**(-6)):
    """
    Computes the dice index of the two binary arrays array1 and array2.

    :param array1: must contain 0s and 1s.
    :param array2: must contain 0s and 1s.
    :return: dice index
    """

    TP = torch.sum(array1 * array2)
    FP = torch.sum(array1) - TP
    FN = torch.sum(array2) - TP

    return 2 * TP / (eps + 2 * TP + FP + FN)


def accuracy_torch(array1, array2):
    """
    Computes the accuracy of the two binary arrays array1 and array2.

    :param array1: must contain 0s and 1s.
    :param array2: must contain 0s and 1s.
    :return: accuracy
    """

    TP = torch.sum(array1 * array2)
    TN = torch.sum(1 - array1 - array2) - TP
    FP = torch.sum(array1) - TP
    FN = torch.sum(array2) - TP

    return (TP + TN) / (TP + TN + FP + FN)


def product_loss(y, y_hat):
    """
    Computes a loss that penalizes the wrong sign
    :param y: predicted coefficients
    :param y_hat: ground truth coefficients
    :return:
    """

    def rebalance(z, delta, eps):
        return (z/torch.abs(z))*(delta+torch.mul(z, z))/(eps+torch.mul(z, z))

    eps = 1.0
    delta = 2.0
    y_new = rebalance(y, delta, eps)
    y_hat_new = rebalance(y_hat, delta, eps)
    y_diff = torch.abs(y_new+y_hat_new)
    e_y_diff = torch.exp(y_diff)
    return torch.mean(e_y_diff)


def mask_loss_function(c, m, col_mat, metric='dice', eps=10**(-4)):
    """
    :param array c: Predicted spline coefficients (batchsize x coeff_x_res x coeff_y_res, e.g 16 x 448 x 448)
    :param array m: Ground truth mask (batchsize x im_x_res x im_y_res, e.g. 16 x 56 x 56)
    :param array col_mat_uni: (im_res x coeff_res, e.g. 448 x 56)
    """

    m = m / torch.max(m)  # Ensure (0, 1) arrays
    z = evaluate_from_col_mat(c, col_mat)

    sgn = z / (eps + torch.abs(z))
    m_pred = (sgn + 1)/2

    loss = 0
    if metric == 'mse':
        y_diff = z - (2*m - 1)  # Density-based MSE
        loss = torch.mean(torch.mul(y_diff, y_diff))
    elif metric == 'mae':
        y_diff = z - (2*m - 1)
        loss = torch.mean(torch.abs(y_diff))
    elif metric == 'mse_sgn':
        y_diff = m - m_pred  # Implicit MSE
        loss = torch.mean(torch.mul(y_diff, y_diff))
    elif metric == 'iou':
        loss = 1 - intersection_over_union_torch(m, m_pred)
    elif metric == 'dice':
        loss = 1 - dice_index_torch(m, m_pred)
    elif metric == 'acc':
        loss = 1 - accuracy_torch(m, m_pred)

    return loss


# Separate loss functions are needed if querried multiple times for independent metrics
def mask_loss_function_mse(c, m, col_mat):
    return mask_loss_function(c, m, col_mat, metric='mse')


def mask_loss_function_mae(c, m, col_mat):
    return mask_loss_function(c, m, col_mat, metric='mae')


def mask_loss_function_iou(c, m, col_mat):
    return mask_loss_function(c, m, col_mat, metric='iou')


def mask_loss_function_dice(c, m, col_mat):
    return mask_loss_function(c, m, col_mat, metric='dice')


def mask_loss_function_acc(c, m, col_mat):
    return mask_loss_function(c, m, col_mat, metric='acc')
