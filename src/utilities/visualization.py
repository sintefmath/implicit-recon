import numpy as np
from matplotlib import pyplot as plt

from src.utilities.spline_utils import get_level_set_from_coefficients
from src.utilities.spline_utils import evaluate_from_col_mat


def gt_prediction_comparison_figure(imgs, coeffs, gt_lbl, epoch, loss, scheduler, col_mat, linewidths=0.5, phase='val'):
    """
    Generate images for comparing prediction to ground truth.

    :param array        imgs      : |batch| x num_in_channels x in_res x in_res array with input images
    :param array        coeffs    : |batch| x out_res x out_res array with predicted coefficients
    :param array        gt_lbl    : |batch| x in_res x in_res array with ground truth mask
    :param int          epoch     : Current epoch
    :param float        loss      : Current (phase) loss
    :param lr_schedular scheduler : torch.optim.lr_scheduler for changing the learning rate while training
    :param array        col_mat   : Univariate collocation matrix size |parameters| x |basis|
    :param float        linewidths: Float or array of linewidths to be passed to mpl contour functionality
    :param str          phase     : 'train'/'val'/'test' phase
    :return: Matplotlib figure showing ground truth image, predicted mask, ground truth mask, predicted spline values
    """

    input_img = imgs[0].cpu().detach().numpy()
    mask_img = gt_lbl[0].cpu().detach().numpy()
    input_img = np.rollaxis(input_img, axis=0, start=3)
    z_threshold = get_level_set_from_coefficients(coeffs[:1, :, :], col_mat)[1][0]  # 0th batch element
    evaluated_predicted_spline = evaluate_from_col_mat(coeffs[:1, :, :], col_mat).cpu().detach().numpy()[0]

    fig = plt.figure()
    plt.axis('off')
    plt.subplot(2, 2, 1, xticks=[], yticks=[])
    plt.contour(mask_img, cmap=plt.cm.viridis, linewidths=linewidths)
    plt.imshow(input_img.squeeze(), cmap=plt.cm.gray)
    plt.xlabel('GT image')

    plt.subplot(2, 2, 4, xticks=[], yticks=[])
    plt.contour(mask_img, cmap=plt.cm.viridis, linewidths=linewidths)
    plt.imshow(z_threshold, cmap=plt.cm.gray, alpha=0.7)
    plt.xlabel('Predicted mask')

    plt.subplot(2, 2, 3, xticks=[], yticks=[])
    plt.contour(mask_img, cmap=plt.cm.viridis, linewidths=linewidths)
    plt.imshow(mask_img, cmap=plt.cm.gray, alpha=0.7)
    plt.xlabel('GT mask')

    plt.subplot(2, 2, 2, xticks=[], yticks=[])
    plt.imshow(evaluated_predicted_spline)
    plt.colorbar()
    plt.xlabel('Predicted spline val.')

    lr = scheduler.optimizer.param_groups[0]['lr']  # Alternat., lr = scheduler.get_lr()  # lr = scheduler.get_last_lr()
    plt.suptitle(f'Epoch = {epoch:d}  loss = {loss:.3f}  LR = {lr}  phase={phase}')

    return fig
