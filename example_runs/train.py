import matplotlib.pyplot as plt
import datetime
import json
import argparse
import tqdm
from pathlib import Path, PosixPath
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/..")

import torch
from torch.utils.data import DataLoader

from src.metrics.evaluation_metrics import mask_loss_function, mask_loss_function_iou, mask_loss_function_dice, \
    mask_loss_function_acc, mask_loss_function_mse, mask_loss_function_mae
from src.utilities.data_loaders import ProcessedDataSet
from src.utilities.spline_utils import bspline_collocation_matrix
from src.models.implicit_spline_net import VGGTrunc, UNetImplicit

from src.utilities.visualization import gt_prediction_comparison_figure

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def parse_arguments(args=None):
    # Parsing arguments
    parser = argparse.ArgumentParser(prog='train', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Train segmentation network based on implicit representation")

    # Model parameters
    parser.add_argument('--network', default="UNetImplicit", type=str, choices=["VGGTrunc1", "VGGTrunc2", "UNetImplicit"],
                        help="The network type")
    parser.add_argument('--scalings', default=4, type=int, help='Number of downsamplings (and up-convolutions)')
    parser.add_argument('--filters', default=64, type=int, help='Number of filters of the first convolutional block')
    parser.add_argument('--code_size', default=8, type=int, help='Spatial dimension of the bottleneck layer in the U-Net')
    parser.add_argument('--degree', default=1, type=int, help='Degree of the splines')
    parser.add_argument('--input_res', default=512, type=int, help='Resolution of the input images')
    parser.add_argument('--num_input_slices', default=1, type=int, choices=[1, 3],
                        help='Input channel dimensions, i.e., number of input slices')

    # Paths
    parser.add_argument('--base_data_dir', default=".", type=str,
                        help='Base directory for the training data')
    parser.add_argument('--sub_data_dir', default=".", type=str, help='Relative path to be appended to base_data_dir. '
                                                                      'Should be descriptive of the data, e.g. CT/C2T')
    parser.add_argument('--image_dir', default="images", type=str, help='Relative path to ground truth images')
    parser.add_argument('--mask_dir', default="masks", type=str, help='Relative path to ground truth masks')
    parser.add_argument('--base_output_dir', default=".", type=str, help='Base directory for the output')
    parser.add_argument('--model_in', default=None, type=str,
                        help='Start training from a trained network with compatible parameters by loading a .pth file')
    parser.add_argument('--model_out', default=None, type=str,
                        help='Save the trained network weights/configuration to a .pth/.json file')

    # Training parameters
    parser.add_argument('--init_epoch', default=0, type=int,
                        help='Initial epoch for training (e.g. when starting from pretrained network)')
    # parser.add_argument('--final_epoch', default=10, type=int,
    #                    help='Final epoch in training (i.e., num. epochs + init_epochs)')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size during training and testing')
    parser.add_argument('--seed', default=40, type=int, help='Seed for random number generator')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate for the training')
    parser.add_argument('--step_lr_size', default=200, type=int,
                        help='Number of epochs to wait before reducing LR, for StepLR scheduler')
    parser.add_argument('--loss_strings', nargs='+', default=['mse', 'iou', 'dice', 'acc'], type=str,
                        help='Loss functions to be considered.')
    parser.add_argument('--mae_loss', default=0, type=int, help='Coefficient of Mask-MAE loss term')
    parser.add_argument('--mse_loss', default=0, type=int, help='Coefficient of Mask-MSE loss term')
    parser.add_argument('--dice_loss', default=1, type=int, help='Coefficient of 1-Dice loss term')
    parser.add_argument('--iou_loss', default=0, type=int, help='Coefficient of 1-IoU loss term')
    parser.add_argument('--acc_loss', default=0, type=int, help='Coefficient of 1-Accuracy loss term')

    # Logging parameters
    parser.add_argument('--write_model_frequency', default=1, type=int,
                        help='Number of epochs to wait before storing model weights to disk')

    config = parser.parse_args(args)
    config.it_counter = 0

    # If model_in is specified, update configuration from model_in to ensure consistency
    if config.model_in is not None:
        try:
            # Import configuration from model file
            model_in = os.path.splitext(config.model_in)[0]
            with open(model_in + '.json', 'r') as fp:
                dct = vars(config)
                dct_model = json.load(fp)

                # Overwrite relevant parameters specified by the model
                for key in ["network", "scalings", "filters", "code_size", "degree", "input_res", "output_res",
                            "num_input_slices", "val_loss", "it_counter"]:
                    dct[key] = dct_model[key]

                dct["init_epoch"] = dct_model["epoch"]

                config = argparse.Namespace(**dct)
        except EnvironmentError:
            print("Unable to open input model configuration file, sticking to original parameters.")

    if config.network == "VGGTrunc1":
        config.output_res = 128
    elif config.network == "VGGTrunc2":
        config.output_res = 64
    else:
        config.output_res = config.code_size * 2 ** config.scalings

    # Factors for loss function
    config.coeffs_loss = {s: getattr(config, s + "_loss") for s in config.loss_strings}
    config.loss_string = "_".join(
        [str(round(config.coeffs_loss[s], 2)) + s for s in config.coeffs_loss if config.coeffs_loss[s]])
    config.val_loss = {s: {} for s in config.loss_strings}

    # Title of the images in Tensorboard
    long_to_short = {'network': 'net', 'scalings': 'sc', 'filters': 'f', 'code_size': 'b', 'degree': 'p',
                     'input_res': 'I', 'output_res': 'O', 'num_input_slices': 'nc', 'base_data_dir': 'bddir',
                     'sub_data_dir': 'sd_dir', 'image_dir': 'i_dir', 'mask_dir': 'm_dir', 'base_output_dir': 'bo_dir',
                     'model_in': 'm_in', 'model_out': 'm_out', 'n_epochs': 'ne', 'batch_size': 'bs',
                     'seed': 's', 'learning_rate': 'lr', 'step_lr_size': 's_lr_s', 'mae_loss': 'lmae', 'mse_loss': 'lmse',
                     'dice_loss': 'ldice', 'iou_loss': 'liou', 'acc_loss': 'lacc',
                     'write_model_frequency': 'wmf', 'loss_string': 'loss'}

    config.title = "-".join([val + "." + str(getattr(config, key)) for key, val in long_to_short.items()])

    # Paths
    config.train_dir = config.base_data_dir / Path(config.sub_data_dir) / Path("train")
    config.val_dir = config.base_data_dir / Path(config.sub_data_dir) / Path("val")
    config.base_output_dir = Path(config.base_output_dir)
    config.model_dir = config.base_output_dir / Path("example_output/saved_models")

    dtime = datetime.datetime.now().strftime('%b-%d-%y@%X')
    config.log_dir = config.base_output_dir / Path("example_output/logs") / config.title / dtime

    return config


def validate(implicit_spline_net, col_mat, test_loader, scheduler, epoch, config, summary_writer):
    with torch.no_grad():
        implicit_spline_net.eval()

        val_loss = 0
        val_losses = {s: 0 for s in config.coeffs_loss}
        for j, test_data in enumerate(tqdm.tqdm(test_loader, desc='Validating', position=0, leave=True)):
            test_img, test_lbl = test_data
            if torch.cuda.is_available():
                test_img = test_img.cuda()
                test_lbl = test_lbl.cuda()

            test_prediction = implicit_spline_net(test_img).squeeze()
            test_prediction = test_prediction.view(test_img.shape[0], config.output_res, config.output_res)

            for s in config.coeffs_loss:
                val_losses[s] += globals()["mask_loss_function_" + s](test_prediction, test_lbl, col_mat)

        for s in val_losses:
            val_losses[s] /= len(test_loader)
            if config.coeffs_loss[s]:
                val_loss += val_losses[s] * config.coeffs_loss[s]

        val_loss_items = [(val_losses[s].item(), config.coeffs_loss[s], s) for s in val_losses]
        val_loss_item = 0
        for item, w, s in val_loss_items:
            val_loss_item += w * item
            summary_writer.add_scalar(tag='Validation/loss_mask_' + s, scalar_value=item, global_step=config.it_counter)
            if s in ['dice', 'iou', 'acc']:
                summary_writer.add_scalar(tag='Metric/' + s, scalar_value=1 - item, global_step=config.it_counter)

            config.val_loss[s][epoch] = item

        print(f"Validation loss = {val_loss_item:.4f}")
        summary_writer.add_scalar(tag='Validation/loss', scalar_value=val_loss_item, global_step=config.it_counter)

        fig = gt_prediction_comparison_figure(test_img, test_prediction, test_lbl, epoch, val_loss, scheduler, col_mat)
        summary_writer.add_figure('Test/loss', figure=fig, global_step=epoch, close=False)
        plt.close(fig)

        return val_loss


def train(implicit_spline_net, col_mat, train_loader, val_loader, optimizer, scheduler, config, summary_writer):
    for epoch in range(config.init_epoch, config.init_epoch + config.n_epochs):
        print(f'\nEPOCH = {epoch} / {config.init_epoch + config.n_epochs}')
        for step, data in enumerate(tqdm.tqdm(train_loader, desc='Training epoch', leave=False)):
            implicit_spline_net.train()
            img, lbl = data

            if torch.cuda.is_available():
                img = img.cuda()
                lbl = lbl.cuda()

            predicted_coefficients = implicit_spline_net(img).squeeze()
            predicted_coefficients = predicted_coefficients.view(img.shape[0], config.output_res, config.output_res)
            optimizer.zero_grad()

            training_losses = {s: mask_loss_function(predicted_coefficients, lbl, col_mat, metric=s)
                               for s in config.coeffs_loss.keys()}

            loss = 0
            for s in config.coeffs_loss:
                if config.coeffs_loss[s]:
                    loss += training_losses[s] * config.coeffs_loss[s]

            loss.backward()
            optimizer.step()

            training_loss_items = [(training_losses[s].item(), config.coeffs_loss[s], s) for s in training_losses]
            training_loss_item = 0
            for item, w, s in training_loss_items:
                training_loss_item += w * item
                summary_writer.add_scalar(tag='Train/loss_mask_' + s, scalar_value=item, global_step=config.it_counter)

            summary_writer.add_scalar(tag='Train/loss_mask', scalar_value=training_loss_item, global_step=config.it_counter)

            config.it_counter += 1
            if step == 0:
                val_loss = validate(implicit_spline_net, col_mat, val_loader, scheduler, epoch, config, summary_writer)
                scheduler.step()  # scheduler.step(val_loss)  # scheduler.step(val_loss) with ReduceLROnPlateau

        if epoch % config.write_model_frequency == 0 or epoch == config.init_epoch + config.n_epochs - 1:
            if config.model_out is None:
                model_out_root = f'{config.model_dir}/{config.title}_{datetime.date.today()}_epoch{epoch}'
            else:
                model_out_root = os.path.splitext(config.model_out)[0]

            # Save weights to PyTorch model file
            torch.save(implicit_spline_net.state_dict(), model_out_root + ".pth")

            # Save corresponding configuration namespace to file
            with open(model_out_root + '.json', 'w') as fpath:
                config.epoch = epoch + 1
                dct = {k: v.__str__() if isinstance(v, PosixPath) else v for k, v in config.__dict__.items()}
                json.dump(dct, fpath)


def main(config):
    col_mat = bspline_collocation_matrix(d=config.degree, n_in=config.output_res, n_out=config.input_res)

    # Instantiate a Pytorch neural network
    if config.network in ['VGGTrunc1', 'VGGTrunc2']:
        implicit_spline_net = VGGTrunc(config)
    else:
        implicit_spline_net = globals()[config.network](config)

    # If model_in is specified, load pretrained weights
    if config.model_in is not None:
        try:
            model_in = os.path.splitext(config.model_in)[0]
            implicit_spline_net.load_state_dict(torch.load(model_in + ".pth"))
        except EnvironmentError:
            print("Unable to open input model weights file, starting from scratch.")

        implicit_spline_net.eval()

    if torch.cuda.is_available():
        implicit_spline_net.cuda()

    # summary(implicit_spline_net, (config.num_input_slices, config.input_res, config.input_res))
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    # Loading data
    train_data = ProcessedDataSet(config.train_dir, normalize_coefficients=False, in_channels=config.num_input_slices,
                                  img_dir=config.image_dir, mask_dir=config.mask_dir, output_size=config.input_res)
    val_data = ProcessedDataSet(config.val_dir, normalize_coefficients=False, in_channels=config.num_input_slices,
                                img_dir=config.image_dir, mask_dir=config.mask_dir, output_size=config.input_res)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(implicit_spline_net.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10,
    #                                                        factor=0.316)  # With scheduler.step(val_loss)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 120], gamma=0.1)  # For LR=0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr_size, gamma=0.1)
    if torch.cuda.is_available():
        implicit_spline_net.cuda()

    train(implicit_spline_net, col_mat, train_loader, val_loader, optimizer, scheduler, config, summary_writer)

    summary_writer.close()


if __name__ == '__main__':
    ARGS = None

    config_args = parse_arguments(args=ARGS)
    main(config_args)
