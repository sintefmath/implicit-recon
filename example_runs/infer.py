import json
import tqdm
import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/..")

import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.models.implicit_spline_net import VGGTrunc, UNetImplicit
from src.utilities.data_loaders import ImageDataSet
from src.utilities.spline_utils import evaluate_from_col_mat, bspline_collocation_matrix
# from src.utilities.spline_utils import get_level_set_from_coefficients


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(prog='infer', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Inference of a pretrained network on entire directory")
    # Model parameters
    parser.add_argument('--model_path', type=str, help="Path to the PyTorch model file", default=None)
    parser.add_argument('--network', default="UNetImplicit", type=str,
                        choices=["VGGTrunc1", "VGGTrunc2", "UNetImplicit"], help="The network type")
    parser.add_argument('--scalings', default=4, type=int, help='Number of downsamplings (and up-convolutions)')
    parser.add_argument('--filters', default=64, type=int, help='Number of filters of the first convolutional block')
    parser.add_argument('--code_size', default=8, type=int,
                        help='Spatial dimension of the bottleneck layer in the U-Net')
    parser.add_argument('--degree', default=1, type=int, help='Degree of the splines')
    parser.add_argument('--input_res', default=512, type=int, help='Resolution of the input images')
    parser.add_argument('--num_input_slices', default=1, type=int,
                        help='Input channel dimensions, i.e., number of input slices')

    # Paths
    parser.add_argument('--output_path', type=str, default=None,
                        help='Base path to directory for the output binary masks')
    parser.add_argument('--image_paths', nargs='+', type=str, default=None,
                        help="Directories with input images, to be processed in order of specification")
    parser.add_argument('--mask_path', type=str, default="masks",
                        help="Relative path for output binary masks")

    # Inference parameters
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size during inference')

    config = parser.parse_args(args)

    model_in = os.path.splitext(config.model_path)[0]

    # Update configuration from model file
    with open(model_in + '.json', 'r') as fp:
        dct = vars(config)
        dct_model = json.load(fp)

        # Overwrite parameters crucial specified by the model
        for key in ["network", "scalings", "filters", "code_size", "degree", "input_res", "output_res",
                    "num_input_slices"]:
            dct[key] = dct_model[key]

        dct["init_epoch"] = dct_model["epoch"]
        dct["epoch"] = dct_model["epoch"]

        config = argparse.Namespace(**dct)

    if config.network == "VGGTrunc1":
        config.output_res = 128
    elif config.network == "VGGTrunc2":
        config.output_res = 64
    else:
        config.output_res = config.code_size * 2**config.scalings

    # Paths
    config.output_path = Path(config.output_path)
    config.model_path = Path(config.model_path)
    config.mask_path = Path(config.mask_path)

    return config


def infer(implicit_spline_net, col_mat, imgs_loader, config):
    img_counter = 0
    for imgs in tqdm.tqdm(imgs_loader, desc='Inference', leave=True):
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        coeffs = implicit_spline_net(imgs).squeeze()
        coeffs = coeffs.view(imgs.shape[0], config.output_res, config.output_res)
        pred_evals = evaluate_from_col_mat(coeffs, col_mat).cpu().detach().numpy()
        for j in range(pred_evals.shape[0]):
            pred = pred_evals[j]
            pred[pred <= 0] = 0
            pred[pred > 0] = 255

            img = Image.fromarray(pred)
            img = img.convert("L")

            fname = str(img_counter).zfill(10)
            img.save(config.output_path / f"{fname}.png")
            img_counter += 1


def main(config):
    col_mat = bspline_collocation_matrix(d=config.degree, n_in=config.output_res, n_out=config.input_res)

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    # Instantiate a Pytorch neural network and load weights
    if config.network in ['VGGTrunc1', 'VGGTrunc2']:
        implicit_spline_net = VGGTrunc(config)
    else:
        implicit_spline_net = globals()[config.network](config)

    implicit_spline_net.load_state_dict(torch.load(config.model_path))
    implicit_spline_net.eval()

    if torch.cuda.is_available():
        implicit_spline_net.cuda()

    # Prepare the dataset
    imgs_dataset = ImageDataSet(config.image_paths, 512, in_channels=config.num_input_slices)
    imgs_loader = DataLoader(imgs_dataset, batch_size=config.batch_size, shuffle=False)

    infer(implicit_spline_net, col_mat, imgs_loader, config)


if __name__ == '__main__':
    ARGS = None

    config_args = parse_arguments(args=ARGS)
    main(config_args)
