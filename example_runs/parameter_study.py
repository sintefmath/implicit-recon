import argparse
from train import main as main_train
from train import parse_arguments as parse_arguments_train

# Parsing arguments
parser = argparse.ArgumentParser(prog='parameter_study', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Parameter study on the CHD dataset")

# Model parameters
parser.add_argument('--network', default="UNetImplicit", type=str, choices=["VGGTrunc1", "VGGTrunc2", "UNetImplicit"],
                    help="The network type")
parser.add_argument('--out_res', default=[64], nargs='+', type=int, help="Output resolutions")
parser.add_argument('--degree', default=[2], nargs='+', type=int, help="Degree of the splines")
parser.add_argument('--scalings', default=[3], nargs='+', type=int, help='Number of downsamplings (and up-convolutions)')

# Paths
parser.add_argument('--base_data_dir', default=".", type=str, help='Base directory for the training data')
parser.add_argument('--base_output_dir', default=".", type=str, help='Base directory for the output')
parser.add_argument('--sub_data_dir', default=".", type=str, help='Relative path to be appended to base_data_dir. '
                                                                  'Should be descriptive of the data, e.g. CT/C2T')
parser.add_argument('--image_dir', default="images", type=str, help='Relative path to ground truth images')
parser.add_argument('--mask_dir', default="masks", type=str, help='Relative path to ground truth masks')

# Training parameters
parser.add_argument('--n_epochs', default=101, type=int, help='Number of epochs for training')
config = parser.parse_args()

for o in config.out_res:
    for d in config.scalings:
        for p in config.degree:
            c = o // 2 ** d

            config.model_path = f"../example_output/saved_models/weights_chd_ct_table1-O{o}-d{d}-p{p}.pth"
            ARGS_TRAIN = ["--base_data_dir", config.base_data_dir, "--sub_data_dir", config.sub_data_dir,
                          "--base_output_dir", config.base_output_dir, "--image_dir", config.image_dir,
                          "--mask_dir", config.mask_dir, "--degree", str(p), "--scalings", str(d),
                          "--code_size", str(c), "--model_in", config.model_path, "--model_out", config.model_path,
                          "--network", config.network, "--n_epochs", str(config.n_epochs)]

            config_args_train = parse_arguments_train(args=ARGS_TRAIN)
            main_train(config_args_train)
