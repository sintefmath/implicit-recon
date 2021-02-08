from pathlib import Path
import numpy as np
import os
import argparse
import time

from PIL import Image

from skimage import measure
from stl import mesh


def export_mesh(fname, faces, verts, flip=True):
    """
    :param str/Path fname: Path to the output file
    :param np.array faces: Numpy array of size #faces x 3
    :param np.array verts: Numpy array of size #vertices x 3
    :param bool     flip : Whether to flip the orientation of the faces by reversing the order of its vertices
    :return:
    """
    my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            if flip:
                my_mesh.vectors[i][j] = verts[f[-j], :]
            else:
                my_mesh.vectors[i][j] = verts[f[j], :]

    print(fname)
    my_mesh.save(fname)


def get_image(fname, box, config):
    """
    :param str       fname : The filename of the image to be loaded
    :param tuple     box   : A box used for cropping the image
    :param namespace config: Configuration namespace
    :return: Processed image as a numpy array
    """
    img = Image.open(config.in_path / fname)
    img = img.crop(box)
    if config.out_res is not None:
        x_res = config.out_res[0] if config.out_res[0] > 0 else img.size[0]
        y_res = config.out_res[1] if config.out_res[1] > 0 else img.size[1]
        img = img.resize((x_res, y_res))

    return img


def reduce_images(fnames, div, config, z_stride=1):
    """
    :param list      fnames  : List of filenames
    :param int       div     : Index in filenames to be used as the first of z_stride images
    :param int       z_stride: Number of images to be used in the reduction
    :param namespace config  : Configuration namespace
    :return: An image as a numpy array that consolidates the z_stride images in the batch
    """
    box = (config.crop, config.crop, 720 - config.crop, 720 - config.crop)

    return np.mean([np.array(get_image(fnames[z_stride * div + mod], box, config)) for mod in range(z_stride)], axis=0)


def main(config):
    fnames = os.listdir(config.in_path)
    fnames = [fname for fname in fnames if fname.endswith("png") ]
    fnames.sort()
    final_slice = min(config.final_slice, len(fnames))
    fnames = fnames[config.init_slice:final_slice % len(fnames) + 1]

    if config.out_res is None or config.out_res[2] <= 0:
        z_res = len(fnames)
        z_stride = 1
    else:
        z_res = config.out_res[2]
        z_stride = len(fnames) // z_res

    voxels = np.array([reduce_images(fnames, div, config, z_stride=z_stride) for div in range(z_res)])

    verts, faces, normals, values = measure.marching_cubes(voxels, 0, spacing=(config.aspect_ratio, 1., 1))
    export_mesh(config.out_path / config.out_fname, faces, verts)


def parse_arguments(args=None):
    # Parsing arguments
    parser = argparse.ArgumentParser(prog='png_to_mesh', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Assemble an STL file from a directory with binary bitmaps ordered "
                                                 "lexicographically")
    parser.add_argument('--in_path', type=str, help="Path to the directory with binary bitmaps", default=".")
    parser.add_argument('--out_path', help='Output directory', type=str, default=".")
    parser.add_argument('--out_fname', default='model.stl', type=str)
    parser.add_argument('--out_res', nargs='+', type=int, default=None,  # default=[64, 64, 64],
                        help="Output resolution x_res y_res z_res")
    parser.add_argument('--crop', default=0, type=int)
    parser.add_argument('--init_slice', default=0, type=int, help="Index of first bitmap to consider")
    parser.add_argument('--final_slice', default=-1, type=int, help="Index of final bitmap to consider")
    parser.add_argument('--aspect_ratio', default=1.0, type=float, help="Aspect ratio across-slice/within-slice")
    config = parser.parse_args(args)
    config.in_path = Path(config.in_path)
    config.out_path = Path(config.out_path)

    assert config.out_res is None or len(config.out_res) == 3, "Argument out_res is not correctly specified"

    return config


if __name__ == "__main__":
    ARGS = None

    start_time = time.time()
    config_args = parse_arguments(args=ARGS)
    main(config_args)
    print(f"STL creation took {time.time() - start_time} seconds")
