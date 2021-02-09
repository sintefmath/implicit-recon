from pathlib import Path
import skfmm
import os
import argparse
import numpy as np
from PIL import Image
import PIL
import nibabel as nib
import matplotlib.pyplot as plt


def load_image_array_from_file(filename):
    """
    Returns the specified image as an integer-array.

    :param str filename: path to file
    :return: integer array
    """
    return np.array(Image.open(filename), dtype=np.int)


def redefine_true_false_values(arr):
    """
    Sets all zeros in the array to -1. This is to make sure the fast-marching-method works properly.

    :param np.array arr: array to modify
    :return: modified array
    """
    array = arr.copy()
    array[array == 0] = -1

    return array


def array_to_distance_field(arr):
    """
    Computes and returns the signed-distance-to-boundary array corresponding to the given array.

    :param np.array arr: input array
    :return: signed-distance-field
    """
    try:
        ret_arr = skfmm.distance(arr)
    except:
        ret_arr = np.zeros(arr.shape[0:2])

    return ret_arr


def array_to_distance_field_slicewise(arr):
    """
    Computes and returns the signed-distance-to-boundary array for each slice separately.

    :param np.array arr: input array
    :return: signed-distance-field
    """
    slices = []
    for i in range(0, arr.shape[2]):
        slices += [array_to_distance_field(arr[:, :, i])]

    return np.moveaxis(np.array(slices), [0], [2])


def distance_field_to_point_cloud(arr):
    """
    Converts the array of shape (m, n) to an array of shape (m, n, 3)
    representing the array as a point cloud.

    :param np.array arr: array to convert
    :return: a point cloud of shape (m, n, 3)
    """
    m, n = arr.shape
    pc = np.zeros((m * n, 3))

    for i in range(m):
        for j in range(n):
            pc[i * m + j] = i, j, arr[i, j]

    return pc


def save_array_as_image(arr, im_dir, im_id):
    """
    Saves image stored in array to directory with given label id

    :param np.array arr   : input image array
    :param str      im_dir: output directory
    :param          im_id : output id label
    """
    imrange = max(arr.shape) // 4
    plt.imshow(arr, vmin=-imrange, vmax=imrange)
    plt.axis('off')
    out_im = im_dir / f'{im_id:010d}.png'
    plt.savefig(out_im)
    plt.clf()


def save_array_as_point_cloud_csv(arr, csv_dir, csv_id):
    """
    Saves array as a point cloud in csv format to given directory and label id

    :param np.array  arr    : input array
    :param str       csv_dir: output directory
    :param           csv_id : output id label
    """
    m, n = arr.shape
    point_cloud = np.zeros((m * n, 3))
    for k in range(m):
        for l in range(n):
            point_cloud[k * n + l] = k, l, arr[k, l]

    out_file = csv_dir / f'{csv_id:010d}.csv'
    np.savetxt(out_file, point_cloud, delimiter=' ', fmt=['%d', '%d', '%f'])


def resize_3d_array_slicewise(arr, res, resample=3):
    """
    Resizes array slice-by-slice 

    :param arr: input array (must be 3D)
    :param res: output resolution will be (res, res, num_slices)
    :param resample: resample technique (see PIL.Image.resize documentation)
    """
    ret_arr = np.zeros((res, res, arr.shape[2]))
    for i in range(0, ret_arr.shape[2]):
        ret_arr[:, :, i] = np.array(Image.fromarray(arr[:, :, i]).resize((res, res), resample))

    return ret_arr


def create_directories(config):
    """
    Create directories for the output.

    :param namespace config: Namespace containing the parameters of the script.
    """
    assert not os.path.exists(config.output_dir), f"Output dir {config.output_dir} already exists. Aborting."

    os.mkdir(config.output_dir)

    for data_vol in config.val_test_train:
        data_vol_dir = config.output_dir / data_vol
        os.mkdir(data_vol_dir)

        if config.single_slice_images:
            os.mkdir(data_vol_dir / f"images_{config.out_res}/")
        if config.three_slice_images:
            os.mkdir(data_vol_dir / f"images_3_slice_{config.out_res}/")
        for mask_type in config.mask_types:
            os.mkdir(data_vol_dir / Path(mask_type + f"_masks_{config.out_res}/"))
            if config.distance_fields_3d:
                os.mkdir(data_vol_dir / Path(mask_type + f"_3d_distances_{config.out_res}/"))
                if config.save_distance_field_images:
                    os.mkdir(data_vol_dir / Path(mask_type + f"_3d_distance_images_{config.out_res}/"))

                os.mkdir(data_vol_dir / Path(mask_type + f"_3d_spline_{config.out_res}/"))
            if config.distance_fields_2d:
                os.mkdir(data_vol_dir / Path(mask_type + f"_2d_distances_{config.out_res}/"))
                if config.save_distance_field_images:
                    os.mkdir(data_vol_dir / Path(mask_type + f"_2d_distance_images_{config.out_res}/"))
                os.mkdir(data_vol_dir / Path(mask_type + f"_2d_spline_{config.out_res}/"))


def parse_arguments(args=None):
    # Parsing arguments
    parser = argparse.ArgumentParser(prog='compute_distance_fields', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Extract single/three-channels images and masks, resize if desired, and "
                                                 "compute distance fields")
    parser.add_argument('--train_idxs', nargs='+', type=int,
                        default=[1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 19, 20, 21, 22, 24, 26, 28, 29, 30, 34, 35, 36, 37, 38,
                                 39, 41, 42, 46, 47, 49, 56, 58, 59, 61, 63, 64, 65, 66, 67],
                        help='Indices of the volumes in the training set')  # 62 removed
    parser.add_argument('--val_idxs', nargs='+', type=int, default=[3, 6, 8, 10, 17, 31, 32, 33, 45, 50, 52, 54, 68],
                        help='Indices of the volumes in the validation set')  # 43 removed
    parser.add_argument('--test_idxs', nargs='+', type=int, default=[2, 12, 18, 23, 25, 27, 40, 44, 48, 51, 53, 55, 57, 60],
                        help='Indices of the volumes in the test set')
    parser.add_argument('--three_slice_images', dest='three_slice_images', default=True, action='store_true',
                        help='Whether to take 3 consecutive slices at a time')
    parser.add_argument('--single_slice_images', dest='single_slice_images', default=True, action='store_true',
                        help='Whether to take 1 slice at a time')
    parser.add_argument('--compute_3d_distance_fields', dest='distance_fields_3d', default=True, action='store_true',
                        help='Whether to compute distance fields to any voxel in the volume')
    parser.add_argument('--compute_2d_distance_fields', dest='distance_fields_2d', default=True, action='store_true',
                        help='Whether to compute distance fields to any pixel in the current slice')
    parser.add_argument('--save_distance_field_images', dest='save_distance_field_images', default=True,
                        action='store_true')
    parser.add_argument('--in_res', default=512, type=int, help="Input resolution")
    parser.add_argument('--out_res', default=128, type=int, help="Output resolution")
    parser.add_argument('--input_dir', default="./", type=str, help="Input path for image data")
    parser.add_argument('--mask_types', nargs='+', type=str, default=["BP", "MYO"], help="Mask types to process")

    config = parser.parse_args(args)

    config.input_dir = Path(config.input_dir)
    config.output_dir = config.input_dir / Path(f"output_{str(config.out_res)}/")
    config.val_test_train = {"train": config.train_idxs, "val": config.val_idxs, "test": config.test_idxs}
    config.resize = config.in_res != config.out_res

    return config


def main(config):
    create_directories(config)

    for data_vol in config.val_test_train:
        data_vol_dir = config.output_dir / data_vol

        image_id = 0
        mask_id = 0
        for vol_idx in config.val_test_train[data_vol]:
            print(f"Processing volume {vol_idx} as {data_vol} volume")
            vol_idx_str = str(vol_idx).zfill(2)

            for mask_type in config.mask_types:
                print("Processing " + mask_type)

                # Redefine paths
                mask_dir = data_vol_dir / Path(mask_type + f"_masks_{config.out_res}/")

                if config.distance_fields_2d:
                    _2d_distance_dir = data_vol_dir / Path(mask_type + f"_2d_distances_{config.out_res}/")
                    _2d_distance_field_image_dir = data_vol_dir / Path(mask_type + f"_2d_distance_images_{config.out_res}/")

                if config.distance_fields_3d:
                    _3d_distance_dir = data_vol_dir / Path(mask_type + f"_3d_distances_{config.out_res}/")
                    _3d_distance_field_image_dir = data_vol_dir / Path(mask_type + f"_3d_distance_images_{config.out_res}/")

                mask_id_tmp = mask_id

                # Load mask
                in_mask = config.input_dir / Path(mask_type + "/" + mask_type + vol_idx_str + ".nii.gz")
                mask_vol = nib.load(in_mask)

                # Prepare and compute distances
                mask_array = mask_vol.get_fdata()
                mask_array_tf = redefine_true_false_values(mask_array)
                if config.distance_fields_3d:
                    distances_3d = array_to_distance_field(mask_array_tf)
                if config.distance_fields_2d:
                    distances_2d = array_to_distance_field_slicewise(mask_array_tf)

                if config.resize:
                    mask_array = resize_3d_array_slicewise(mask_array, config.out_res, PIL.Image.NEAREST)
                    if config.distance_fields_3d:
                        distances_3d = resize_3d_array_slicewise(distances_3d, config.out_res, PIL.Image.BILINEAR)
                    if config.distance_fields_2d:
                        distances_2d = resize_3d_array_slicewise(distances_2d, config.out_res, PIL.Image.BILINEAR)

                # Prepare mask for output as 8-bit png
                mask_array = np.uint8(mask_array * 255)

                # Iterate through slices and output images and distances
                num_slices = mask_array.shape[2]
                for j in range(1, num_slices - 1):
                    # Output masks
                    mask = Image.fromarray(mask_array[:, :, j])
                    out_mask = mask_dir / f'{mask_id_tmp:010d}.png'
                    mask.save(out_mask)

                    # Output distance fields and images
                    if config.distance_fields_2d:
                        distance_2d = distances_2d[:, :, j]
                        save_array_as_point_cloud_csv(distance_2d, _2d_distance_dir, mask_id_tmp)
                        if config.save_distance_field_images:
                            save_array_as_image(distance_2d, _2d_distance_field_image_dir, mask_id_tmp)

                    if config.distance_fields_3d:
                        distance_3d = distances_3d[:, :, j]
                        save_array_as_point_cloud_csv(distance_3d, _3d_distance_dir, mask_id_tmp)
                        if config.save_distance_field_images:
                            save_array_as_image(distance_3d, _3d_distance_field_image_dir, mask_id_tmp)

                    # Increment id
                    mask_id_tmp += 1

            mask_id = mask_id_tmp

            # Load image and convert to uint8
            in_img = config.input_dir / f"IM52525/CTH0{vol_idx_str}_IMA.nii.gz"
            img_vol = nib.load(in_img).get_fdata()
            img_vol = np.uint8(np.clip(img_vol, 0, 2040) / 8)

            # Redefine paths
            im_3_slice_dir = data_vol_dir / f"images_3_slice_{config.out_res}/"
            im_dir = data_vol_dir / f"images_{config.out_res}/"

            # Output images
            num_slices = img_vol.shape[2]
            for j in range(1, num_slices - 1):
                if config.three_slice_images:
                    img3 = Image.fromarray(img_vol[:, :, j - 1:j + 2])

                    if config.resize:
                        img3 = img3.resize((config.out_res, config.out_res))

                    out_im3 = im_3_slice_dir / f'{image_id:010d}.png'
                    img3.save(out_im3)

                if config.single_slice_images:
                    img = Image.fromarray(img_vol[:, :, j])

                    if config.resize:
                        img = img.resize((config.out_res, config.out_res))

                    out_im = im_dir / f'{image_id:010d}.png'
                    img.save(out_im)

                image_id += 1


if __name__ == '__main__':
    # config = parse_arguments(['--input_dir', "/home/georgm/Dropbox/Data/Projects/ANALYST/CHD_orig/"])
    config = parse_arguments()
    main(config)
