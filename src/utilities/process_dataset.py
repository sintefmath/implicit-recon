import argparse
import json
from pathlib import Path

import numpy as np
import random
import skfmm
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tqdm import tqdm


def degenerate_bounding_box(bounding_box, eps=1.0e-14):
    """
    Checks a bounding box for whether it prescribes positive area or not. If not, return True, as the box is degenerate.

    :param bounding_box: axis-aligned bounding box, (x1, y1, x2, y2)
    :param eps: numerical tolerance
    :return: True of the box is degenerate, False otherwise.
    """

    x1, y1, x2, y2 = bounding_box
    if abs(y2 - y1) < eps or abs(x2 - x1) < eps:
        return True
    return False


def resize_crop_and_save_image(bounding_box, image_obj: Image, output_image_size, data_output_folder, img_id):
    """
    Given a bounding box, an image, the desired img output size, a data-folder, and an image_ID, resizes the image, and saves
    it.

    :param bounding_box: axis-aligned bounding box (x1, y1, x2, y2)
    :param image_obj: the image to crop.
    :param output_image_size: the desired output image-size
    :param data_output_folder: the desired output folder
    :param img_id: the image-id (or, filename)
    :return: the cropped image.
    """

    cropped_image = image_obj.crop(box=bounding_box)
    cropped_image = cropped_image.resize((output_image_size, output_image_size), Image.BILINEAR)
    file_name = f"{data_output_folder.resolve()}/{img_id:010d}.png"
    cropped_image.save(file_name, 'PNG')

    return cropped_image


def create_masked_image(polygon, original_image_size):
    """
    Given a polygon which delineates an object mask, creates a black and white image representing the full mask.

    :param polygon: a list of vertices
    :param original_image_size: the image size.
    :return: a masked image.
    """

    masked_img = Image.new('1', original_image_size, color=1)
    poly_draw = ImageDraw.Draw(masked_img)
    polygon = [(x, y) for x, y in polygon]
    poly_draw.polygon(polygon, fill=0, outline=0)

    return masked_img


def create_and_save_signed_distance_field_maps(masked_image, output_folder, image_id, threshold=200,
                                               save_distance_field_images=False, output_image_folder=None):
    """
    Given a masked image, computes and saves the corresponding signed-distance-field to the boundary.

    :param masked_image: masked image of interest
    :param output_folder: the desired output folder
    :param image_id: the image-id (filename)
    :return: None
    """

    fn = lambda x: 1 if x > threshold else 0
    masked_image = masked_image.convert('L').point(fn, '1')
    pixels = np.array(masked_image, dtype=np.int)

    if np.any(pixels):
        pixels[pixels == 0] = -1
        distances = skfmm.distance(pixels)
    else:
        pixels -= 1
        distances = np.zeros(pixels.shape)

    if save_distance_field_images and output_image_folder is not None:
        plt.imshow(distances)
        plt.axis('off')
        plt.savefig((output_image_folder / f'{image_id:010d}.png').resolve())
        plt.clf()

    m, n = distances.shape

    point_cloud = np.zeros((m * n, 3))

    for i in range(m):
        for j in range(n):
            point_cloud[i * n + j] = i, j, distances[i, j]

    file_name = (output_folder / f'{image_id:010d}.csv').resolve()
    np.savetxt(file_name, point_cloud, delimiter=' ', fmt=['%d', '%d', '%f'])


def process_ct_dataset(root_folder='.', data_set='val', output_folder='output', image_s=512, classes=None,
                       save_distance_field_images=True):
    """
    This function does all preprocessing needed on a fresh copy of the CHD CT dataset.

    :param str  root_folder  : The path to raw dataset directory
    :param str  data_set     : train/test/val
    :param str  output_folder: where to store the processed files
    :param str  image_s      : the resulting image and mask will be of size (output_image_size)^2.
    :param list classes      : classes to include. If not specified, we use a default selection.
    """

    # Load image
    # Load json

    # For each image:
    #    Rescale the image to output-image-size.
    #    Plot the object mask on a similarly sized image, black and white.
    #    Create a distance-map.
    #
    #    Save cropped image, cropped image mask, cropped distance map.

    root_folder = Path(root_folder)

    image_folder = root_folder / 'input_images' / data_set
    mask_folder = root_folder / 'input_masks' / data_set
    data_output_folder = root_folder / output_folder / data_set
    image_output_folder = data_output_folder / 'images'
    mask_output_folder = data_output_folder / 'masks'
    distance_field_output_folder = data_output_folder / 'distances'
    distance_field_image_output_folder = data_output_folder / 'distance_images'

    if not image_output_folder.exists():
        image_output_folder.mkdir(parents=True)

    if not mask_output_folder.exists():
        mask_output_folder.mkdir(parents=True)

    if not distance_field_output_folder.exists():
        distance_field_output_folder.mkdir(parents=True)

    if save_distance_field_images and not distance_field_image_output_folder.exists():
        distance_field_image_output_folder.mkdir(parents=True)

    cropped_image_counter = 0
    image_files = sorted(image_folder.glob('*.png'))
    mask_files = sorted(mask_folder.glob('*.png'))

    if classes is None:
        classes = ['heart']

    for image, mask in tqdm(zip(image_files, mask_files), desc=f'{"iterating over images":<50}', leave=False):
        image: Path
        label: Path

        image_obj = Image.open(image)
        cropped_image = image_obj.resize((image_s, image_s), Image.BILINEAR)
        file_name = f"{image_output_folder.resolve()}/{cropped_image_counter:010d}.png"
        cropped_image.save(file_name, 'PNG')

        mask_obj = Image.open(mask)
        cropped_mask = mask_obj.resize((image_s, image_s), Image.BILINEAR)
        file_name = f"{mask_output_folder.resolve()}/{cropped_image_counter:010d}.png"
        cropped_mask.save(file_name, 'PNG')

        create_and_save_signed_distance_field_maps(cropped_mask, distance_field_output_folder,
                                                   cropped_image_counter, 200, save_distance_field_images,
                                                   distance_field_image_output_folder)

        cropped_image_counter += 1


def process_c2t_dataset(root_path='.', out_path='./output', split={'train': 0.6, 'val': 0.2, 'test': 0.2},
                        split_type='sorted', image_s=512, segmentation_type="contours_filled_with_support_avg5"):
    """
    This function does all preprocessing needed on a fresh copy of the C2T dataset.
    """
    if split_type == "sorted":
        out_path = out_path + "sorted"

    root_path = Path(root_path)
    out_path = Path(out_path)

    # Input
    image_path_in = root_path / 'cropped'
    mask_path_in = root_path / segmentation_type
    assert image_path_in.exists(), "Image path is not found"
    assert mask_path_in.exists(), "Mask path is not found"
    assert not out_path.exists(), f"The output path {out_path} already exists. Remove it first."

    # Make output directories
    phases = list(split.keys())
    for phase in phases:
        image_path_out = out_path / phase / 'images'
        mask_path_out = out_path / phase / 'masks'

        if not image_path_out.exists():
            image_path_out.mkdir(parents=True)

        if not mask_path_out.exists():
            mask_path_out.mkdir(parents=True)

    cropped_image_counter = 0
    image_files = sorted(image_path_in.glob('*.png'))
    mask_files = sorted(mask_path_in.glob('*.png'))
    for image, mask in tqdm(zip(image_files, mask_files), desc=f'{"iterating over images":<50}', leave=False):
        if split_type == "sorted":
            if cropped_image_counter < split['train'] * len(image_files):
                phase = "train"
            elif cropped_image_counter < (split['train'] + split['val']) * len(image_files):
                phase = "val"
            else:
                phase = "test"
        else:
            phase = random.choices(list(split.keys()), list(split.values()))[0]

        image_obj = Image.open(image)
        image_cropped = image_obj.resize((image_s, image_s), Image.BILINEAR)
        image_path_out = out_path / phase / 'images'
        file_name = f"{image_path_out.resolve()}/{cropped_image_counter:010d}.png"
        image_cropped.save(file_name, 'PNG')

        mask_obj = Image.open(mask)
        mask_cropped = mask_obj.resize((image_s, image_s), Image.BILINEAR)
        mask_path_out = out_path / phase / 'masks'
        file_name = f"{mask_path_out.resolve()}/{cropped_image_counter:010d}.png"
        mask_cropped.save(file_name, 'PNG')

        cropped_image_counter += 1


def main():
    parser = argparse.ArgumentParser(prog='process_dataset', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Split the dataset into train, val and test data.")
    parser.add_argument('--image_s', default=512, type=int, help="Image resolution")
    parser.add_argument('--root_path', type=str, help="Root path to the data directory", default="./")
    config = parser.parse_args()

    config.out_path = config.root_path + f"out_{str(config.image_s)}"
    split = {'train': 0.4, 'val': 0.3, 'test': 0.3}
    process_c2t_dataset(root_path=config.root_path, out_path=config.out_path, image_s=config.image_s, split=split,
                        split_type="sorted")


if __name__ == '__main__':
    main()
