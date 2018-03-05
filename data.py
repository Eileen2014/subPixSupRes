from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from keras.preprocessing.image import ImageDataGenerator


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest, exist_ok=True)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def download_bsd500(dest="dataset"):
    output_image_dir = join(dest, "BSR/BSDS500/data/images")

    if not exists(output_image_dir):
        makedirs(dest, exist_ok=True)
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def image_generator():
    return ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect'
    )


def get_input_set(img_dims, batch_size=32, seed=42, interpolation="lanczos", path=None):
    if path is None:
        root_dir = download_bsd500()
    else:
        root_dir = path
    # train_dir = join(root_dir, "train")
    return image_generator().flow_from_directory(
        root_dir,
        target_size=img_dims,
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        interpolation=interpolation)


def get_output_set(img_dims, upscale_factor, batch_size=32, seed=42, interpolation="lanczos", path=None):
    if path is None:
        root_dir = download_bsd500()
    else:
        root_dir = path
    # test_dir = join(root_dir, "test")
    return image_generator().flow_from_directory(
        root_dir,
        target_size=tuple(upscale_factor * x for x in img_dims),
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        interpolation=interpolation)
