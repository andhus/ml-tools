from __future__ import print_function, division

import os
import argparse
import urllib

import tarfile

from glob import glob


def mkdirp(path):
    """Recursively creates directories to the specified path"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('{} exists and is not a directory'.format(path))
    else:
        os.makedirs(path)


def unpack(filepath, target_dir=None):
    if target_dir is None:
        target_dir = os.path.abspath(os.path.join(filepath, os.pardir))

    if filepath.endswith("tar.gz") or filepath.endswith(".tgz"):
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=target_dir)
    elif filepath.endswith("tar"):
        with tarfile.open(filepath, "r:") as tar:
            tar.extractall(path=target_dir)
    else:
        raise ValueError('TODO')


def download(source_url, filepath):

    def report(*args):
        print("blocknum: {}, bs: {}, size: {}".format(*args))

    print('Downloading: {} to {}'.format(source_url, filepath))
    urllib.urlretrieve(source_url, filepath, reporthook=report)


def get_filenames_in_dir(directory):
    return [os.path.basename(fp) for fp in glob(os.path.join(directory, '*'))]


def has_files(directory, filenames):
    existing_files = get_filenames_in_dir(directory)
    return set(filenames).issubset(existing_files)


def assert_has_files(directory, filenames):
    existing_files = get_filenames_in_dir(directory)
    if not set(filenames).issubset(existing_files):
        raise AssertionError('TODO')


DATASET_ROOT = os.path.join('text', 'europarl', 'v7')
SOURCES = ['http://www.statmt.org/europarl/v7/fr-en.tgz']

REQUIRED_FILES = [
    'europarl-v7.fr-en.fr',
    'europarl-v7.fr-en.fr'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local-dataset-dir',
        type=str,
        default='/Users/andershuss/Datasets/')
    parser.add_argument(
        '--gcs-dataset-dir',
        type=str,
        default='gs://huss-ml-dev/datasets/')
    args = parser.parse_args()

    local_ds_root = os.path.join(args.local_dataset_dir, DATASET_ROOT)
    gcs_ds_root = os.path.join(args.gcs_dataset_dir, DATASET_ROOT)
    mkdirp(local_ds_root)

    if not has_files(local_ds_root, REQUIRED_FILES):
        for source in SOURCES:
            filepath = os.path.join(local_ds_root, os.path.basename(source))
            if not os.path.exists(filepath):
                download(source, filepath)
            print('Unpacking file...')
            unpack(filepath)

    assert_has_files(local_ds_root, REQUIRED_FILES)
    # TODO transfer to GCS
