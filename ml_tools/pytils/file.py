from __future__ import print_function, division

import os

from glob import glob


def mkdirp(path):
    """Recursively creates directories to the specified path"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('{} exists and is not a directory'.format(path))
    else:
        os.makedirs(path)


def get_filenames(directory):
    return [os.path.basename(fp) for fp in glob(os.path.join(directory, '*'))]


def has_files(directory, filenames):
    existing_files = get_filenames(directory)
    return set(filenames).issubset(existing_files)


def assert_has_files(directory, filenames):
    existing_files = get_filenames(directory)
    if not set(filenames).issubset(existing_files):
        raise AssertionError('TODO')


def has_sub_paths(path, sub_paths):
    for sub_path in sub_paths:
        if not os.path.exists(os.path.join(path, sub_path)):
            return False
    return True
