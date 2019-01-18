from __future__ import print_function, division

import hashlib
import os
import tarfile
from contextlib import contextmanager


@contextmanager
def temp_dirtar(path):
    """Provides temporary tared version of directory"""
    temp_tar_path = path + '.tmp.tgz'
    if os.path.exists(temp_tar_path):
        raise IOError('{} already exists'.format(temp_tar_path))
    with tarfile.open(temp_tar_path, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
    try:
        yield temp_tar_path
    finally:
        os.remove(temp_tar_path)


def get_hash(filepath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    # Arguments
        filepath: path to the file or directory being validated. If a path to a
            directory is passed the directory is first tared before hashed.
        algorithm: hash algorithm, one of 'sha256', or 'md5'.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        The file hash
    """
    if algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'md5':
        hasher = hashlib.md5()
    else:
        raise ValueError('`algorithm` must one of "sha256" and "md5"')

    def _get_hash(fp):
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    if os.path.isdir(filepath):
        with temp_dirtar(filepath) as tarpath:
            return _get_hash(tarpath)

    return _get_hash(filepath)


def is_valid_hash(filepath, expected_hash, algorithm='sha256', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    # Arguments
        filepath: path to the file being validated
        expected_hash:  The expected hash string of the file.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        Whether the file is valid
    """
    if str(get_hash(filepath, algorithm, chunk_size)) == str(expected_hash):
        return True
    else:
        return False


class HashReference(object):

    default_algorithm = 'sha256'
    supported_algorithms = [default_algorithm, 'md5']

    def __init__(self, value, algorithm):
        self.value = value
        if algorithm not in self.supported_algorithms:
            raise ValueError('`algorithm` must be one of {}'.format(
                self.supported_algorithms)
            )
        self.algorithm = algorithm

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            value = config['value']
            algorithm = config.get('algorithm', cls.default_algorithm)
        else:
            value = config
            algorithm = cls.default_algorithm
        return cls(value, algorithm)

    def is_valid(self, path):

        return is_valid_hash(path, self.value, self.algorithm)

    def get_hash(self, path):

        return get_hash(path, algorithm=self.algorithm)

    def __eq__(self, other):
        return isinstance(other, HashReference) and (
            other.value == self.value and other.algorithm == self.algorithm
        )
