from __future__ import print_function, division

import os
import shutil
import gzip
import tarfile
import zipfile

import six


class GzipFileProxy(object):
    """Proxy for gzip module to follow same api as tarfile and zipfile"""
    supported_extension = '.gz'

    @classmethod
    def open(cls, filepath):
        if not filepath.endswith(cls.supported_extension):
            raise ValueError(
                'only supports file names ending with '
                '"{}"'.format(cls.supported_extension))
        instance = cls()
        instance.filepath = filepath
        instance._file = gzip.open(filepath, 'rb')

        return instance

    def close(self):
        self._file.close()

    def extractall(self, path):
        source_filename = os.path.basename(self.filepath)
        target_filename = source_filename[:-len(self.supported_extension)]
        with open(os.path.join(path, target_filename), 'wb') as f_out:
            shutil.copyfileobj(self._file, f_out)

    @classmethod
    def is_gzipfile(cls, filepath):
        if not filepath.endswith(cls.supported_extension):
            return False
        try:
            t = gzip.open(filepath, 'rb')
            t.close()
            return True
        except Exception:
            # NOTE not verified what file formats are accepted or what exceptions
            # raised
            return False

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.__exit__(exc_type, exc_val, exc_tb)


def extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if path is None:
        path = os.path.abspath(os.path.join(file_path, os.path.pardir))

    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip', 'gzip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        elif archive_type == 'gzip':
            open_fn = GzipFileProxy.open
            is_match_fn = GzipFileProxy.is_gzipfile
        else:
            return False

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False
