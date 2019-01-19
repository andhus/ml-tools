from __future__ import print_function, division

import abc
import os
import tarfile
from warnings import warn
from pprint import pprint

import six

from ml_tools.dataset import url
from ml_tools.dataset.archive import extract_archive
from ml_tools.dataset.hash import HashReference


class LocalTarget(object):

    def __init__(self, path, dataset_root, hash_reference=None):
        self.path = path
        self.dataset_root = dataset_root
        if isinstance(hash_reference, HashReference):
            self.hash = hash_reference
        elif hash_reference is not None:
            self.hash = HashReference.from_config(hash_reference)
        else:
            self.hash = None

    @property
    def abspath(self):
        return os.path.join(self.dataset_root, self.path)

    def exists(self):
        return os.path.exists(self.abspath)

    def ready(self, check_hash=True):
        if not self.exists():
            return False

        if check_hash:
            if self.hash is None:
                raise ValueError('no hash provided for {}'.format(self.path))
            return self.hash.is_valid(self.abspath)

        return True

    @classmethod
    def from_config(cls, config, dataset_root):
        return cls(dataset_root=dataset_root, **config)

    def print_hash(self):
        hash_conf = self.hash or HashReference('')
        true_hash = hash_conf.get_hash(self.abspath)

        if hash_conf.algorithm == hash_conf.default_algorithm:
            hash_repr = true_hash
        else:
            hash_repr = {'algorithm': hash_conf.algorithm, 'value': true_hash}
        pprint({'path': self.path, 'hash': hash_repr})


def parse_target(target, dataset_root):
    if isinstance(target, LocalTarget):
        return target

    if not isinstance(target, dict):
        raise NotImplementedError(
            'target must either be an instance of LocalTarget or a dict '
            'with a valid target config',
        )

    return LocalTarget.from_config(target, dataset_root)


class SourceABC(LocalTarget):

    @abc.abstractmethod
    def fetch(self):
        raise NotImplementedError()


class URLSource(SourceABC):

    def __init__(
        self,
        url,
        extract='auto',
        **kwargs
    ):
        super(URLSource, self).__init__(**kwargs)
        self.url = url
        self.extract = extract

    @classmethod
    def from_config(cls, config, dataset_root):
        """From the type of config used to define datasets.

        simplest form; local path will be end of url `filename.tar.gz`
        {'url': 'http://www.domain.org/.../filename.tar.gz',
         'hash': 'cb8953f29229...'}

        specify path in dataset
        {'url': 'http://www.domain.org/.../filename.tar.gz',
         'path' new-filename.tgz,
         'hash': 'cb8953f29229...'}
        """
        config = config.copy()
        url = config.pop('url')
        path = config.pop('path', os.path.basename(url))

        return cls(url=url, path=path, dataset_root=dataset_root, **config)

    def fetch(self, check_hash=True):
        url.download(self.url, self.abspath)
        assert self.ready(check_hash)

# TODO
# class DatasetSource(SourceABC):
#
#     def __init__(self, dataset):
#         super(DatasetSource, self).__init__(None, None)
#         self.dataset = dataset
#
#     def fetch(self):
#         self.dataset.require()
#
#     def exists(self):
#         return self.dataset.exists()
#
#     def ready(self, check_hash=True):
#         return self.dataset.ready(check_hash)


def parse_source(source, dataset_root):
    if isinstance(source, SourceABC):  # TODO just check api.
        return source

    if not isinstance(source, dict):
        raise NotImplementedError(
            'source must either be an instance of Source or a dict '
            'with a valid source config',
        )

    if 'url' in source:
        return URLSource.from_config(source, dataset_root)
    else:
        raise NotImplementedError('Only URLSources supported.')


class Pack(LocalTarget):
    """TODO"""
    default_tar_extension = '.pack.tgz'

    def __init__(self, build_paths=None, **kwargs):
        super(Pack, self).__init__(**kwargs)
        self.build_paths = build_paths

    @classmethod
    def from_config(cls, config, dataset_root):
        """From the type of config used to define datasets.

        {
            'path': 'training-parallel-nc-v9-fr-en.pack.tgz',
            'hash': '02bcfae33373...',
            'build_paths': ['news-commentary-v9.fr-en.en', ...]
            'build_path': 'news-commentary-v9.fr-en.en'
        }
        """
        config = config.copy()
        build_paths = config.pop('build_paths')
        if isinstance(build_paths, six.string_types):
            build_paths = [build_paths]
        path = config.pop('path', None)
        if path is None:
            assert len(build_paths) == 1
            path = build_paths[0] + cls.default_tar_extension

        return cls(
            path=path,
            build_paths=build_paths,
            dataset_root=dataset_root,
            **config
        )

    def pack(self):
        with tarfile.open(self.abspath, "w:gz") as tar:
            for build_path in self.build_paths:
                build_abspath = os.path.join(self.dataset_root, build_path)
                tar.add(build_abspath, arcname=build_path)

    def unpack(self):
        extract_archive(self.abspath, path=self.dataset_root)


def parse_pack(pack, dataset_root):
    if isinstance(pack, Pack):
        return pack

    if not isinstance(pack, dict):
        raise NotImplementedError(
            'pack must either be an instance of Pack or a dict '
            'with a valid pack config',
        )

    return Pack.from_config(pack, dataset_root)
