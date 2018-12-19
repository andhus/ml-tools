from __future__ import print_function, division

import os
import argparse

from ml_tools.pytils.conf import ConfigMember, OneOf, IterableOf, IsInstance, IsNone
from ml_tools.pytils.file import mkdirp, has_files, has_sub_paths

from ml_tools.dataset.fetch_utils import get_file
from ml_tools.dataset.config import CONFIG

try:
    from google.cloud import storage as gcs
except ImportError:
    storage = None


class HashConfig(ConfigMember):
    """Can be used to validate"""
    # TODO remove?
    default = {
        'value': None,
        'algorithm': 'auto'
    }
    validate = {
        'value': OneOf(IsNone(), str),
        'algorithm': OneOf('md5', 'sha256')
    }


class SourceConfig(ConfigMember):
    """Can be used to validate"""
    # TODO remove?
    default = {
        'url': None,
        'filename': None,
        'hash': None,
        'extract': 'auto',
        'required_sub_paths': tuple([])
    }


class DatasetBase(object):
    relative_path = '.'
    sources = []

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def _get_path(cls, path):
        if path is not None:
            return path
        else:
            return os.path.join(CONFIG.root, cls.relative_path)

    @classmethod
    def exists(cls, path=None):
        path = cls._get_path(path)
        if not os.path.exists(path):
            return False
        for source in cls.sources:
            if not has_sub_paths(path, source['sub_paths']):
                return False
        return True

    @classmethod
    def assert_exists(cls, path=None):
        path = cls._get_path(path)
        if not cls.exists(path):
            raise AssertionError('TODO')

    @classmethod
    def require(cls, path=None):
        path = cls._get_path(path)
        if not cls.exists(path):
            mkdirp(path)
        for source in cls.sources:
            cls.require_source(source, path)

    @classmethod
    def require_source(cls, source, path):
        filename = source.get('filename', os.path.basename(source['url']))
        hash_config = source.get('hash', {'value': None, 'algorithm': None})
        missing_source = (
            CONFIG.keep_sources
            and not os.path.exists(os.path.join(path, filename))
        )
        if missing_source or not has_sub_paths(path, source['sub_paths']):
            if CONFIG.prefer_cloud:
                # TODO try fetch source from cloud
                pass
            get_file(
                fname=filename,
                origin=source['url'],
                extract=source.get('extract', 'auto'),
                cache_dir=path,
                cache_subdir='.',
                file_hash=hash_config.get('value'),
                hash_algorithm=hash_config.get('algorithm', 'auto')
            )
        if not CONFIG.keep_sources:
            os.remove(os.path.join(path, filename))

    @classmethod
    def upload(cls, path=None, destination=None):
        path = cls._get_path(path)
        if destination is None:
            destination = os.path.join(CONFIG.cloud.root, cls.relative_path)

        if destination.startswith('gs://'):
            if gcs is None:
                raise ImportError('TODO')
            bucket_name = destination.split('/')[2]
            client = gcs.Client()
            bucket = client.get_bucket(bucket_name)
            for source in cls.sources:
                filename = source.get('filename', os.path.basename(source['url']))
                filepath = os.path.join(path, filename)
                blob = bucket.blob(destination)
                blob.upload_from_filename(filepath)

    @classmethod
    def cmdline(cls):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(help='sub-command help')
        require_parser = subparsers.add_parser(
            'require',
            help='fetch {} dataset if not already available'.format(cls.name())
        )
        require_parser.add_argument(
            '--target-path',
            type=str,
            default=cls._get_path(None),
            help='local path for storing dataset'
        )
        require_parser.set_defaults(func=lambda args: cls.require(args.target_path))
        upload_parser = subparsers.add_parser(
            'upload',
            help='upload this dataset to cloud storage'
        )
        upload_parser.add_argument(
            '--target-path',
            type=str,
            default=os.path.join(CONFIG.cloud.root, cls.relative_path),
            help='target URI for storing dataset'
        )
        upload_parser.add_argument(
            '--source-path',
            type=str,
            default=os.path.join(CONFIG.cloud.root, cls.relative_path),
            help='local source path for to upload from storing dataset'
        )
        upload_parser.set_defaults(
            func=lambda args: cls.upload(
                path=args.source_path,
                destination=args.target_path
            )
        )
        args = parser.parse_args()
        return args.func(args)

    @classmethod
    def load_data(cls, path=None, **kwargs):
        __doc__ = cls._load_data.__doc__
        path = cls._get_path(path)
        return cls._load_data(path, **kwargs)

    def _load_data(self, path, **kwargs):
        raise NotImplementedError()
