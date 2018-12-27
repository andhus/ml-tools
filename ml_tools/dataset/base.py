from __future__ import print_function, division

import abc
import os
import argparse
import tarfile
import zipfile
from contextlib import contextmanager
from warnings import warn

from ml_tools.pytils.conf import ConfigMember, OneOf, IsNone
from ml_tools.pytils.file import mkdirp

from ml_tools.dataset.fetch_utils import (
    download,
    validate_file,
    extract_archive,
    hash_file
)
from ml_tools.dataset.config import CONFIG


try:
    from tensorflow.python.lib.io import file_io as tf_file_io
except ImportError:
    tf_file_io = None

try:
    from google.cloud import storage as gcs
except ImportError:
    storage = None

#
# class HashConfig(ConfigMember):
#     """Can be used to validate"""
#     # TODO remove?
#     default = {
#         'value': None,
#         'algorithm': 'auto'
#     }
#     validate = {
#         'value': OneOf(IsNone(), str),
#         'algorithm': OneOf('md5', 'sha256')
#     }
#
#
# class SourceConfig(ConfigMember):
#     """Can be used to validate"""
#     # TODO remove?
#     default = {
#         'url': None,
#         'filename': None,
#         'hash': None,
#         'extract': 'auto',
#         'required_sub_paths': tuple([])
#     }


class DatasetNotAvailable(IOError):
    pass


gcs_prefix = 'gs://'


class CloudIOError(IOError):
    pass


def get_gcs_bucket_and_object_name(uri):
    assert uri.startswith(gcs_prefix)
    splits = uri.split('/')  # [gs:, '', 'bucket', 'path' 'to', 'target']
    bucket_name = splits[2]
    object_name = '/'.join(splits[3:])
    return bucket_name, object_name


def _gcs_copy(source_filepath, target_filepath):
    """Copies a file to/from/within Google Cloud Storage (GCS).
    # Arguments
        source_filepath: String, path to the file on filesystem or object on GCS to
            copy from.
        target_filepath: String, path to the file on filesystem or object on GCS to
            copy to.
        overwrite: Whether we should overwrite an existing file/object at the target
            location, or instead ask the user with a manual prompt.
    """
    if tf_file_io is None:
        raise ImportError('Google Cloud Storage file transfer requires TensorFlow.')
    # if not overwrite and tf_file_io.file_exists(target_filepath):
    #     proceed = ask_to_proceed_with_overwrite(target_filepath)
    #     if not proceed:
    #         return
    with tf_file_io.FileIO(source_filepath, mode='rb') as source_f:
        with tf_file_io.FileIO(target_filepath, mode='wb') as target_f:
            target_f.write(source_f.read())


def save_to_cloud(source_path, target_uri):
    if target_uri.startswith(gcs_prefix):
        # prefer tensorflow option
        if tf_file_io is not None:
            _gcs_copy(source_path, target_uri)
            return

        if gcs is None:
            raise ImportError('you must have google.cloud.storage installed')
        bucket_name, object_name = get_gcs_bucket_and_object_name(target_uri)
        client = gcs.Client()
        bucket = client.get_bucket(bucket_name)
        print(
            'uploading {} to GCS bucket: {}, object: {}'.format(
                source_path, bucket_name, object_name
            )
        )
        blob = bucket.blob(object_name)
        blob.upload_from_filename(source_path)
    else:
        raise ValueError('cloud uri not supported')


def load_from_cloud(source_uri, target_path):
    if source_uri.startswith(gcs_prefix):
        # prefer tensorflow option
        if tf_file_io is not None:
            # TODO doesn't rais if file not available
            _gcs_copy(source_uri, target_path)
            return

        if gcs is None:
            raise ImportError('you must have google.cloud.storage installed')
        bucket_name, object_name = get_gcs_bucket_and_object_name(source_uri)
        client = gcs.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(object_name)
        if blob.exists():
            print(
                'loading GCS bucket: {}, object: {} to {}'.format(
                    bucket_name, object_name, target_path
                )
            )
            blob.download_to_filename(target_path)
        else:
            raise CloudIOError('object does not exists')
    else:
        raise ValueError('cloud target not supported')


# TODO
# - work with instances instead, class can have one class attribute config


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


class HashConfig(object):

    default_algorithm = 'sha256'

    def __init__(self, conf):
        if isinstance(conf, str):
            self.value = conf
            self.algorithm = self.default_algorithm
        else:
            self.value = conf['value']
            self.algorithm = conf.get('algorithm', self.default_algorithm)

    def validate(self, path):
        if os.path.isfile(path):
            return validate_file(path, self.value, self.algorithm)

        if os.path.isdir(path):
            with temp_dirtar(path) as tarpath:
                return validate_file(tarpath, self.value, self.algorithm)

        raise IOError('Could not verify {}'.format(path))


class LocalTarget(object):

    def __init__(self, path, dataset_root, hash=None):
        self.path = path
        self.hash = HashConfig(hash) if hash is not None else None
        self.dataset_root = dataset_root

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
                warn('no hash provided for {}'.format(self.path))
                return True
            return self.hash.validate(self.abspath)

        return True

    @classmethod
    def from_config(cls, config, dataset_root):
        return cls(dataset_root=dataset_root, **config)


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
    def require(self):
        raise NotImplementedError


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
        url = config.pop('url')
        path = config.pop('path', os.path.basename(url))

        return cls(url=url, path=path, dataset_root=dataset_root, **config)

    def fetch(self, check_hash=True):
        download(self.url, self.abspath)
        assert self.ready(check_hash)


class DatasetSource(SourceABC):

    def __init__(self, dataset):
        super(DatasetSource, self).__init__(None, None)
        self.dataset = dataset

    def require(self):
        self.dataset.require()

    def exists(self):
        return self.dataset.exists()

    def ready(self, check_hash=True):
        return self.dataset.ready(check_hash)


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
    """

    """
    default_tar_extension = 'pack.tgz'
    default_gz_extentision = 'pack.gz'

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
        build_paths = config.pop('build_paths')
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
                tar.add(build_path, arcname=os.path.basename(build_path))

    def unpack(self):
        with tarfile.open(self.abspath, "r:gz") as tar:
            tar.extractall()


def parse_pack(pack, dataset_root):
    if isinstance(pack, Pack):
        return pack

    if not isinstance(pack, dict):
        raise NotImplementedError(
            'pack must either be an instance of Pack or a dict '
            'with a valid pack config',
        )

    return Pack.from_config(pack, dataset_root)



# class SourcePack(LocalTarget):
#
#     def __init__(self, source):
#         super(SourcePack, self).__init__(None, None)
#         self.source = source
#

# class Builder(object):
#
#     def __call__(self, sources, builds):
#         raise NotImplementedError()
#
#
# class Extract(Builder):
#
#     def __call__(self, sources, builds):
#         for source in sources:
#             extract_archive(source.path, path=None, archive_format=source.extract)
#         # TODO verify that builds were generated
#
#
# class TarExtractSelect(Builder):
#
#     def __init__(self, tarpath_to_path):


class DatasetBase2(object):
    """
    Arguments:
        root_abspath: override path/to/dataset_root/
        dataset_home: override path/to/dataset_home/ should not be specified if
            root_abspath is given

    """

    # if packs not specified => use sources
    config = {
        'root': 'path/to/dsroot',
        'sources': [],
        'builds': [],
        'packs': [],
    }
    # builder = None
    # packer = None

    def __init__(
        self,
        root_abspath=None,
        dataset_home=None,
        cloud_root_abspath=None,
        cloud_dataset_home=None
    ):
        if root_abspath is not None:
            if dataset_home is not None:
                warn('dataset_home has no effect when root_abspath is specified')
            self.root_abspath = root_abspath
        else:
            if dataset_home is None:
                dataset_home = CONFIG.home
            self.root_abspath = os.path.join(dataset_home, self.config['root'])

        if cloud_root_abspath is not None:
            if cloud_dataset_home is not None:
                warn(
                    'cloud_dataset_home has no effect when cloud_root_abspath is '
                    'specified'
                )
            self.cloud_root_abspath = cloud_root_abspath
        else:
            if cloud_dataset_home is None:
                cloud_dataset_home = CONFIG.cloud.home
            self.cloud_root_abspath = os.path.join(
                cloud_dataset_home,
                self.config['root']
            )

        self.sources = [
            parse_source(s, self.root_abspath) for s in self.config['sources']]
        self.builds = [
            parse_target(b, self.root_abspath) for b in self.config['builds']]
        packs = self.config.get('packs', None)
        if packs:
            self.packs = [
                parse_pack(p, self.root_abspath) for p in self.config['packs']
            ]
        else:
            self.packs = None  # assuming sources used as packs

    def require(self, check_hash=True):
        if self.build_ready(check_hash):
            return

        print('checking if packed')
        # same as checking if sources exists if packs=None
        if self.pack_ready(check_hash):
            print('found packed, unpacking')
            self.unpack()
            assert self.build_ready()
            return

        print('checking if available from cloud')
        try:
            self.fetch_pack(check_hash)
            self.unpack()
            assert self.build_ready()
            return
        except Exception as e:
            print('failed fetching -> unpacking Error: {}'.format(e))

        print('check if sources exists')
        if self.sources_ready(check_hash):
            print('sources, exists, building...')
            self.build()
            assert self.build_ready(check_hash)
            return

        print('trying to fetch sources')
        self.fetch_sources()
        self.sources_ready(check_hash)
        self.build()
        assert self.build_ready(check_hash)
        # raise DatasetNotAvailable()

    def require_sources(self, check_hash=True):
        for source in self.sources:
            if not source.ready(check_hash):
                source.fetch()
                assert source.ready()

    def sources_ready(self, check_hash=True):
        for source in self.sources:
            if not source.ready(check_hash):
                return False
        return True

    def fetch_sources(self):
        for source in self.sources:
            source.fetch()

    def require_build(self, check_hash=True):
        """Default is to extract sources, run post_process"""
        if not self.build_ready(check_hash):
            self.build()
            assert self.build_ready()

    def build_ready(self, check_hash=False):
        for build in self.builds:
            if not build.ready(check_hash):
                return False
        return True

    def build(self):
        for source in self.sources:
            extract_archive(source.path, path=None, archive_format=source.extract)
        self.post_process()

    def pack(self):
        if self.packs is None:
            # using sources
            self.require_sources()
        else:
            for pack in self.packs:
                pack.pack()

    def pack_ready(self, check_hash):
        if self.packs is None:
            return self.sources_ready(check_hash)

        for pack in self.packs:
            if not pack.ready(check_hash):
                return False

        return True

    def unpack(self):
        if self.packs is None:
            # using sources
            self.require_build()
        else:
            for pack in self.packs:
                pack.unpack()

    def fetch_pack(self, check_hash=True):
        if self.pack_ready(check_hash):
            warn('pack already available')
            return

        packs = self.packs or self.sources  # use sources if packs None
        for pack in packs:
            pack_uri = os.path.join(self.cloud_root_abspath, pack.path)
            load_from_cloud(pack_uri, pack.abspath)
            assert pack.ready(check_hash)

    def upload_pack(self, check_hash=True):  # TODO override?
        if not self.pack_ready(check_hash):
            raise RuntimeError('TODO')
        packs = self.packs or self.sources  # use sources if packs None
        for pack in packs:
            pack_uri = os.path.join(self.cloud_root_abspath, pack.path)
            # TODO check if already exists?
            save_to_cloud(pack.abspath, pack_uri)
            assert pack.ready(check_hash)

    def load_data(self, require=True, check_hash=True, **kwargs):
        __doc__ = self._load_data.__doc__
        if require:
            self.require(check_hash)

        return self._load_data(path=self.root_abspath, **kwargs)

    def get_abspath(self, relpath):
        return os.path.join(self.root_abspath, relpath)

    # OVERRIDE THESE
    def post_process(self):
        """Implement for additional logic"""
        return

    def _load_data(self, path, **kwargs):
        raise NotImplementedError()


class DatasetBase(object):
    """

    What problems are we solving:
    - Reproducable and transparent
        - transformations and compositions of available data
        - all step declarative or implemented in python
    - Safe
        - all steps hashed
    - Organized
    - Dependencies/versions
        - don't duplicate when not needed
    - Seamless local and cloud dev
        - optimized packs in cloud
    - Extendable


    FETCH SOURCE -> (H) -> BUILD -> (H) -> (ready)

    (ready) -> (H) -> PACK -> (H) -> UPLOAD PACK -> (cloud ready)

    (cloud ready) -> FETCH PACK -> (H) -> UNPACK -> (H) -> (ready)


    REUIRE <- (H) <- (ready) <- UNPACK <- (H) <- FETCH PACK
                        \
                         <- BUILD <- (H) <- FETCH SOURCE


    """

    PACK_USE_SOURCES = 'use_sources'  # TODO deprecate - if none fallback on sources
    PACK_ARCHIVE_BUILDS = 'archive_builds'
    DEFAULT_PACK_METHODS = [PACK_USE_SOURCES, PACK_ARCHIVE_BUILDS]

    root = None
    sources = None
    builds = None
    pack_method = None
    packs = None

    _abs_root = None

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def abspath(cls, relpath):
        if cls._abs_root is None:
            abs_dataset_root = os.path.abspath(
                os.path.join(
                    CONFIG.home,
                    cls.root
                )
            )
        else:
            abs_dataset_root = cls._abs_root

        return os.path.join(abs_dataset_root, relpath)

    @classmethod
    def fetch_sources(cls):
        for source in cls.sources:
            target_name = source.get('target', os.path.basename(source['url']))
            target_path = cls.abspath(target_name)
            download(source['url'], target_path)

    @classmethod
    def try_fetch_sources(cls):
        try:
            cls.fetch_sources()
            return True
        except Exception as e:  # TODO narrow down
            print('failed to fetch sources: {}'.format(e))
            return False

    @classmethod
    def sources_fetched(cls, check_hash=False):
        for source in cls.sources:
            if not cls._verify_target(source, check_hash):
                return False
        return True

    @classmethod
    def _verify_target(cls, obj, check_hash):
        target_relpath = obj.get(
            'target',
            os.path.basename(obj['url']) if 'url' in obj else None
        )

        if target_relpath is None:
            raise ValueError('No target for obj: {}'.format(obj))

        target_abspath = cls.abspath(target_relpath)
        if not os.path.exists(target_abspath):
            return False

        if check_hash:
            hash_config = obj.get('hash', {'value': None, 'alg': 'auto'})
            if hash_config.get('value', None) is None:
                warn('no hash provided for obj: {}'.format(obj))
                return True

            return validate_file(
                fpath=target_abspath,
                file_hash=hash_config['value'],
                algorithm=hash_config.get('algorithm')
            )

        return True

    @classmethod
    def _print_hash(cls, obj):
        target_relpath = obj.get(
            'target',
            os.path.basename(obj['url']) if 'url' in obj else None
        )

        if target_relpath is None:
            raise ValueError('No target for obj: {}'.format(obj))

        target_abspath = cls.abspath(target_relpath)
        if not os.path.exists(target_abspath):
            raise RuntimeError('missing object: {}'.format(obj))
        algorithm = obj.get('hash', {'algorithm': 'sha256'}).get('algorithm', 'sha256')
        file_hash = hash_file(target_abspath, algorithm=algorithm)
        print('{} ({}): {}'.format(target_relpath, algorithm, file_hash))


    @classmethod
    def assert_sources_fetched(cls, check_hash=False):
        if not cls.sources_fetched(check_hash):
            raise AssertionError('sources could not be verified')

    @classmethod
    def build(cls):
        """Default is to extract sources, run post_process"""
        for source in cls.sources:
            target_name = source.get('target', os.path.basename(source['url']))
            target_path = cls.abspath(target_name)
            extract_archive(
                target_path,
                path=None,
                archive_format=source.get('extract', 'auto')
            )
        cls.post_process()

    @classmethod
    def post_process(cls):
        """Implement for additional logic"""
        return

    @classmethod
    def is_built(cls, check_hash=False):
        for build in cls.builds:
            if not cls._verify_target(build, check_hash):
                return False
        return True

    @classmethod
    def assert_built(cls, check_hash=False):
        if not cls.is_built(check_hash):
            raise AssertionError('build could not be verified')

    @classmethod
    def list_build_hashes(cls):
        print('---- Build targets hashes: ----')
        for build in cls.builds:
            cls._print_hash(build)

    @classmethod
    def list_source_hashes(cls):
        print('---- Source targets hashes: ----')
        for source in cls.sources:
            cls._print_hash(source)


    # TODO simplify PACK methods by infering "packs" once based on method if not
    # specified.

    @classmethod
    def pack(cls):
        if cls.pack_method == cls.PACK_USE_SOURCES:
            if not cls.sources_fetched():
                cls.fetch_sources()
            return

        if cls.pack_method == cls.PACK_ARCHIVE_BUILDS:
            # TODO make simple archive of builds (jointly or separate optional?)
            raise NotImplementedError('')

        else:
            raise NotImplementedError(
                '`pack_method` must be one of {}, or override the `pack` class '
                'method'.format(cls.DEFAULT_PACK_METHODS)
            )

    @classmethod
    def is_packed(cls, check_hash=False):
        if cls.pack_method == cls.PACK_USE_SOURCES:
            return cls.sources_fetched(check_hash)

        if cls.packs is not None:
            for pack in cls.packs:
                if not cls._verify_target(pack, check_hash):
                    return False
            return True

        if cls.pack_method == cls.PACK_ARCHIVE_BUILDS:
            # TODO infer target names if not given, check hashes if specified
            raise NotImplementedError('')

        else:
            raise NotImplementedError(
                '`pack_method` must be one of {}, or if `pack` is '
                'overloaded you must also implement the `is_packed` '
                'method'.format(cls.DEFAULT_PACK_METHODS)
            )

    @classmethod
    def assert_packed(cls, check_hash=False):
        if not cls.is_packed(check_hash):
            raise AssertionError('could not verify packed')

    @classmethod
    def cloud_uri(cls, relpath):
        return os.path.join(
            CONFIG.cloud.home,
            cls.root,
            relpath
        )

    @classmethod
    def fetch_pack(cls, dataset_root_uri=None, check_hash=True):
        if cls.is_packed(check_hash):
            warn('pack already available')
            return

        if cls.pack_method == cls.PACK_USE_SOURCES:
            for source in cls.sources:
                target_relpath = source.get('target', os.path.basename(source['url']))
                target_abspath = cls.abspath(target_relpath)
                if dataset_root_uri is None:
                    source_uri = cls.cloud_uri(target_relpath)
                else:
                    source_uri = os.path.join(dataset_root_uri, target_relpath)
                load_from_cloud(source_uri, target_abspath)
            cls.assert_sources_fetched(check_hash)
            return

        if cls.packs is not None:
            for pack in cls.packs:
                target_relpath = pack['target']
                target_abspath = cls.abspath(target_relpath)
                if dataset_root_uri is None:
                    source_uri = cls.cloud_uri(target_relpath)
                else:
                    source_uri = os.path.join(dataset_root_uri, target_relpath)
                load_from_cloud(source_uri, target_abspath)
            return



        if cls.pack_method == cls.PACK_ARCHIVE_BUILDS:
            # TODO infer target names if not given, load from cloud.
            raise NotImplementedError('')

    @classmethod
    def try_fetch_pack(cls):
        try:
            cls.fetch_pack()
            return True
        except CloudIOError:
            return False

    @classmethod
    def unpack(cls):
        if cls.pack_method == cls.PACK_USE_SOURCES:
            cls.build()
            return

        if cls.pack_method == cls.PACK_ARCHIVE_BUILDS:
            # TODO infer target names if not given, run default extract
            raise NotImplementedError('')

    @classmethod
    def upload_pack(cls, dataset_root_uri=None, check_hash=True):
        if cls.pack_method == cls.PACK_USE_SOURCES:
            for source in cls.sources:
                target_relapth = source.get('target', os.path.basename(source['url']))
                target_abspath = cls.abspath(target_relapth)
                if dataset_root_uri is None:
                    target_uri = cls.cloud_uri(target_relapth)
                else:
                    target_uri = os.path.join(dataset_root_uri, target_relapth)
                save_to_cloud(source_path=target_abspath, target_uri=target_uri)
            return

        if cls.packs is not None:
            if not cls.is_packed(check_hash):
                cls.pack()
                cls.assert_packed(check_hash)
            for pack in cls.packs:
                target_relapth = pack['target']
                target_abspath = cls.abspath(target_relapth)
                if dataset_root_uri is None:
                    target_uri = cls.cloud_uri(target_relapth)
                else:
                    target_uri = os.path.join(dataset_root_uri, target_relapth)
                save_to_cloud(source_path=target_abspath, target_uri=target_uri)
            return

        if cls.pack_method == cls.PACK_ARCHIVE_BUILDS:
            # TODO infer target names if not given, load from cloud.
            raise NotImplementedError('')

        raise NotImplementedError('')

    @classmethod
    def require(cls, check_hash=True):
        if cls.is_built(check_hash):
            # TODO force pack to check hash?
            return

        print('checking if packed')
        if cls.is_packed(check_hash):
            print('found packed, unpacking')
            cls.unpack()
            cls.assert_built(check_hash)
            return

        mkdirp(cls.abspath('.'))

        print('checking if available from cloud')
        if cls.try_fetch_pack():
            print('was fetched, verifying')
            cls.assert_packed(check_hash)
            cls.unpack()
            cls.assert_built(check_hash)
            return

        print('check if sources exists')
        if cls.sources_fetched(check_hash):
            print('sources, exists, building...')
            cls.build()
            cls.assert_built(check_hash)
            return

        print('trying to fetch sources')
        if cls.try_fetch_sources():
            cls.assert_sources_fetched(check_hash)
            cls.build()
            cls.assert_built(check_hash)
            return

        raise DatasetNotAvailable()

    @classmethod
    @contextmanager
    def custom_abs_root(cls, path):
        cls._abs_root = path
        yield
        cls._abs_root = None

    @classmethod
    def cmdline(cls):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(help='sub-command help')

        # REQUIRE
        def require(args):
            with cls.custom_abs_root(args.target_path):
                cls.require(check_hash=args.check_hash)

        require_parser = subparsers.add_parser(
            'require',
            help='fetch {} dataset if not already available'.format(cls.name())
        )
        require_parser.add_argument(
            '--target-path',
            type=str,
            default=cls.abspath('.'),
            help='local path for storing dataset'
        )
        require_parser.add_argument(
            '--check-hash',
            type=bool,
            default=True,
            help='Verify hash of data'
        )
        require_parser.set_defaults(func=require)

        # UPLOAD
        def upload(args):
            with cls.custom_abs_root(args.source_path):
                if cls.is_packed(args.check_hash):
                    print('Found packed version, uploading...')
                    cls.upload_pack(dataset_root_uri=args.target_uri)
                    print('Upload successful.')
                    return

                if cls.is_built():
                    print('Found no packed, but built dataset, packing...')
                    cls.pack()
                    cls.assert_packed(args.check_hash)
                    print('Done packing, uploading...')
                    cls.upload_pack(dataset_root_uri=args.target_uri)
                    print('Upload successful.')
                    return

                print(
                    'Could not upload dataset, there is no built or '
                    'packed version available at: {}, run `require` '
                    'method first.'.format(args.source_path)
                )

        upload_parser = subparsers.add_parser(
            'upload',
            help='upload {} dataset to cloud storage'.format(cls.name())
        )
        upload_parser.add_argument(
            '--target-uri',
            type=str,
            default=os.path.join(CONFIG.cloud.home, cls.root),
            help='target URI for storing dataset'
        )
        upload_parser.add_argument(
            '--source-path',
            type=str,
            default=os.path.join(CONFIG.home, cls.root),
            help='local source path for to upload from storing dataset'
        )
        upload_parser.add_argument(
            '--check-hash',
            type=bool,
            default=True,
            help='Verify hash of packed before upload data'
        )
        upload_parser.set_defaults(func=upload)

        # LIST HASH
        def list_hash(args):
            if args.phase == 'build':
                cls.list_build_hashes()
            elif args.phase == 'source':
                cls.list_source_hashes()
            else:
                raise NotImplementedError('')

        hash_parser = subparsers.add_parser(
            'hash',
            help='list hashes'.format(cls.name())
        )
        hash_parser.add_argument(
            '--phase',
            type=str,
            default='build',
            help='one of [source, build, pack]'
        )
        hash_parser.set_defaults(func=list_hash)

        args = parser.parse_args()
        return args.func(args)

    @classmethod
    def load_data(cls, path=None, require=True, check_hash=True, **kwargs):
        __doc__ = cls._load_data.__doc__
        if require:
            if path is not None:
                with cls.custom_abs_root(path):
                    cls.require(check_hash)
            else:
                cls.require(check_hash)

        if path is not None:
            abs_root = path
        else:
            abs_root = cls.abspath('.')
        return cls._load_data(path=abs_root, **kwargs)

    @classmethod
    def _load_data(cls, path, **kwargs):
        raise NotImplementedError()
