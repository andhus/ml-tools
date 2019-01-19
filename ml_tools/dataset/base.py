from __future__ import print_function, division

import os
import abc
from warnings import warn

from ml_tools.dataset import cloud
from ml_tools.dataset.archive import extract_archive
from ml_tools.dataset.config import CONFIG
from ml_tools.dataset.target import parse_source, parse_target, parse_pack


class DatasetError(Exception):
    pass


class DatasetNotAvailable(DatasetError):
    pass


class DatasetBase(object):
    """
    Arguments:
        root_abspath: override path/to/dataset_root/
        dataset_home: override path/to/dataset_home/ should not be specified if
            root_abspath is given

    Requires self.config to exist, either as class property or build in init of
    extending class before calling super...
    """
    __metaclass__ = abc.ABCMeta

    config = None

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

    def post_process(self):
        """Implement for additional logic"""
        return

    @abc.abstractmethod
    def _load(self, path, **kwargs):
        raise NotImplementedError()

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
            print('sources exists, building...')
            self.build()
            assert self.build_ready(check_hash)
            return

        print('trying to fetch sources')
        self.fetch_sources()
        self.sources_ready(check_hash)
        self.build()
        assert self.build_ready(check_hash)

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
            if source.extract:  # skip if `None` or `False`
                extract_archive(
                    source.abspath,
                    path=self.root_abspath,
                    archive_format=source.extract
                )
        self.post_process()

    def pack(self):
        if self.packs is None:
            # using sources
            self.require_sources()
        else:
            for pack in self.packs:
                pack.pack()

    def pack_ready(self, check_hash=False):
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

    def fetch_pack(self, check_hash=True, force=False):
        packs = self.packs or self.sources  # use sources if packs None
        for pack in packs:
            if pack.ready(check_hash) and not force:
                warn('pack already available: {}'.format(pack))
                continue
            pack_uri = os.path.join(self.cloud_root_abspath, pack.path)
            cloud.load_from_cloud(pack_uri, pack.abspath)
            assert pack.ready(check_hash)

    def upload_pack(self, check_hash=True):  # TODO override?
        if not self.pack_ready(check_hash):
            raise RuntimeError('TODO')
        packs = self.packs or self.sources  # use sources if packs None
        for pack in packs:
            pack_uri = os.path.join(self.cloud_root_abspath, pack.path)
            # TODO check if already exists?
            cloud.save_to_cloud(pack.abspath, pack_uri)
            assert pack.ready(check_hash)

    def load(self, require=True, check_hash=True, **kwargs):
        __doc__ = self._load.__doc__
        if require:
            self.require(check_hash)
        else:
            if not self.build_ready(check_hash):
                raise DatasetNotAvailable(
                    'Dataset {} is not available, set `require=True` to '
                    'automatically fetch data'.format(self.__class__.__name__)
                )

        return self._load(**kwargs)

    def abspath_to(self, relpath):
        return os.path.join(self.root_abspath, relpath)

    def list_hashes(self):
        self.list_source_hashes()
        self.list_build_hashes()
        self.list_pack_hashes()

    def list_source_hashes(self):
        print('---- Source targets hashes: ----')
        for source in self.sources:
            source.print_hash()

    def list_build_hashes(self):
        print('---- Build targets hashes: ----')
        for build in self.builds:
            build.print_hash()

    def list_pack_hashes(self):
        print('---- Pack targets hashes: ----')
        if self.packs is None:
            print('(using sources)')
        else:
            for pack in self.packs:
                pack.print_hash()
