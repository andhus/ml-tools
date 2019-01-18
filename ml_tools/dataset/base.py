from __future__ import print_function, division

import os
from warnings import warn

from ml_tools.dataset.cloud import load_from_cloud, save_to_cloud
from ml_tools.dataset.archive import extract_archive
from ml_tools.dataset.config import CONFIG
from ml_tools.dataset.target import parse_source, parse_target, parse_pack


class DatasetBase(object):
    """
    Arguments:
        root_abspath: override path/to/dataset_root/
        dataset_home: override path/to/dataset_home/ should not be specified if
            root_abspath is given

    Requires self.config to exist, either as class property or build in init of
    extending class before calling super...
    """

    # if packs not specified => use sources
    # config = {
    #     'root': 'path/to/dsroot',
    #     'sources': [],
    #     'builds': [],
    #     'packs': [],
    # }
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
                    source.path,
                    path=None,
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

    # OVERRIDE THESE
    def post_process(self):
        """Implement for additional logic"""
        return

    def _load_data(self, path, **kwargs):
        raise NotImplementedError()

    # TODO OLD CMDLINE STUFF
    # @classmethod
    # def cmdline(cls):
    #     parser = argparse.ArgumentParser()
    #     subparsers = parser.add_subparsers(help='sub-command help')
    #
    #     # REQUIRE
    #     def require(args):
    #         with cls.custom_abs_root(args.target_path):
    #             cls.require(check_hash=args.check_hash)
    #
    #     require_parser = subparsers.add_parser(
    #         'require',
    #         help='fetch {} dataset if not already available'.format(cls.name())
    #     )
    #     require_parser.add_argument(
    #         '--target-path',
    #         type=str,
    #         default=cls.abspath('.'),
    #         help='local path for storing dataset'
    #     )
    #     require_parser.add_argument(
    #         '--check-hash',
    #         type=bool,
    #         default=True,
    #         help='Verify hash of data'
    #     )
    #     require_parser.set_defaults(func=require)
    #
    #     # UPLOAD
    #     def upload(args):
    #         with cls.custom_abs_root(args.source_path):
    #             if cls.is_packed(args.check_hash):
    #                 print('Found packed version, uploading...')
    #                 cls.upload_pack(dataset_root_uri=args.target_uri)
    #                 print('Upload successful.')
    #                 return
    #
    #             if cls.is_built():
    #                 print('Found no packed, but built dataset, packing...')
    #                 cls.pack()
    #                 cls.assert_packed(args.check_hash)
    #                 print('Done packing, uploading...')
    #                 cls.upload_pack(dataset_root_uri=args.target_uri)
    #                 print('Upload successful.')
    #                 return
    #
    #             print(
    #                 'Could not upload dataset, there is no built or '
    #                 'packed version available at: {}, run `require` '
    #                 'method first.'.format(args.source_path)
    #             )
    #
    #     upload_parser = subparsers.add_parser(
    #         'upload',
    #         help='upload {} dataset to cloud storage'.format(cls.name())
    #     )
    #     upload_parser.add_argument(
    #         '--target-uri',
    #         type=str,
    #         default=os.path.join(CONFIG.cloud.home, cls.root),
    #         help='target URI for storing dataset'
    #     )
    #     upload_parser.add_argument(
    #         '--source-path',
    #         type=str,
    #         default=os.path.join(CONFIG.home, cls.root),
    #         help='local source path for to upload from storing dataset'
    #     )
    #     upload_parser.add_argument(
    #         '--check-hash',
    #         type=bool,
    #         default=True,
    #         help='Verify hash of packed before upload data'
    #     )
    #     upload_parser.set_defaults(func=upload)
    #
    #     # LIST HASH
    #     def list_hash(args):
    #         if args.phase == 'build':
    #             cls.list_build_hashes()
    #         elif args.phase == 'source':
    #             cls.list_source_hashes()
    #         else:
    #             raise NotImplementedError('')
    #
    #     hash_parser = subparsers.add_parser(
    #         'hash',
    #         help='list hashes'.format(cls.name())
    #     )
    #     hash_parser.add_argument(
    #         '--phase',
    #         type=str,
    #         default='build',
    #         help='one of [source, build, pack]'
    #     )
    #     hash_parser.set_defaults(func=list_hash)
    #
    #     args = parser.parse_args()
    #     return args.func(args)
