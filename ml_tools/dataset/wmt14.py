from __future__ import print_function, division

import os
import shutil
import tarfile

from ml_tools.dataset import DatasetBase
from ml_tools.dataset.fetch_utils import extract_archive
from ml_tools.pytils.file import mkdirp


class WMT14Dev(DatasetBase):
    root = 'text/wmt14/dev'

    languages = ['cs', 'de', 'en', 'es', 'fr', 'ru']
    filenames = ['newstest2013.{}'.format(lang) for lang in languages]
    language_to_filename = dict(zip(languages, filenames))

    sources = [
        {'url': 'http://www.statmt.org/wmt14/dev.tgz',
         'hash': {
            'value': 'cda0f85309e8ea4c9c2bc142cd795fb2771a5939ed5b4527b525dae05fa0c145',
            'algorithm': 'sha256'}}]

    builds = [{'target': fn} for fn in language_to_filename.values()]
    packs = [
        {'target': 'wmt14-dev.pack.tgz'}
    ]

    @classmethod
    def build(cls):
        [source] = cls.sources
        target_name = source.get('target', os.path.basename(source['url']))
        target_path = cls.abspath(target_name)
        tmp_dir = cls.abspath('tmp')
        mkdirp(tmp_dir)
        extract_archive(
            target_path,
            path=tmp_dir,
            archive_format=source.get('extract', 'auto')
        )
        for build in cls.builds:
            filename = build['target']
            shutil.move(
                os.path.join(tmp_dir, 'dev', filename), cls.abspath(filename)
            )
        shutil.rmtree(tmp_dir)

    @classmethod
    def pack(cls):
        pack_dir = cls.abspath('wmt14-dev.pack')
        mkdirp(pack_dir)
        for build in cls.builds:
            filename = build['target']
            shutil.copy(cls.abspath(filename), os.path.join(pack_dir, filename))

        [pack] = cls.packs
        with tarfile.open(cls.abspath(pack['target']), "w:gz") as tar:
            tar.add(pack_dir, arcname=os.path.basename(pack_dir))
        shutil.rmtree(pack_dir)

    @classmethod
    def unpack(cls):
        [pack] = cls.packs
        pack_dir = cls.abspath('wmt14-dev.pack')
        extract_archive(
            cls.abspath(pack['target']),
            path=None,  # same dir as file
            archive_format=pack.get('extract', 'auto')
        )
        for build in cls.builds:
            filename = build['target']
            shutil.move(os.path.join(pack_dir, filename), cls.abspath(filename))
        shutil.rmtree(pack_dir)

    @classmethod
    def _load_data(cls, path):
        data = {}
        for language, filename in sorted(cls.language_to_filename.items()):
            with open(os.path.join(path, filename), 'r') as f:
                data[language] = f.readlines()
        return data


if __name__ == '__main__':
    WMT14Dev.cmdline()
