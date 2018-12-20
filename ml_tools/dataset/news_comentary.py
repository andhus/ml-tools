from __future__ import print_function, division

import os
import shutil
import tarfile

from ml_tools.dataset import DatasetBase
from ml_tools.dataset.fetch_utils import extract_archive
from ml_tools.pytils.file import mkdirp


class NewsCommentaryV9FrEn(DatasetBase):
    language_to_filename = {
        'fr': 'news-commentary-v9.fr-en.fr',
        'en': 'news-commentary-v9.fr-en.en'
    }
    root = 'text/news-commentary/v9'
    sources = [
        {'url': 'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
         'hash': {
            'value': 'cb8953f292298e6877ae433c98912b927cb0766b303f4540512ddd286c748485',
            'algorithm': 'sha256'}}]
    builds = [
        {'target': language_to_filename['fr'],
         'hash': {
             'value': '02bcfae333730635a5007def8affe228904938330801db35d3c12d9e915fb3e4',
             'algorithm': 'sha256'}},
        {'target': language_to_filename['en'],
         'hash': {
             'value': '7d171abc43dafa572a95754e5fd680d3e57088eb0c4697c19c29483f83b11754',
             'algorithm': 'sha256'}}
    ]
    packs = [
        {'target': 'training-parallel-nc-v9-fr-en.tgz'}
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
            shutil.move(os.path.join(tmp_dir, 'training', filename), cls.abspath(filename))
        shutil.rmtree(tmp_dir)

    @classmethod
    def pack(cls):
        pack_dir = cls.abspath('fr-en.pack')
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
        pack_dir = cls.abspath('fr-en.pack')
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
    NewsCommentaryV9FrEn.cmdline()
