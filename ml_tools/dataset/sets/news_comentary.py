from __future__ import print_function, division

import os
import shutil

from ml_tools.dataset.base import DatasetBase
from ml_tools.dataset.fetch_utils import extract_archive
from ml_tools.pytils.file import mkdirp


class NewsCommentaryV9FrEn(DatasetBase):
    language_to_filename = {
        'fr': 'news-commentary-v9.fr-en.fr',
        'en': 'news-commentary-v9.fr-en.en'
    }
    config = dict(
        root='text/news-commentary/v9_refactor',
        sources=[
            {'url': 'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
             'hash': 'cb8953f292298e6877ae433c98912b927cb0766b303f4540512ddd286c748485'}
        ],
        builds=[
            {'path': language_to_filename['fr'],
             'hash': '02bcfae333730635a5007def8affe228904938330801db35d3c12d9e915fb3e4'},
            {'path': language_to_filename['en'],
             'hash': '7d171abc43dafa572a95754e5fd680d3e57088eb0c4697c19c29483f83b11754'}
        ],
        packs=[
            {'path': 'training-parallel-nc-v9-fr-en.pack.tgz',
             'build_paths': language_to_filename.values()}
        ]
    )

    def build(self):
        [source] = self.sources
        tmp_dir = self.get_abspath('tmp')
        mkdirp(tmp_dir)
        extract_archive(source.abspath, path=tmp_dir, archive_format=source.extract)
        for build in self.builds:
            shutil.move(os.path.join(tmp_dir, 'training', build.path), build.abspath)
        shutil.rmtree(tmp_dir)

    def _load_data(self, path):
        data = {}
        for language, filename in sorted(self.language_to_filename.items()):
            with open(os.path.join(path, filename), 'r') as f:
                data[language] = f.readlines()
        return data


if __name__ == '__main__':
    dataset = NewsCommentaryV9FrEn()
    # NewsCommentaryV9FrEn.cmdline()
