from __future__ import print_function, division

import os

from ml_tools.dataset import DatasetBase


class EuroParlV7FrEn(DatasetBase):
    language_to_filename = {
        'fr': 'europarl-v7.fr-en.fr',
        'en': 'europarl-v7.fr-en.en'
    }
    relative_path = os.path.join('text', 'europarl', 'v7')
    required_sub_paths = sorted(language_to_filename.values())
    sources = ['http://www.statmt.org/europarl/v7/fr-en.tgz']

    @classmethod
    def _load_data(cls, path):
        data = {}
        for language, filename in sorted(cls.language_to_filename.items()):
            with open(os.path.join(path, filename), 'r') as f:
                data[language] = f.readlines()


if __name__ == '__main__':
    EuroParlV7FrEn.cmdline_require()
