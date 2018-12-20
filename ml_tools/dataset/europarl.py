from __future__ import print_function, division

import os

from ml_tools.dataset import DatasetBase


class EuroParlV7FrEn(DatasetBase):
    language_to_filename = {
        'fr': 'europarl-v7.fr-en.fr',
        'en': 'europarl-v7.fr-en.en'
    }
    root = 'text/europarl/v7'
    sources = [
        {'url': 'http://www.statmt.org/europarl/v7/fr-en.tgz',
         'hash': {
             'value': '80f5d52113e23bfe51f890569ed69d2a1348127bca0894d07d1d23e0d0fac1e9',
             'algorithm': 'sha256'}}]
    builds = [
        {'target': language_to_filename['fr'],
         'hash': {
             'value': 'dfc2a6e92df0922241a12f2e33c4d1c74689958db85985a2914b34e774eee653',
             'algorithm': 'sha256'}},
        {'target': language_to_filename['en'],
         'hash': {
             'value': '89fb4a28d1a7b97c8a77a8f94691db3d09abbfe131ce8f9e98398a4b943bacb2',
             'algorithm': 'sha256'}}
    ]
    pack_method = DatasetBase.PACK_USE_SOURCES
    packs = None

    @classmethod
    def _load_data(cls, path):
        data = {}
        for language, filename in sorted(cls.language_to_filename.items()):
            with open(os.path.join(path, filename), 'r') as f:
                data[language] = f.readlines()

        return data


if __name__ == '__main__':
    EuroParlV7FrEn.cmdline()
