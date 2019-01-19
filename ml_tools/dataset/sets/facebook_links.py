from __future__ import print_function, division

import os

from ml_tools.dataset import DatasetBase


class LinesIterator(object):
    """Iterates over and optionally post process the lines of a file.

    Args:
        filepath (str): file to iterate over.
        limit (int | None): If provided, only iterate over first `limit` lines.
        postprocess (f: str -> object | None): A function to apply to each line
            before output.
    """
    def __init__(self, filepath, limit=None, postprocess=None):
        self.filepath = filepath
        self.limit = limit
        self.postprocess = postprocess or (lambda x: x)
        self._len = None

    def __iter__(self):
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                if self.limit and i == self.limit:
                    break
                yield self.postprocess(line)
            self._len = i + 1

    def __len__(self):
        if self._len is None:
            with open(self.filepath) as f:
                for i, _ in enumerate(f):
                    pass
            self._len = i + 1

        return self._len


class FacebookLinks(DatasetBase):
    root = 'graph/facebook-links'
    sources = [
        {
            'url': 'http://socialnetworks.mpi-sws.mpg.de/data/facebook-links.txt.gz',
            'hash': {
                'value': '32d149f76c3421a08b03bfc629a9de6ce65b6fa56272cd9d5424e3de5b1acff2',
                'algorithm': 'sha256'
            },
            'target': 'facebook-links.txt.gz'  # could be left empty
        }
    ]
    builds = [
        {
            'target': 'facebook-links.txt',
            'hash': {
                'value': '01f9d8eb285a8da27bdfc3ccbb5ed338ce0a2d60c601e4d2d9b3b91379740196',
                'algorithm': 'sha256'
            }
        }
    ]
    pack_method = DatasetBase.PACK_USE_SOURCES

    @classmethod
    def post_process(cls):
        return

    @classmethod
    def _load(cls, path, limit=None):
        """
        # Arguments
            unique (bool):
            limit (int):

        # Returns
            LineIterator
        """
        post_process_f = lambda line: tuple(
            sorted([int(v) for v in line.split()[:2]])
        )
        filepath = os.path.join(path, cls.builds[0]['target'])
        return LinesIterator(filepath, limit=limit, postprocess=post_process_f)


if __name__ == '__main__':
    args = FacebookLinks.cmdline()
