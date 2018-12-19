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
    relative_path = os.path.join('graph', 'facebook')
    sources = [
        {
            'url': 'http://socialnetworks.mpi-sws.mpg.de/data/facebook-links.txt.gz',
            # 'hash': {'value': None, 'alg': 'auto'},
            'sub_paths': [
                'facebook-links.txt',
                # 'facebook-links-unique.txt'  # TODO during extraction
            ]
        }
    ]

    @classmethod
    def postprocess(cls):
        pass  # TODO

    @classmethod
    def _load_data(cls, path, unique=False, limit=None):
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
        if unique:
            filepath = os.path.join(path, cls._unique_sub_path)
            if not os.path.exists(filepath):
                org_filepath = os.path.join(path, cls.required_sub_paths[0])
                seen = set([])
                with open(org_filepath) as f_org:
                    with open(filepath, 'w') as f_unique:
                        for line in f_org:
                            res = post_process_f(line)
                            if res not in seen:
                                seen.add(res)
                                f_unique.write(line)
        else:
            filepath = os.path.join(path, cls.required_sub_paths[0])

        return LinesIterator(filepath, limit=limit, postprocess=post_process_f)


if __name__ == '__main__':
    args = FacebookLinks.cmdline()
