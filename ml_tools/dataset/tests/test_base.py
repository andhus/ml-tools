from __future__ import print_function, division

import os

from ml_tools.dataset.test_utils import (
    test_env,
    mocked_url,
    mocked_cloud
)


class TestDatasetBase(object):
    from ml_tools.dataset.base import DatasetBase as TestClass

    def test_flow_basic(self):

        class Dataset(self.TestClass):
            config = {
                'root': 'root',
                'sources': [{'url': 'http://test.txt'}],
                'builds': [{'path': 'test.txt'}],
                'packs': [{'build_paths': ['test.txt']}]
            }

            def post_process(self):
                self.post_process_called = True

            def _load(self, path, **kwargs):
                pass

        with test_env(url=mocked_url({'http://test.txt': 'test'})) as env:
            ds = Dataset()
            ds.fetch_sources()
            env.assert_exists(os.path.join('root', 'test.txt'))
            ds.build()
            ds.pack()
            env.assert_exists(os.path.join('root', 'test.txt.pack.tgz'))
            ds.upload_pack()
            env.assert_in_cloud(os.path.join('root', 'test.txt.pack.tgz'))

# TODO mkdirp in require
