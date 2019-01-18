from __future__ import print_function, division

import os

from ml_tools.dataset.hash import HashReference
from ml_tools.dataset.test_utils import mocked_url, TempDirMixin, in_temp_dir

from nose.tools import assert_equal, assert_raises


class TestLocalTarget(object):
    from ..target import LocalTarget as TestClass

    def test_init(self):
        default_kwargs = dict(
            path='path/to/file_or_dir',
            dataset_root='/dataset_root'
        )
        # no hash reference
        local_target = self.TestClass(**default_kwargs)
        assert_equal(local_target.hash, None)

        # hash reference from config
        local_target = self.TestClass(
            hash_reference='hash',
            **default_kwargs
        )
        assert_equal(local_target.hash, HashReference.from_config('hash'))

        # instance of hash reference
        hash_ref = HashReference(
            value='hash',
            algorithm=HashReference.default_algorithm
        )
        local_target = self.TestClass(
            hash_reference=hash_ref,
            **default_kwargs
        )
        assert_equal(local_target.hash, hash_ref)

    def test_abspath(self):
        relpath = 'path/to/file_or_dir'
        dataset_root = '/dataset_root'
        local_target = self.TestClass(path=relpath, dataset_root=dataset_root)
        assert_equal(local_target.abspath, os.path.join(dataset_root, relpath))

    @in_temp_dir
    def test_exists(self, temp_dataset_root):
        local_target = self.TestClass(path='file', dataset_root=temp_dataset_root)
        assert not local_target.exists()
        with open(local_target.abspath, 'w') as f:
            f.write('test')
        assert local_target.exists()

    @in_temp_dir
    def test_ready_with_hash(self, temp_dataset_root):
        local_target = self.TestClass(
            path='file',
            dataset_root=temp_dataset_root,
            hash_reference=(
                '9f86d081884c7d659a2feaa0c55ad015'
                'a3bf4f1b2b0b822cd15d6c15b0f00a08'
            )
        )
        assert not local_target.ready()
        with open(local_target.abspath, 'w') as f:
            f.write('test')
        assert local_target.ready(check_hash=True)

        # wrong hash
        local_target = self.TestClass(
            path='file',
            dataset_root=temp_dataset_root,
            hash_reference='hash'
        )
        assert local_target.exists()
        assert local_target.ready(check_hash=False)
        assert not local_target.ready(check_hash=True)

    @in_temp_dir
    def test_ready_no_hash(self, temp_dataset_root):
        local_target = self.TestClass(
            path='file',
            dataset_root=temp_dataset_root
        )
        assert not local_target.ready()
        with open(local_target.abspath, 'w') as f:
            f.write('test')
        assert local_target.ready(check_hash=False)

        with assert_raises(ValueError):
            local_target.ready(check_hash=True)


class TestURLSource():
    from ml_tools.dataset.target import URLSource as TestClass

    def test_init(self):
        url_source = self.TestClass(
            url='http://mock.txt',
            path='target.txt',
            dataset_root='/dataset_root'
        )
        assert_equal(url_source.extract, 'auto')
        url_source = self.TestClass(
            url='http://mock.txt',
            path='target.txt',
            dataset_root='/dataset_root',
            extract=None
        )
        assert_equal(url_source.extract, None)

    def test_from_config(self):
        # path specified
        url_source = self.TestClass.from_config(
            config={'url': 'http://mock.txt', 'path': 'target.txt'},
            dataset_root='/dataset_root'
        )
        assert_equal(url_source.path, 'target.txt')

        # path inferred from url
        url_source = self.TestClass.from_config(
            config={'url': 'http://mock.txt'},
            dataset_root='/dataset_root'
        )
        assert_equal(url_source.path, 'mock.txt')

    @in_temp_dir
    def test_fetch(self, temp_dataset_root):
        url_source = self.TestClass(
            url='http://mock-url.txt',
            path='target.txt',
            dataset_root=temp_dataset_root
        )
        with mocked_url(
            'ml_tools.dataset.target.url',
            {url_source.url: 'mock-data'}
        ):
            url_source.fetch(check_hash=False)

        with open(os.path.join(temp_dataset_root, url_source.path)) as target_file:
            target_data = target_file.read()
        assert_equal(target_data, 'mock-data')


# class TestPack(TempDirMixin):
#
#     def test_ini