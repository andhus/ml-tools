from __future__ import print_function, division

import os

from ml_tools.dataset.hash import HashReference
from ml_tools.dataset.test_utils import mocked_url, in_temp_dir

from nose.tools import assert_equal, assert_raises


class TestLocalTarget(object):
    from ml_tools.dataset.target import LocalTarget as TestClass

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


class TestPack(object):
    from ml_tools.dataset.target import Pack as TestClass

    def test_init(self):
        pack = self.TestClass(
            path='test.pack.tgz',
            dataset_root='/dataset_root',
            build_paths=['thing.txt']
        )

    def test_from_config(self):
        # full spec
        pack = self.TestClass.from_config(
            config={
                'path': 'test.pack.tgz',
                'build_paths': ['test.txt']
            },
            dataset_root='/dataset_root'
        )
        assert_equal(pack.path, 'test.pack.tgz')
        assert_equal(pack.build_paths, ['test.txt'])

        # path inferred
        pack = self.TestClass.from_config(
            config={'build_paths': ['test.txt']},
            dataset_root='/dataset_root'
        )
        assert_equal(pack.path, 'test.txt.pack.tgz')
        assert_equal(pack.build_paths, ['test.txt'])

        # single build path(s)
        pack = self.TestClass.from_config(
            config={'build_paths': 'test.txt'},
            dataset_root='/dataset_root'
        )
        assert_equal(pack.path, 'test.txt.pack.tgz')
        assert_equal(pack.build_paths, ['test.txt'])

    @in_temp_dir
    def test_pack_unpack(self, dataset_root):
        pack = self.TestClass(
            path='test.pack.tgz',
            dataset_root=dataset_root,
            build_paths=['test.txt']
        )
        build_abspath = os.path.join(dataset_root, 'test.txt')
        pack_abspath = os.path.join(dataset_root, pack.path)

        with open(build_abspath, 'w') as f:
            f.write('test')

        pack.pack()
        assert os.path.exists(pack_abspath)
        os.remove(build_abspath)
        assert not os.path.exists(build_abspath)
        pack.unpack()
        assert os.path.exists(build_abspath)
        with open(build_abspath) as f:
            assert_equal(f.read(), 'test')

    @in_temp_dir
    def test_pack_unpack_nested(self, dataset_root):
        pack = self.TestClass(
            path='test.pack.tgz',
            dataset_root=dataset_root,
            build_paths=[
                'dir1/test1.txt',
                'dir1/dir2/test2.txt'
            ]
        )
        build_abspaths = [
            os.path.join(dataset_root, build_path)
            for build_path in pack.build_paths
        ]
        pack_abspath = os.path.join(dataset_root, pack.path)

        os.makedirs(os.path.join(dataset_root, 'dir1', 'dir2'))
        with open(build_abspaths[0], 'w') as f:
            f.write('test1')
        with open(build_abspaths[1], 'w') as f:
            f.write('test2')

        pack.pack()
        assert os.path.exists(pack_abspath)
        for build_abspath in build_abspaths:
            os.remove(build_abspath)
            assert not os.path.exists(build_abspath)
        pack.unpack()
        for build_abspath in build_abspaths:
            assert os.path.exists(build_abspath)
        with open(build_abspaths[0]) as f:
            assert_equal(f.read(), 'test1')
        with open(build_abspaths[1]) as f:
            assert_equal(f.read(), 'test2')

    @in_temp_dir
    def test_pack_unpack_nested_merged(self, dataset_root):
        pack = self.TestClass(
            path='test.pack.tgz',
            dataset_root=dataset_root,
            build_paths=['dir1/test.txt']
        )
        build_abspath = os.path.join(dataset_root, 'dir1', 'test.txt')
        other_file = os.path.join(dataset_root, 'dir1', 'other.txt')

        pack_abspath = os.path.join(dataset_root, pack.path)

        os.mkdir(os.path.join(dataset_root, 'dir1'))
        with open(build_abspath, 'w') as f:
            f.write('test')
        with open(other_file, 'w') as f:
            f.write('other text')

        pack.pack()
        assert os.path.exists(pack_abspath)
        os.remove(build_abspath)
        assert not os.path.exists(build_abspath)
        pack.unpack()
        assert os.path.exists(build_abspath)
        with open(build_abspath) as f:
            assert_equal(f.read(), 'test')

        assert os.path.exists(other_file)
        with open(other_file) as f:
            assert_equal(f.read(), 'other text')
