from __future__ import print_function, division

import os
import shutil
import tempfile
from contextlib import contextmanager
from functools import wraps
from io import BytesIO

from mock import Mock, patch

from six.moves.urllib.error import URLError

from ml_tools.dataset.cloud import CloudIOError
from ml_tools.dataset.config import CONFIG


@contextmanager
def temp_filename(**kwargs):
    """Context that returns a temporary filename and deletes the file on exit if
    it still exists (so that this is not forgotten).
    """
    _, temp_fname = tempfile.mkstemp(**kwargs)
    yield temp_fname
    if os.path.exists(temp_fname):
        os.remove(temp_fname)


@contextmanager
def temp_dir(**kwargs):
    temp_dirname = tempfile.mkdtemp(**kwargs)
    yield temp_dirname
    if os.path.exists(temp_dirname):
        shutil.rmtree(temp_dirname)


def in_temp_dir(test_func):
    @wraps(test_func)
    def test_wrapper(self=None):
        with temp_dir() as _temp_dir:
            if self is not None:
                return test_func(self, _temp_dir)
            else:
                return test_func(_temp_dir)

    return test_wrapper


class TempDirMixin(object):
    """Mixin for nose test class"""
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

    def tear_down(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def get_temp_path(self, relpath):
        return os.path.join(self.temp_dir, relpath)


class mocked_url(object):

    def __init__(
        self,
        url_to_object=None,
        url_module='ml_tools.dataset.target.url'
    ):
        self.url_to_object = url_to_object or {}
        self.url_module = url_module

        self.patched_url_module = None
        self._is_started = False

    def download(self, url, target):
        if url not in self.url_to_object:
            raise URLError('{} does not exist'.format(url))
        obj = self.url_to_object[url]
        with open(target, 'w') as target_file:
            target_file.write(obj)
        # else:
        #     with open(target, 'wb') as target_file:
        #         with open(obj, 'rb') as source_file:
        #             target_file.write(source_file.read())

    def start(self):
        """Start mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if self._is_started:
            raise RuntimeError('start called on already started mocked_url')

        mock_module = Mock()
        mock_module.download = self.download
        self.patched_url_module = patch(self.url_module, new=mock_module)
        self.patched_url_module.start()
        self._is_started = True

    def stop(self):
        """Stop mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if not self._is_started:
            raise RuntimeError('stop called on unstarted tf_file_io_proxy')
        self.patched_url_module.stop()
        self._is_started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class mocked_cloud(object):
    def __init__(
        self,
        uri_to_object=None,
        cloud_module='ml_tools.dataset.base.cloud'
    ):
        self.uri_to_object = uri_to_object or {}
        self.cloud_module = cloud_module

        self.patched_cloud_module = None
        self._is_started = False

    def save_to_cloud(self, source_path, target_uri):
        target_stream = BytesIO()
        with open(source_path, 'rb') as f:
            target_stream.write(f.read())
        self.uri_to_object[target_uri] = target_stream

    def load_from_cloud(self, source_uri, target_path):
        if source_uri not in self.uri_to_object:
            raise CloudIOError('{} does not exist')
        self.uri_to_object[source_uri].seek(0)
        with open(target_path, 'w') as f:
            f.write(self.uri_to_object[source_uri].read())

    def start(self):
        """Start mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if self._is_started:
            raise RuntimeError('start called on already started mocked_url')

        mock_module = Mock()
        mock_module.save_to_cloud = self.save_to_cloud
        mock_module.load_from_cloud = self.load_from_cloud
        self.patched_cloud_module = patch(self.cloud_module, new=mock_module)
        self.patched_cloud_module.start()
        self._is_started = True

    def stop(self):
        """Stop mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if not self._is_started:
            raise RuntimeError('stop called on unstarted tf_file_io_proxy')
        self.patched_cloud_module.stop()
        self._is_started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class test_env(object):

    def __init__(self, url=None, cloud=None):
        self.url = url or mocked_url()
        self.cloud = cloud or mocked_cloud()
        self.dataset_home = None
        self._org_dataset_home = None
        self._org_cloud_home = None

    def __enter__(self):
        self.url.start()
        self.cloud.start()
        self.dataset_home = tempfile.mkdtemp()
        self._org_dataset_home = CONFIG.home
        CONFIG.home = self.dataset_home
        self._org_cloud_home = CONFIG.cloud.home
        CONFIG.cloud.home = 'gs://cloud_home_test'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cloud.stop()
        self.url.stop()
        CONFIG.home = self._org_dataset_home
        self._org_dataset_home = None
        CONFIG.cloud.home = self._org_cloud_home
        self._org_cloud_home = None
        if os.path.exists(self.dataset_home):
            shutil.rmtree(self.dataset_home)
            self.dataset_home = None

    def abspath_to(self, *relpath):
        return os.path.join(self.dataset_home, *relpath)

    def assert_exists(self, relpath):
        if not os.path.exists(self.abspath_to(relpath)):
            raise AssertionError('{} does not exist'.format(relpath))

    def assert_in_cloud(self, relpath):
        if os.path.join(CONFIG.cloud.home, relpath) not in self.cloud.uri_to_object:
            raise AssertionError('{} does not exist'.format(relpath))
