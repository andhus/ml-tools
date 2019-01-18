from __future__ import print_function, division

from ..test_utils import TempDirMixin

from nose.tools import assert_equal, assert_raises


class TestHashReference(TempDirMixin):
    from ..hash import HashReference as TestClass

    def test_init(self):
        for alg in self.TestClass.supported_algorithms:
            hash_ref = self.TestClass(value='hash', algorithm=alg)
            assert_equal(hash_ref.algorithm, alg)
            assert_equal(hash_ref.value, 'hash')
        with assert_raises(ValueError):
            self.TestClass(value='hash', algorithm='-')

    def test_from_config(self):
        hash_conf = self.TestClass.from_config('hash')
        assert_equal(hash_conf.algorithm, self.TestClass.default_algorithm)
        assert_equal(hash_conf.value, 'hash')

        hash_conf = self.TestClass.from_config({'value': 'hash'})
        assert_equal(hash_conf.algorithm, self.TestClass.default_algorithm)
        assert_equal(hash_conf.value, 'hash')

        for alg in self.TestClass.supported_algorithms:
            hash_ref = self.TestClass.from_config(
                {'value': 'hash', 'algorithm': alg}
            )
            assert_equal(hash_ref.algorithm, alg)
            assert_equal(hash_ref.value, 'hash')

    def test_validate(self):
        filepath = self.get_temp_path('test.txt')
        with open(filepath, 'w') as f:
            f.write('test')

        for alg, expected_value in [
            ('sha256',
             '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'),
            ('md5',
             '098f6bcd4621d373cade4e832627b4f6')
        ]:
            hash_ref = self.TestClass(value=expected_value, algorithm=alg)
            assert hash_ref.is_valid(filepath)
