from setuptools import setup, find_packages

VERSION = '0.0.2'

setup(
    name='ml-tools',
    version=VERSION,
    description='',
    url='https://github.com/andhus/ml-tools',
    license='MIT',
    install_requires=[],
    extras_require={'gcs': ['google-cloud-storage>=1.13.0']},
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']
    ),
    tests_require=['nose']
)
