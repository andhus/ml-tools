from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'h5py>=2.8.0',
    'Keras>=2.2.4',
]

setup(
    name='ml-engine-machine-translation',
    version='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
