from setuptools import find_packages
from setuptools import setup

setup(
    name='us_msg',
    version='0.0.0',
    packages=find_packages(
        include=('us_msg', 'us_msg.*')),
)
