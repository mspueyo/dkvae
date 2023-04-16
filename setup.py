import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'beatgpt'
DESCRIPTION = 'AI drum machine.'
URL = 'https://github.com/mspueyo/beatgpt'
EMAIL = 'martinsanchezpueyo@gmail.com'
AUTHOR = 'Martin Sanchez'
REQUIRES_PYTHON = '==3.9.7'
VERSION = '0.0.1'

REQUIRED = ['tensorflow', 'tensorflow-io', 'pandas', 'librosa', 'soundfile', 'numpy']

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
)