from setuptools import setup

import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('DataFiles')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Blood_Flow_1D",
    version="0.0.1",
    packages=['Blood_Flow_1D'],
    package_data={'': extra_files},
    include_package_data=True,
    install_requires=[
        'scipy',
        'tqdm',
        'mgmetis',
        'matplotlib',
        'pyacvd',
        'h5py',
        'pyvista',
        'numpy',
        'lxml',
        'networkx',
        'pandas',
        'PyQt5',
        'pytest',
        'PyYAML',
        'schema',
        'vtk'],
)
