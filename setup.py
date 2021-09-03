from setuptools import setup

setup(
    name='PY2DFDTD',
    version='0.1.0',
    author='Seth Golembeski',
    author_email='smgole1@outlook.com',
    packages=['FDTD'],
    url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.txt',
    description='2D Elastic Wave FDM',
    long_description=open('readme.md').read(),
    install_requires=[
        "Numpy",
        "Numba",
    ],
)