from setuptools import setup, find_packages

setup(
    name='dlfs',
    version='0.1',
    author='daredevil9215',
    install_requires=["numpy", "scipy"],
    packages=find_packages(include=['dlfs', 'dlfs_autograd']),
)