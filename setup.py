from setuptools import find_packages, setup


setup(
    name="schedlib",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),    
)
