from setuptools import setup, find_packages

# Read long_description from file
try:
    long_description = open('README.rst', 'r').read()
except FileNotFoundError:
    long_description = ('Please see'
                        ' https://github.com/adamancer/imagery_accessor.git'
                        ' for more information about the imagery_accessor'
                        ' package.')

setup(name='imagery_accessor',
      version='0.1',
      description=("Extends xarray with earthpy plotting methods"),
      long_description=long_description,
      classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
      ],
      url='https://github.com/adamancer/imagery_accessor.git',
      author='adamancer',
      author_email='mansura@si.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      include_package_data=True,
      zip_safe=False)
