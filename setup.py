from setuptools import setup, Extension, find_packages
from codecs import open
from os import path
import warnings

package_name = 'lsynth'
example_dir = 'examples/'
data_dir = 'datasets/'

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    author='Ishanu Chattopadhyay',
    author_email='research@paraknowledge.ai',
    version = str(version['__version__']),
    packages=find_packages(),
    package_data={'lsynth': ['assets/*']},
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/lsynth',
    license='LICENSE',
    description='Evaluation of how good a synthetic dataset is compared to the original with presuppossing structural constraints',
    keywords=[
        'machine learning', 
        'statistics'],
    download_url='https://github.com/zeroknowledgediscovery/lsynth/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "scikit-learn", 
        "scipy", 
        "numpy",  
        "pandas",
        "quasinet>=0.1.63",
        "scipy",
        "sdv>=1.8"],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"],
    include_package_data=True,
    )
