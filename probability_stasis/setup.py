from setuptools import setup, find_packages

setup(
    name='probability_stasis',
    version='0.1.0',
    description='A library for filtering and stabilizing probability predictions',
    author='iChrispincoder',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)