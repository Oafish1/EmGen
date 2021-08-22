from setuptools import find_packages, setup

with open('emgen/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='emgen',
    description=readme,
    version=__version__,
    packages=find_packages(include=('emgen')),
    test_suite='tests',
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        'pytorch-lightning',
        'sklearn',
        'torch'
        ],
    tests_require=['pytest']
)
