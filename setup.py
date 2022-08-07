from setuptools import setup

setup(
    name='HROCH',
    version='1.0.0',
    description='Python HROCH wrapper',
    author='Jano',
    author_email='jan.pigos@gmail.com',
    url='https://github.com/janoPig/HROCH/',
    packages=['HROCH'],
    data_files=[('bin', ['hroch.bin', 'hroch.exe'])],
    install_requires=['numpy>=1.22.3'],
)
