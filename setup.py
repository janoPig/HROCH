from setuptools import setup

setup(
    name='HROCH',
    version='1.0.0',
    description='Python HROCH wrapper',
    author='Jano',
    author_email='jan.pigos@gmail.com',
    url='https://github.com/janoPig/HROCH/',
    packages=['HROCH'],
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=['numpy>=1.22.3'],
)
