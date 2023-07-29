## this file is responsible for creating the machine learning application 
## as a python package and even deploying it to pypi
## then anyone can install it using pip install <package_name>

from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = '-e .'

def get_requirements(filepath:str) -> List[str]:
    '''
    this function is used to get the requirements from the requirements.txt file
    '''
    with open(filepath) as fp:
        requirements = fp.read().splitlines()
        requirements = [requirement for requirement in requirements if not requirement.startswith(HYPHEN_E_DOT)]
    return requirements


setup(
    name='e2emlproject',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='End to end machine learning application using python',
    long_description=open('README.md').read(),
    install_requires=get_requirements('requirements.txt'),
    url='',
    author='Vamsi Duvvuri',
    author_email='vamsi.duvvuri@icloud.com'
)
