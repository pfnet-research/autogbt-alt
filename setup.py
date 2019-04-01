import re
import os
from pathlib import Path
from setuptools import setup, find_packages

base_dir = Path(__file__).parent
os.chdir(str(base_dir))
with open('autogbt/__init__.py') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)
with open('./requirements.txt') as fp:
    install_requires = list(map(lambda d: d.strip(), fp.readlines()))
with open('./requirements-dev.txt') as fp:
    dev_requires = list(map(lambda d: d.strip(), fp.readlines()))

setup(
    name='autogbt',
    version=version,
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=install_requires,
    extras_requires={
        'dev': dev_requires,
    },
)
