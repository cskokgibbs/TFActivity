from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="TFActivity",
    version='0.1.0',
    description="TFActivity: estimating transcription factor activity with local network component analysis",
    long_description=open('README.md').read(),
    url="https://github.com/cskokgibbs/TFActivity",
    author="Claudia Skok Gibbs",
    author_email="cskokgibbs@flatironinstitute.org",
    maintainer="Claudia Skok Gibbs",
    maintainer_email="cskokgibbs@flatironinstitute.org",
    packages=find_packages(include=["TFActivity", 'TFActivity.test']),
    zip_safe=False
)
