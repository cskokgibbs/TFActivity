from distutils.core import setup

setup(
    name='TFActivity',
    version='0.1.0',
    author='Claudia Skok Gibbs',
    author_email='cskokgibbs@flatironinstitute.org',
    packages=['TFActivity', 'TFActivity.test'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Transcription Factor Activity Calculations',
    long_description=open('README.txt').read(),
)