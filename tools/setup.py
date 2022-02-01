#!python
# run "python setup.py install" to install scripts to python path in current environment

from setuptools import setup, find_packages

setup(name='evimo-tools',
      version='1.0',
      description='EV-IMO command line tools',
      author='Levi Burner',
      author_email='lburner@umd.edu',
      url='https://github.com/better-flow/evimo/tools',
      # packages=find_packages(),
      install_requires=[
            'numpy','argparse', 'tqdm', 'easygui', 'scipy'
      ],
      scripts=['evimo_flow.py'],
      # entry_points = {'console_scripts': ['evimo_flow=evimo_flow.py']}
)
