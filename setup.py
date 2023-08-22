#!/usr/bin/env python

from distutils.core import setup

setup(name='GoTube',
      version='0.1',
      description='Scalable stochastic verification of continuous-depth models',
      author='Sophie Neubauer',
      author_email='sophie@datenvorsprung.at',
      url='https://github.com/DatenVorsprung/GoTube',
      packages=['gotube'],
      install_requires=['jax', 'jaxlib']
      )
