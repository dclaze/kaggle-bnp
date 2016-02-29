from setuptools import setup

setup(name='kaggle_bnp',
      version='0.1',
      description='kaggle submission for [BNP Paribas Cardif Claims Management](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)',
      url='https://github.com/dclaze/kaggle-bnp',
      author='Doug Colaizzo',
      author_email='dcolaizzo@gmail.com',
      license='MIT',
      packages=['kaggle_bnp'],
      entry_points = {
            'console_scripts': ['kaggle-bnp=kaggle_bnp.command_line:main'],
      },
      zip_safe=False)