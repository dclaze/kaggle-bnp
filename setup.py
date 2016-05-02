from setuptools import setup

setup(name='kaggle_santander',
      version='0.1',
      description='kaggle submission for santander',
      url='https://github.com/dclaze/kaggle-santander',
      author='Doug Colaizzo',
      author_email='dcolaizzo@gmail.com',
      license='MIT',
      packages=['kaggle_santander'],
      entry_points = {
            'console_scripts': ['kaggle-santander=kaggle_santander.command_line:main'],
      },
      install_requires = ['scikit-learn', 'xgboost', 'numpy'],
      zip_safe=False)