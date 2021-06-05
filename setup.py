from setuptools import setup, find_namespace_packages


setup(
    name = 'funda_forecasting',
    version = '0.1',
    description = '',
    packages = find_namespace_packages(where = 'funda_forecasting'),
    package_dir = {'': 'funda_forecasting'},
    include_package_data = True,
    install_requires = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotnine',
        'scikit-learn',
        'tensorflow',
        'xgboost',
        'pickle',
        'datetime',
        'json',
        'os',
        'pathlib',
        'sys',
        'warnings',
        'statistics',
        'itertools',
        'copy',
        'git+https://github.com/tensorflow/docs'
    ],
    python_requires = '==3.8.5'
)