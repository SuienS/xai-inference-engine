from setuptools import setup, find_packages

setup(
    name='example',
    version='0.1.0',
    packages=find_packages(include=['src.xai_inference_engine', 'src.xai_inference_engine.*']),
    install_requires=[
        'colorama==0.4.6',
        'contourpy==1.1.1',
        'cycler==0.12.1',
        'filelock==3.12.4',
        'fonttools==4.43.1',
        'fsspec==2023.10.0',
        'Jinja2==3.1.2',
        'kiwisolver==1.4.5',
        'MarkupSafe==2.1.3',
        'matplotlib==3.8.0',
        'mpmath==1.3.0',
        'networkx==3.2',
        'numpy==1.26.1',
        'packaging==23.2',
        'Pillow==10.1.0',
        'pyparsing==3.1.1',
        'pyproject_hooks==1.0.0',
        'python-dateutil==2.8.2',
        'six==1.16.0',
        'sympy==1.12',
        'tomli==2.0.1',
        'torch==2.1.0',
        'typing_extensions==4.8.0',
    ],
    extras_require={
        'interactive': ['jupyter'],
    },
    # setup_requires=['pytest-runner', 'flake8'],
    # tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main'] # TODO: Command shortcuts
    # }
    # package_data={'': ['*.json']} # TODO: additional data files
)