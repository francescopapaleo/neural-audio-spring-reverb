from setuptools import setup, find_packages

setup(name='neural_springrev',
        version='0.1',
        description='Neural Audio Spring Reverb',
        author='Francesco Papaleo',
        url='https://github.com/francescopapaleo/neural-audio-spring-reverb',
        packages=find_packages(),
        install_requires=[
                'torch',
                'torchaudio',
                'numpy',
                'matplotlib',
                'scipy',
                'h5py',
                'auraloss',
                'tensorboard',
        ],
)
