from setuptools import find_packages, setup

setup(
    name='survival_plus_x',
    packages=find_packages(),
    version='0.1.0',
    description='CNNs and ViTs for outcome prediction with additional tasks',
    author='Sebastian Starke',
    license='GNU GPLv3',
    install_requires=[
        'einops>=0.3',
        'lifelines>=0.27.0',
        'monai==0.8.1',
        'nibabel>=3.2.2',
        'numpy',
        'pandas',
        'pytorch_lightning>=1.5.8',
        'scikit_image>=0.19.2',
        'scikit_learn>=0.24.2',
        'scipy==1.8.0',
        'torch==1.11.0'
    ]
)
