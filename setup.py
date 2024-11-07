from setuptools import setup, find_packages

requirements = ['anomalib',
                'bagpy',
                'cvxpy',
                'numpy',
                'lpips',
                'matplotlib',
                'moviepy',
                'opencv-python',
                'pandas',
                'Pillow',
                'pytorch-ood',
                'pyyaml',
                'rosnumpy',
                'rospkg',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'seaborn',
                'shapely',
                'tabulate',
                'torch',
                'torchvision',
                'tqdm'] 

setup(
    name="src",
    version="1.0.0",
    description="Probabilistic and Reconstruction-based Competency Estimation (PaRCE)",
    packages=find_packages(),
    install_requires=requirements
)