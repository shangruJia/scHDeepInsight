from setuptools import setup, find_packages

setup(
    name="scHDeepInsight",  # Package name
    version="0.1.7",  # Version number
    author="Shangru JIA",  
    author_email="jiashangru@g.ecc.u-tokyo.ac.jp", 
    description="A tool for processing and hierarchically annotating immune scRNA-seq data with DeepInsight and CNN.",  # Package description
    long_description=open('README.md').read(),  # Read detailed description from README.md
    long_description_content_type="text/markdown",  # Type of the README file
    url="https://github.com/shangruJia/scHDeepInsight",  # Project homepage link
    packages=find_packages(),  # Automatically find and include all modules
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python version requirement
    install_requires=[
        "numpy",
        "pandas",
        "scanpy",
        "torch",
        "efficientnet-pytorch",
        "anndata",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "scipy"
        #"pyDeepInsight @ git+https://github.com/alok-ai-lab/pyDeepInsight.git@master#egg=pyDeepInsight"
    ],  # Dependencies
    include_package_data=True,  # Include static files in the package
    package_data={
        "scHDeepInsight": ["pretrained_files_immune/*.csv", "pretrained_files_immune/*.obj", "pretrained_files_immune/*.pth"],
    },
)
