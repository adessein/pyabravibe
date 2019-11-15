import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyabravibe",
    version="1.0",
    author="Arnaud Dessein",
    author_email="adessein@protonmail.com",
    description="Python version of Anders Brandt AbraVibe Matlab Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adessein/pyabravibe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ]
)
