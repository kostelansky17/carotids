import setuptools
 
setuptools.setup(
    name="carotids-kostelansky",
    version="1.0",
    author="Martin Kostelansky",
    author_email="martin.kostelansky@fel.cvut.cz",
    project_urls={
        "github": "https://github.com/kostelansky17/carotids",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
