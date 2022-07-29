from setuptools import setup

with open("./README.md") as fp:
    long_description = fp.read()

setup(
    name="adbpyg_adapter",
    author="Anthony Mahanna",
    author_email="anthony.mahanna@arangodb.com",
    description="Convert ArangoDB graphs to PyG & vice-versa.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arangoml/pyg-adapter",
    keywords=["arangodb", "pyg", "adapter"],
    packages=["adbpyg_adapter"],
    include_package_data=True,
    python_requires=">=3.7",
    license="Apache Software License",
    install_requires=[
        "requests>=2.27.1",
        "rich>=12.5.1",
        "pandas>=1.3.5",
        "python-arango>=7.4.1",
        "torch>=1.12.0",
        "torch-sparse>=0.6.14",
        "torch-scatter>=2.0.9",
        "torch-geometric>=2.0.4",
        "setuptools>=45",
    ],
    extras_require={
        "dev": [
            "black>=22.6.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.790",
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "coveralls>=3.3.1",
            "types-setuptools>=63.2.1",
            "types-requests>=2.28.5",
            "networkx>=2.5.1",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
)
