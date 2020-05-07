import setuptools

with open("README.md", "r") as fh:
    readme = fh.read()

setuptools.setup(
    long_description=readme,
    url="https://github.com/UOA-CS302-2020/Roxora-AI",
    packages=setuptools.find_packages(),
    ],
    python_requires='>=3.6',
)
