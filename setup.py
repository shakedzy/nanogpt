import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent.resolve()

PACKAGE_NAME = "nanogpt"
AUTHOR = "Shaked Zychlinski"
AUTHOR_EMAIL = "shakedzy@gmail.com"

LICENSE = "MIT"
VERSION = '0.1.0'
DESCRIPTION = 'NanoGPT'
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf8")
LONG_DESC_TYPE = "text/markdown"

requirements = (HERE / "requirements.txt").read_text(encoding="utf8")
INSTALL_REQUIRES = [s.strip() for s in requirements.split("\n")]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data={
        PACKAGE_NAME: ['__resources__/*'],  
    },
)
