from setuptools import setup, find_packages
from setuptools.extension import Extension
import Cython.Distutils

VERSION = "0.1.1"


class BuildExt(Cython.Distutils.build_ext):
    pass

setup(
    name='rh_icon',
    version=VERSION,
    packages=find_packages(include=["rh_icon", "rh_icon/*"]),
    author="Bjoern Andres",
    author_email="bjoern@andres.sc",
    maintainer="Lee Kamentsky",
    maintainer_email="lee_kamentsky@g.harvard.edu",
    url="https://github.com/rhoana/icon",
    download_url="https://github.com/archive/%s.tar.gz" % version,
    ext_modules = [Extension(
        "rh_icon/partition_comparison",
        ["partition_comparison/src/partition_comparison.pyx"],
        include_dirs=["partition_comparison/include"],
        language="c++")],
    install_requires=[
        "Cython>=0.23.0",
        "Theano>=0.8.2",
        "mahotas>=1.4.1",
        "progressbar>=2.3",
        "tornado>=4.3",
        "rh_logger",
        "rh_config"
    ],
    entry_points=dict(console_scipts=[
        "icon-webserver = rh_icon.web.server:main",
        "icon-train = rh_icon.model.train:main",
        "icon-segment = rh_icon.mode.segment:main"
    ])
    
)