from setuptools import setup, find_packages
from setuptools.extension import Extension
import Cython.Distutils
import glob

VERSION = "0.1.1"

rh_icon_package_data = [_[len("rh_icon/"):] for _ in
                        glob.glob("rh_icon/web/resources/*")+
                        glob.glob("rh_icon/web/resources/**/*")]

setup(
    name='rh_icon',
    version=VERSION,
    packages=find_packages(
        include=["rh_icon", "rh_icon.*", "rh_icon.*.*", "rh_icon.*.*.*"]),
    author="Bjoern Andres",
    author_email="bjoern@andres.sc",
    maintainer="Lee Kamentsky",
    maintainer_email="lee_kamentsky@g.harvard.edu",
    url="https://github.com/rhoana/icon",
    download_url="https://github.com/archive/%s.tar.gz" % VERSION,
    ext_modules = [Extension(
        name="rh_icon.partition_comparison._partition_comparison",
        sources=["rh_icon/partition_comparison/src/partition_comparison.pyx"],
        include_dirs=["rh_icon/partition_comparison/include"],
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
    entry_points=dict(console_scripts=[
        "icon-webserver = rh_icon.web.server:main",
        "icon-train = rh_icon.model.train:main",
        "icon-segment = rh_icon.model.segment:main"
    ]),
    package_data=dict(rh_icon=rh_icon_package_data)
)