import os

base_dir = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(base_dir, 'gai/api/VERSION')
with open(version_file, 'r') as f:
    VERSION = f.read().strip()

from setuptools import setup, find_packages
from os.path import abspath
import subprocess, os, sys, shutil
from setuptools.command.install import install
import json

thisDir = os.path.dirname(os.path.realpath(__file__))

def parse_requirements(filename):
    with open(os.path.join(thisDir, filename)) as f:
        required = f.read().splitlines()
    return required

class CustomInstall(install):
    def run(self):
        home_dir = os.path.expanduser("~")
        gairc_file = os.path.join(home_dir, ".gairc")

        if not os.path.isfile(gairc_file):
            with open(gairc_file, 'w') as f:
                config = {
                    "app_dir": "~/gai"
                }
                f.write(json.dumps(config, indent=4))

        gai_dir = os.path.join(home_dir, "gai")
        os.makedirs(gai_dir, exist_ok=True)

        gai_models_dir = os.path.join(home_dir, "gai","models")
        os.makedirs(gai_models_dir, exist_ok=True)

        if not os.path.isfile("gai.json"):
            raise Exception("gai.json file not found. Please make sure the file is in the root directory of the project.")
        else:
            print("Copying gai.json to ~/gai")
        shutil.copy("gai.json", gai_dir)

        # Proceed with the installation
        install.run(self)

setup(
    name='gai-gen',
    version=VERSION,
    author="kakkoii1337",
    author_email="kakkoii1337@gmail.com",
    packages=find_packages(exclude=["tests*","gai.api"]),
    description = """Gai/Gen: Multi-Modal Wrapper Library for Local LLM. The library is designed to provide a simplified and unified interface for seamless switching between multi-modal open source language models on a local machine and OpenAI APIs.""",
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3.10',
        "Development Status :: 3 - Alpha",        
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",        
        'Operating System :: OS Independent',
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",        
        "Topic :: Scientific/Engineering :: Artificial Intelligence",        
    ],
    python_requires='>=3.10',        
    install_requires=[
    ],
    extras_require={
        "TTT": parse_requirements("requirements_ttt.txt"),
        "ITT": parse_requirements("requirements_itt.txt"),
        "ITT2": parse_requirements("requirements_itt2.txt"),
        'STT': parse_requirements("requirements_stt.txt"),
        'TTS': parse_requirements("requirements_tts.txt"),          
        'RAG': parse_requirements("requirements_rag.txt"),
        'TTC': parse_requirements("requirements_ttc.txt")
    },
    cmdclass={
        'install': CustomInstall,
    },    
)