import shutil
from setuptools import setup, find_packages
import os,json
from setuptools.command.install import install

base_dir = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(base_dir, 'VERSION')
with open(version_file, 'r') as f:
    VERSION = f.read().strip()
thisDir = os.path.dirname(os.path.realpath(__file__))

def parse_requirements(filename):
    with open(os.path.join(thisDir, filename)) as f:
        required = f.read().splitlines()
    return required

class CustomInstall(install):
    def run(self):
        home_dir = os.path.expanduser("~")
        gairc_file = os.path.join(home_dir, ".gairc")
        with open(gairc_file, 'w') as f:
                config = {
                    "app_dir": "~/gai"
                }
                f.write(json.dumps(config, indent=4))

        gai_dir = os.path.join(home_dir, "gai")
        os.makedirs(gai_dir, exist_ok=True)

        gai_models_dir = os.path.join(home_dir, "gai","models")
        os.makedirs(gai_models_dir, exist_ok=True)

        config = os.path.join(thisDir,"gai.yml")
        if not os.path.isfile("gai.yml"):
            raise Exception("gai.yml file not found. Please make sure the file is in the root directory of the project.")
        else:
            print("Copying gai.yml to ~/gai")
        shutil.copy("gai.yml", gai_dir)

        # Proceed with the installation
        install.run(self)

setup(
    name='gai-lib',
    version=VERSION,
    author="kakkoii1337",
    author_email="kakkoii1337@gmail.com",
    packages=find_packages(),
    description = """""",
    long_description="Refer to https://gai-labs.github.io/gai/gai-lib for more information",
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
        parse_requirements("requirements.txt")
    ],
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'ttt=gai.cli.ttt:main',
            'tts=gai.cli.tts:main',
            'chunker=gai.cli.chunker:main',
            'pdf2txt=gai.cli.pdf2txt:main',
            'txt2md=gai.cli.txt2md:main',
            'gg=gai.cli.gg2:main',
            'summary=gai.cli.summarize:main',
            'scrape=gai.cli.scrape:main',
            'gai=gai.cli.Gaicli:main',
        ],
    },
    cmdclass={
        'install': CustomInstall,
    },    
)