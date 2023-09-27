# (Packages/Libraries) Miscellaneous
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        print("Installing IntelliGenes...")
        install.run(self)
        print("Installed IntelliGens!")

setup(
    name = 'IntelliGenes',
    version = '1.0',
    author = 'William DeGroat',
    author_email = 'will.degroat@rutgers.edu',
    description = 'IntelliGenes: AI/ML pipeline for predictive analyses using multi-genomic profiles.',    
    url = 'https://github.com/drzeeshanahmed/intelligenes',
    packages = find_packages(),  
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ], 
    install_requires = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'shap',
        'matplotlib',
        'scipy'
    ],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': [
            'igenes_predict=classification:main',
            'igenes_select=selection:main',
            'igenes=classification:main',
        ]
    },
    include_package_data  = True,
    license = 'GNU General Public License v3.0',
    cmdclass = {
        'install': CustomInstall,
    }
)
