import setuptools
import os
import os.path


# Get the readme file
if os.path.isfile("README.md"):
    with open("README.md", "r") as fh:
        long_description = fh.read()
else:
    long_description = ""

setuptools.setup(
    name="ed_lgt",
    version="0.0.0",
    author="Giovanni Cataldi",
    author_email="giovacataldi96@gmail.com",
    description="Exact Diagonalization for Lattice Gauge Theories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gcataldi96/ed-lgt",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={
        "ed_lgt": "ed_lgt",
        "ed_lgt.modeling": "ed_lgt/modeling",
        "ed_lgt.models": "ed_lgt/models",
        "ed_lgt.operators": "ed_lgt/operators",
        "ed_lgt.symmetries": "ed_lgt/symmetries",
        "ed_lgt.tools": "ed_lgt/tools",
        "ed_lgt.workflows": "ed_lgt/workflows",
    },
    packages=[
        "ed_lgt",
        "ed_lgt.modeling",
        "ed_lgt.models",
        "ed_lgt.operators",
        "ed_lgt.symmetries",
        "ed_lgt.tools",
        "ed_lgt.workflows",
    ],
    python_requires=">=3.6",
    # entry_points = { 'console_scripts': ['build_exec = ed_lgt.bin.compiler:main', ], },
    # These packages are not mandatory for a pip installation. If they are not there
    # they will simply be ignored.
)
