import os
import ast
from setuptools import find_packages, setup


base_dir = os.path.dirname(os.path.abspath(__file__))


def get_version():
    version_path = os.path.join(base_dir, "efficient_cross_entropy", "__init__.py")
    with open(version_path, "r", encoding="utf-8") as fp:
        version_file = fp.read()
    version_ast = ast.parse(version_file)
    for node in version_ast.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__":
                return ast.literal_eval(node.value)
    raise RuntimeError("Unable to find version string.")


setup(
    name='efficient_cross_entropy',
    packages=find_packages(),
    version=get_version(),
    license='MIT',
    description='This repo contains an implementation of a linear projection + cross-entropy loss PyTorch module that has substantially lower memory consumption compared to a standard implementation, with almost no additional compute cost. The memory savings come from two optimizations: 1) overwriting the logits with their gradients in-place and 2) not materializing the entire logits tensor.',
    author='mgmalek',
    author_email='',
    url='https://github.com/mgmalek/efficient_cross_entropy',
    keywords="triton pytorch deep-learning optimization cross-entropy-loss linear-projection fused-kernel gpu-acceleration high-performance-computing",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=["torch", "triton"],
    extras_require={
        "test": [
            "pytest"
        ],
    },
    include_package_data=True,
)