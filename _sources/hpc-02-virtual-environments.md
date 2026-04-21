---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Virtual Environments for High-Performance Computing

## Learning Objectives

- Explain why isolated software environments matter for scientific computing and shared clusters.
- Create and activate a Python virtual environment for interactive and batch HPC workflows.
- Distinguish between `venv`, `conda`, and system-wide module-based software stacks.
- Install packages reproducibly with `pip` and a requirements file.
- Avoid common HPC environment mistakes involving login nodes, compute nodes, and incompatible binaries.
- Design a simple workflow for moving from exploratory work to reproducible batch jobs.


## Why Virtual Environments Matter on HPC Systems

On a laptop, a messy Python environment can be inconvenient.  On a shared cluster, it can completely derail a workflow.  High-performance computing systems are shared by many users, often across many research groups, and the software installed by the system administrators is designed to serve a wide range of needs.  That means the default Python installation may not contain the packages you need, may contain versions that conflict with your code, or may change over time as the cluster is updated.

A **virtual environment** is an isolated Python installation that keeps your project-specific packages separate from the system Python and from other projects.  This isolation improves three things:

1. **Reproducibility**: your code runs against a known set of package versions.
2. **Stability**: installing a new package for one project does not break another project.
3. **Collaboration**: teammates can recreate the same environment from a short specification.

On HPC systems, virtual environments also reduce the temptation to install packages into shared locations or into the global user site directory, both of which tend to create confusing dependency problems.


## The Three Layers You Need to Keep Straight

When working on a cluster, it helps to think in layers:

1. **System and module layer**: software provided by the cluster, often accessed through `module load`.
2. **Python interpreter layer**: the specific Python executable you will run.
3. **Project environment layer**: the packages installed for one project, such as NumPy, SciPy, JAX, or PyTorch.

These layers interact.  For example, you might first load a Python module provided by the cluster, then create a `venv` using that Python interpreter, and finally install project packages inside that environment.

If you mix up these layers, strange things happen.  You may think you installed a package, but it may have been installed into a different interpreter than the one your batch job actually uses.  Or you may build an environment against one compiler or CUDA stack and then run it with another.


## `venv` as the Default Starting Point

For many Python-based scientific workflows, the simplest starting point is the built-in `venv` module.  It ships with Python and creates a lightweight, isolated environment for Python packages.

Here is a common pattern for setting up a `venv` environment on an HPC cluster:

```bash
module load python/3.11
python -m venv ~/envs/chem7870
source ~/envs/chem7870/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyter
```

Here is what each command does:

- `module load python/3.11`: loads the cluster-provided Python 3.11 software module, making that interpreter available in your shell.
- `python -m venv ~/envs/chem7870`: creates a new virtual environment named `chem7870` in the `~/envs` directory using the currently loaded Python interpreter.
- `source ~/envs/chem7870/bin/activate`: activates the virtual environment in the current shell so that `python` and `pip` refer to the environment's own executables.
- `python -m pip install --upgrade pip`: upgrades `pip` inside the virtual environment so package installation uses a recent version of the installer.
- `python -m pip install numpy scipy matplotlib jupyter`: installs the listed packages into the virtual environment rather than into the system Python.

After activation, your shell prompt usually changes to indicate that the environment is active.  The command `which python` should now point to the Python executable inside the environment.

Once you have created the environment, you can use it on later interactive sessions or inside batch jobs by activating it with `source ~/envs/chem7870/bin/activate` before running your code.  This makes it easy to maintain a consistent software environment across different runs and different users.


Some important details:

- It may be necessary to use `python -m pip`, not just `pip`, so the package installer is tied to the active interpreter.  Remember, you can always check which Python and pip you are using with `which python` and `which pip`.  If you are using the virtual environment correctly, both should point to the same location inside the environment directory.
- Create environments in a location you control, such as your home directory or project space.
- Install only what you need for the project.
- Record package versions once the environment works.

### Transporting and Recording Environments

Once you have constructed a virtual environment that works for your project, you should record the package specifications so that you or others can recreate it later.  The most common way to do this is with a `requirements.txt` file, which lists the packages and their versions.

```bash
python -m pip freeze > requirements.txt
```

This command captures the exact versions of all installed packages in the environment.  To recreate the environment later, you can use

```bash
python -m pip install -r requirements.txt
```


## Where `conda` Fits In

`conda` environments are also common in scientific computing, especially when packages have compiled dependencies or when users need a broader cross-language environment manager.  In some settings, `conda` can be more convenient than `venv`, particularly for complex data-science stacks.

A typical `conda` workflow looks like this:

```bash
module load miniconda
conda create -n lecture-hpc python=3.11
conda activate lecture-hpc
pip install numpy scipy matplotlib
```

Here, the `conda` environment is created with a specific Python version, and then `pip` is used to install packages.  You can also use `conda install` to install packages from the conda repositories, which may be preferable for some scientific packages.

However, on HPC systems, `conda` comes with tradeoffs:

- Environment creation can be slower and heavier than `venv`.
- Large environments consume substantial storage.
- Binary compatibility with cluster-provided MPI, CUDA, or compiler stacks requires care.

For this reason, `venv` is often the recommended starting point. 


## Modules and Virtual Environments Are Not Competitors

Students often think they must choose between environment modules and Python virtual environments.  Usually, you need both.

- The **module system** gives you access to cluster-provided software such as Python, CUDA, MPI, or compilers.
- The **virtual environment** isolates the Python packages for your project.

For example, on a GPU cluster you might do something like:

```bash
module purge
module load python/3.11 cuda/12.2
source ~/envs/ml-course/bin/activate
python train_model.py
```

This pattern says: use the cluster's Python and CUDA stack, but use your own project-specific Python packages.


## Batch Jobs Must Activate the Same Environment

One of the most common mistakes in HPC work is setting up an environment interactively and then forgetting to activate it inside the batch script.  Your code works on the login node, but the submitted job fails because it runs with a different Python interpreter.

Here is a simple SLURM script that uses a virtual environment correctly:

```bash
#!/bin/bash
#SBATCH --job-name=python_env_demo
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

module purge
module load python/3.11
source ~/envs/lecture-hpc/bin/activate

python analysis.py
```

The critical idea is consistency: the Python interpreter and package environment used in the job should match the one you tested.


## Common Failure Modes

### 1. Installing into the wrong Python

If `pip install` appears to succeed but `import` fails later, check:

```bash
which python
python -m pip --version
python -c "import sys; print(sys.executable)"
```

These commands should all point to the same environment.

### 2. Building on one node, running on another incompatible stack

If packages depend on compiled libraries, the environment may behave differently depending on what modules were loaded when the environment was created and when the job was run.

### 3. Using the login node for heavy installation or computation

Creating a small environment is usually fine on a login node, but compiling large packages or running actual workloads there is often discouraged or prohibited.  Moreover, certain installations may require access to specific compilers or CUDA versions that are only available on compute nodes -- I have seen cases where I needed to be on a compute node with a gpu to successfully install a GPU-accelerated package, even though the same installation command failed on the login node.



## Practical Recommendations

- Create one environment per project or assignment, not one giant environment for everything.
- Save package specifications in `requirements.txt` or another reproducible format.
- Test the environment with a short script before launching large jobs.
- Activate the environment explicitly in every batch script.
- Prefer scratch or project storage for large datasets, but keep small environment definitions under version control.
- Rebuild environments when they become confusing instead of patching them indefinitely.


## Summary

Virtual environments are a core tool for reliable scientific computing.  On HPC systems, they help you isolate dependencies, avoid conflicts with shared software, and reproduce workflows across interactive sessions and batch jobs.  The main operational rule is simple: be explicit about which Python you are using, install packages into that exact environment, and activate the same environment everywhere you run the code.