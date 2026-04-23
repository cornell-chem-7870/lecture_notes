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

# Command-Line Interfaces and Argparse for Scientific Computing

## Learning Objectives

- Explain what a command-line interface (CLI) is and why it is central to scientific and HPC workflows.
- Build Python programs with robust command-line arguments using argparse.
- Distinguish positional arguments, optional flags, typed arguments, and mutually exclusive options.
- Combine command-line arguments with configuration files for reproducible experiment management.
- Design CLI tools that work cleanly in SLURM batch jobs and parameter sweeps.
- Apply practical reliability patterns, including logging, dry runs, and argument validation.


## Why CLIs Matter in Scientific Computing

In high-performance computing, most serious work is launched through scripts and schedulers, not point-and-click interfaces.  On many clusters, compute nodes have no graphical environment at all, and jobs are submitted remotely through tools like SLURM.  That means your program needs a clear text interface so users can run it with different parameters, automate repeated experiments, and embed it in larger workflows.

A command-line interface is simply a way to pass inputs to a program from the shell.  For example, consider a code `run_simulation.py` that performs a molecular dynamics simulation.  A CLI allows you to specify parameters like the time step, total simulation time, and integration method directly from the command line:

```bash
python run_simulation.py data/structure.xyz --dt 1e-3 -T 10 --method leapfrog
```

Compared with hard-coding values in Python files, a CLI gives you:

1. Reproducibility: the exact parameters are visible in the job script and logs.
2. Automation: shell scripts and job arrays can vary arguments across many runs.
3. Composability: your tool can be combined with other command-line tools.
4. Better collaboration: teammates can run the same command and get the same setup.


## Anatomy of a CLI Command

Most command-line programs accept two classes of inputs:

- Positional arguments: required values based on position.
- Optional arguments (flags): named options beginning with - or --.

In the example above, `data/structure.xyz` is a positional argument, while `--dt`, `-T`, and `--method` are optional flags.  Optional flags can have default values, type constraints, and limited choices.  They can also be boolean switches that turn features on or off.  For instance, -- `--verbose` might enable detailed logging when present, and do nothing when absent.

Some common design conventions:

- Use short names for frequent options, for example `-T 100`.
- Use long names for readability, for example `--t-end 100`.
- Use booleans as switches, for example `--verbose`


## Argparse: The Standard Library Workhorse

Python's argparse module provides a dependable way to define and parse command-line arguments.  For HPC classes, a major advantage is that argparse is part of the standard library, so students do not need extra CLI dependencies on shared systems.

A minimal example:


```python
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple molecular dynamics simulation."
    )
    parser.add_argument("structure_file", type=Path,
                        help="Input structure file (e.g., XYZ or PDB)")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="Time step size")
    parser.add_argument("-T", "--t-end", type=float, required=True,
                        help="Final simulation time")
    parser.add_argument("--method", choices=["euler", "rk4", "leapfrog"],
                        default="rk4", help="Integrator")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()
```

In this example, `argparse.ArgumentParser(...)` creates a parser object that knows how to read command-line inputs.  The first `add_argument(...)` defines a required positional argument (`structure_file`) for the input molecular structure, while later calls define optional flags such as `--dt` and `--method`.  Each argument declaration can include type, default value, and constraints (for example, `choices`).  
Additionally, the `help` parameter provides documentation that is displayed when the user runs the script with `--help`.  This is crucial for usability and for helping users learn to use your code.

Once called, the `args` variable will contain the parsed command-line arguments as attributes, for example `args.dt`, `args.t_end`, and `args.method`.  These values can then be used to control the behavior of the program, such as which integration method to use or how long to run the simulation.
(You can also convert the `args` namespace to a dictionary with `vars(args)` for easier manipulation, but using the namespace attributes is often more convenient, easier to read, and more type-friendly.)

When `parse_args()` is called, argparse reads the actual command line, validates the values, and returns them in the `args` namespace.  The `main()` function then uses these parsed values to control program behavior.  From the perspective of a user, they can run the program with different parameters without changing the code, for example:

```bash
python run_simulation.py data/water.pdb --t-end 5 --dt 0.005 --method leapfrog
```

The user can also get help on the available options by running:

```bash
python run_simulation.py --help
```


## Important Argument Patterns for Scientific Workflows

### Typed Arguments and Validation

Scientific codes often require strict types and ranges.  Argparse can enforce many of these constraints directly.

```python
import argparse


def positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return x


parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=positive_float, required=True)
parser.add_argument("--num-steps", type=int, default=1000)
```

The custom type function catches invalid values early, before expensive computation starts.

### File and Path Arguments

Many tools accept paths to input datasets, checkpoints, and output directories.

```python
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path, help="Input data file")
parser.add_argument("--output-dir", type=Path, default=Path("results"))
```

Using Path objects in your script simplifies path logic and improves portability.

### Mutually Exclusive Modes

Sometimes two options should never be used together.

```python
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true")
group.add_argument("--evaluate", action="store_true")
```

This avoids ambiguous usage and makes program behavior explicit.


## Configuration Files Plus CLI Overrides

As projects grow, putting every parameter on the command line becomes tedious.  A common scientific pattern is:

1. Load defaults from a configuration file.
2. Override selected values from CLI flags.

This gives reproducibility from a saved config and flexibility for quick sweeps.

### An Introduction to TOML

To implement this, we are going to use the TOML format for configuration files, which is human-friendly and supported by Python's standard library in version 3.11 and later.
TOML files are structured as key-value pairs, and they can also include sections for better organization.  For example, you might have a `config.toml` file that looks like this:

```toml
method = "rk4"
dt = 0.001
t_end = 10.0
seed = 42
output_dir = "results/base"
```

(More complex configurations can include nested arguments, lists, and tables, but this simple flat structure is often sufficient.)

Loading this configuration in your Python script can be done using the `tomllib` module.  For instance, you can read the configuration file  using the following code snippet:

```python
import tomllib

def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)
```

This will give you a dictionary with the contents of the TOML file.  The dictionary will have keys corresponding to the sections in the TOML file, so you can access the simulation parameters with `config["method"]`, `config["dt"]`, etc.


And the command is:

```bash
python run_simulation.py --config config.toml --dt 0.0005 --output-dir results/testA
```

A practical implementation pattern:

```python
import argparse
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--method", choices=["euler", "rk4", "leapfrog"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    args = merge_config(args, parser)  # Merge config file values with CLI overrides
    return args


def merge_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """
    Merge command-line arguments with config file values, giving precedence to CLI.
    Code adapted from Micha Feigin's Medium Post, "On using config files with python's argparse."
    """
    if args.config is not None:
        with args.config.open("rb") as f:
            cfg = tomllib.load(f)
            # This assumes 
            parser.set_defaults(**cfg)

        # Reload args to apply config defaults
        args = parser.parse_args()
    return args


def main() -> None:
    cli_args = parse_args()
    config = merge_config(cli_args)
```

Note: YAML and JSON are also common.  TOML is nice because Python 3.11 includes tomllib in the standard library.

### Reproducibility Tip

At run start, write the final merged configuration to your output directory.  This creates an audit trail that allows you to recreate any run later.
To do this, you can use the `tomli_w` library to write the configuration back to a TOML file.  (Unfortunately, the `tomllib` module does not support writing TOML files.)  For example:

```python
import tomli_w

def save_config(args: argparse.Namespace, output_dir: Path) -> None:
    config_path = output_dir / "arguments.toml"
    with config_path.open("wb") as f:
        arguments_dict = vars(args)  # Convert Namespace to dict
        f.write(tomli_w.dumps(arguments_dict).encode("utf-8"))
```


## CLI Design for SLURM and Batch Workloads

A well-designed CLI should be easy to call from batch scripts and job arrays.

Example SLURM script fragment:

```bash
#!/bin/bash
#SBATCH --job-name=ode_sweep
#SBATCH --partition=compute
#SBATCH --array=0-4
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

DT_VALUES=(0.1 0.05 0.02 0.01 0.005)
DT=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}

python run_simulation.py \
  --config config.toml \
  --dt ${DT} \
  --output-dir results/dt_${DT}
```

Key HPC-aware features to include in your CLI:

- --dry-run to print resolved parameters and exit.
- --output-dir to avoid hard-coded paths.
- --seed for deterministic stochastic experiments.
- Clear nonzero exit behavior for invalid inputs.


## Organizing Larger CLIs with Subcommands

For larger projects, one script may need multiple actions.  For instance, in machine learning one might need to train, evaluate, and plot.  Argparse supports subcommands that allow you to group related arguments under different modes of operation.

```python
import argparse

parser = argparse.ArgumentParser(prog="chemtool")
subparsers = parser.add_subparsers(dest="command", required=True)

train_parser = subparsers.add_parser("train", help="Train a model")
train_parser.add_argument("--epochs", type=int, default=100)

eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
eval_parser.add_argument("--ckpt", required=True)

plot_parser = subparsers.add_parser("plot", help="Plot outputs")
plot_parser.add_argument("--input", required=True)
```

This pattern scales better than one giant parser with dozens of unrelated options.


## Reliability and Debugging Patterns

Small CLI design choices save substantial cluster time:

1. Print parsed arguments at startup, for example print(vars(args)).  This makes it easy to verify that your CLI is working as intended and that the correct parameters are being used before launching expensive computations.
2. Validate filesystem assumptions early and fail fast.
3. Separate parse_args from main so logic is testable.
4. Keep defaults sensible and document units, for example seconds, kelvin, angstrom.
5. Include examples in the help text and in project documentation.

A useful dry-run behavior:

```bash
python run_simulation.py --config config.toml --dry-run
```

This should report the final resolved configuration and confirm paths without launching expensive computation.


## Common Mistakes and How to Avoid Them

- Hidden defaults that users do not know exist.
- Arguments that silently accept invalid values.
- Inconsistent naming conventions, for example mixing --t_end and --t-end.
- Hard-coded absolute paths in scripts.
- Different parameter handling between interactive runs and batch jobs.

Most of these are solved by thoughtful parser design, explicit validation, and run metadata logging.


## Suggested In-Class Exercise

Build a script named run_trajectory.py that:

1. Accepts --config, --dt, --t-end, --method, --seed, and --output-dir.
2. Reads defaults from a TOML file.
3. Applies CLI overrides.
4. Supports --dry-run.
5. Writes the final resolved parameters to output_dir/run_config_used.toml.

Then write a SLURM array script that sweeps over dt values and stores outputs in separate folders.

This exercise reinforces CLI design, reproducibility, and scheduler integration in one workflow.


## Summary

For scientific computing and HPC, command-line interfaces are not optional extras.  They are the standard interface between your code, your scheduler, and your research workflow.  Argparse gives a robust and dependency-light foundation for building these interfaces in Python.  When combined with configuration files, clear validation, and batch-friendly design, your tools become easier to automate, debug, reproduce, and scale.
