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

# Supercomputer Architectures for Computational Chemistry

## Learning Objectives

- Understand the basic architecture of modern supercomputing clusters, including login/input nodes, compute nodes, interconnects, and storage tiers.
- Map common computational chemistry workloads (DFT, molecular dynamics, Monte Carlo) to suitable hardware and parallelization strategies.
- Use core SLURM commands to submit, monitor, and manage jobs.
- Write and reason about practical SLURM job scripts for both CPU-only and GPU-accelerated workloads.
- Diagnose common performance bottlenecks (communication, memory, I/O, poor scaling) and propose practical fixes.
- Apply reproducibility and reliability best practices in HPC workflows.


## Why Architecture Matters in Computational Chemistry

In computational chemistry, scientific progress is often limited by three clocks:
1. Wall-clock runtime of each simulation.
2. Queue wait time before the job starts.
3. Human turnaround time to debug and rerun jobs.

A good understanding of supercomputer architecture helps with all three.  It improves your ability to request the right resources, avoid common workflow mistakes, and interpret performance results.

Some representative examples:
- **DFT / electronic structure calculations** often involve large linear algebra kernels and can be communication-heavy at scale.
- **Molecular dynamics (MD)** can run efficiently on GPUs, but performance can degrade if data movement and domain decomposition are poorly configured.
- **Monte Carlo (MC)** workflows often involve many independent replicas, making job arrays and scheduler strategy especially important.


## Cluster Architecture and Workload Mapping

### Anatomy of a Supercomputer Cluster

Most HPC systems are collections of computers, known as **nodes**, connected by a high-speed network.  Each node has its own CPUs, memory, and sometimes GPUs.  Nodes are categorized by their role:

- **Login / input nodes**:
  - Used for editing, compiling small programs, launching jobs, and light preprocessing/postprocessing.
  - Shared by many users.
  - Not intended for heavy computation.
- **Compute nodes**:
  - Dedicated resources where your scheduled jobs actually run.
  - Contain CPUs, memory, and often GPUs.
  - Accessed through the scheduler (not by running heavy jobs on login nodes).
- **Management / service nodes**:
  - Run infrastructure services (scheduler daemons, monitoring, authentication).
  - Typically not user-facing for workload execution.

A practical rule: do interactive lightweight work on login nodes, and run production chemistry workloads only through scheduled compute jobs.


#### Node-Level Hardware Concepts

Compute nodes have a complex hardware topology that affects performance: Each individual CPU is called a **socket**.  Each socket contains multiple **cores**, which are the actual execution units.  If you want to parallelize on a single node, you are parallelizing across cores.   Cores on the same node can share memory, which is the basis for shared-memory parallelism (e.g., OpenMP), allowing them to operate on the same data at once.  In contrast, cores on different nodes cannot share memory directly and must send each other messages over the network.  This is the basis for distributed-memory parallelism (e.g., MPI).


#### CPU vs GPU Nodes

Recently, more and more scientific workloads have been accelerated by Graphics Processing Units (GPUs).  GPUs were originally designed for rendering graphics, where rendering graphics requires performing the same operation on many pixels in parallel.  This makes them well-suited for certain types of scientific computations that can be expressed as data-parallel operations, such as matrix multiplications in DFT, force calculations in MD, or evaluating neurons in a neural network.  Moreover, python packages such as PyTorch and JaX have made it easier to write GPU-accelerated code.

Consequently, modern supercomputers often have a mix of CPU-only nodes and GPU nodes.  Typically, GPU nodes are more expensive to use and have longer queue times, so it's important to choose the right type of node for your workload.


<!-- ### Interconnect and Distributed Parallelism

When a job spans multiple nodes, MPI messages travel over the cluster interconnect.  Two high-level properties matter:

- **Latency**: cost of many small messages.
- **Bandwidth**: throughput for large data transfers.

A chemistry workload with frequent global communication can stall on interconnect overhead even if each node is fast. -->


### Storage Tiers and I/O Behavior

In addition to having different types of compute nodes, supercomputers also have different types of storage that are optimized for different purposes.  The two main types of storage are:

- **Home/project storage**: persistent, shared, often backed up.
- **Scratch storage**: high-throughput, temporary, not always backed up.

You generally want to write large temporary files (e.g., checkpoints, intermediate outputs) to scratch storage, and only write essential outputs (e.g., final results, logs) to home storage.  This is because scratch storage is optimized for high throughput and can handle large files more efficiently, while home storage is optimized for reliability and may have lower performance for large files.  Moreover, writing many small files to home storage can cause performance issues and increase the risk of hitting quotas.

Clusters may also have ``tape storage`` for long-term archival.  This is not intended for active use and can have very long access times, so it's generally not suitable for storing intermediate files or results that you need to access frequently.

### Parallelization Models in Chemistry Codes

Typical execution models:

- **Serial**: one process; simplest, but slow for large systems.
- **Shared-memory threading (OpenMP)**: threads on one node.
- **Distributed-memory (MPI)**: processes across nodes.
- **Hybrid MPI+OpenMP**: balance communication and node-level parallelism.
- **GPU offload**: selected kernels executed on accelerators.

No model is always best.  The right choice depends on problem size, code implementation, and architecture.


## SLURM, Performance Tuning, and Reliable Workflows

SLURM (Simple Linux Utility for Resource Management) is the scheduler that decides when and where jobs run on a cluster, based on resource requests, queue policy, and system availability.  Instead of launching heavy computations directly from a login node, users submit an `sbatch` script, which is a text file containing scheduler directives (such as number of nodes, walltime, memory, and GPU requests) plus the shell commands needed to run the program.  When you run `sbatch my_job.slurm`, SLURM places that script in the queue, allocates matching compute resources when available, executes the script on those resources, and records job status and accounting information.

Sbatch scripts are structured as bash scripts with special `#SBATCH` directives that specify resource requests and job parameters.  For example, a simple script to run a DFT calculation might look like this:

```bash
#!/bin/bash
#SBATCH --job-name=dft_job
#SBATCH --partition=compute
#SBATCH --account=chemistry
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=1000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out 
#SBATCH --error=logs/%x-%j.err

## Code to run the DFT calculation goes below
...
```
We will discuss the meaning of each directive and how to choose appropriate values in the next section.  For now, the key point is that you should never run heavy computations directly on a login node; instead, you should always submit a job script through SLURM to ensure that your job runs on the appropriate compute resources and does not interfere with other users.

### SLURM Concepts You Need Every Week

Common SLURM terms:

- **Partition**: queue of nodes with similar properties/policies.
- **Account**: project or allocation to charge usage.
- **QoS**: quality-of-service policy affecting limits/priority.
- **Job allocation**: resources granted to your job.
- **Job step**: one execution phase within an allocation (often via `srun`).

Core commands:

```bash
sbatch job.slurm      # submit batch job
squeue -u $USER       # inspect queued/running jobs
scancel 123456        # cancel a job
sacct -j 123456       # view accounting info for completed/running jobs
```


### Resource Requests: Getting the Right Shape

Important script directives:

- `--nodes`: number of nodes.
- `--ntasks`: total MPI tasks.
- `--cpus-per-task`: threads per task (often OpenMP threads).
- `--mem` or `--mem-per-cpu`: memory request.
- `--time`: walltime limit.
- `--gres=gpu:<n>`: GPU request.

Under-requesting causes failure; over-requesting increases queue time and can reduce throughput.


### Example 1: CPU-Oriented Hybrid MPI+OpenMP Job

```bash
#!/bin/bash
#SBATCH --job-name=dft_hybrid
#SBATCH --partition=compute
#SBATCH --account=chemistry
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load gcc/13 openmpi/4.1 quantum-espresso/7.3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Total ranks = nodes * ntasks-per-node = 32
# Total threads = ranks * cpus-per-task = 64 logical CPU threads
srun pw.x -in scf.in > scf.out
```

This pattern is useful when the code supports MPI+OpenMP and the problem size benefits from moderate inter-node parallelism.


### Example 2: GPU-Accelerated MD Job

```bash
#!/bin/bash
#SBATCH --job-name=md_gpu
#SBATCH --partition=gpu
#SBATCH --account=chemistry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load cuda/12.2 gromacs/2025

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun gmx mdrun -deffnm nvt -ntmpi 1 -ntomp ${OMP_NUM_THREADS} -gpu_id 0
```

This pattern is useful when one GPU dominates performance and the simulation system is not large enough to justify multi-node decomposition.


### Job Arrays for Replica and Parameter Sweeps

For many independent calculations (MC replicas, parameter scans), arrays reduce scripting overhead:

```bash
#!/bin/bash
#SBATCH --job-name=mc_array
#SBATCH --partition=compute
#SBATCH --account=chemistry
#SBATCH --array=0-31
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out

seed=$((1000 + SLURM_ARRAY_TASK_ID))
srun python run_mc_replica.py --seed ${seed}
```

Here `%A` is the array job id and `%a` is the array index.



### Monitoring and Postmortem Diagnostics

Useful checks during and after runs:

- `squeue -j <jobid>`: state, runtime, node assignment.
- `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocCPUS`: completion info.
- Output logs: code-level warnings, convergence failures, checkpoint activity.

Common failure patterns:
- **OUT_OF_MEMORY**: increase memory request or reduce concurrency.
- **TIMEOUT**: request realistic walltime or improve performance.
- **Low utilization**: mismatch between MPI ranks, thread count, and hardware topology.
- **I/O stalls**: too-frequent checkpoints or poor file layout.


### Performance Tuning Checklist

Before scaling up expensive jobs, run small profiling tests:

1. Confirm correctness on a short trajectory / small basis set.
2. Sweep a small grid of `(MPI ranks, threads, GPUs)`.
3. Record runtime, memory, and throughput metrics.
4. Pick the configuration with best science-per-day, not just best single-run speed.

For many projects, total completed simulations per week is a better KPI than minimum runtime of one run.


## Summary

A successful HPC user in computational chemistry needs both conceptual and operational fluency:

- Understand where computation happens (compute nodes) versus where workflows are managed (login/input nodes).
- Match method and code behavior to architecture (CPU, GPU, memory, interconnect, filesystem).
- Use SLURM effectively for single jobs, arrays, and dependency pipelines.
- Treat performance tuning and reproducibility as part of scientific method, not optional afterthoughts.

In practice, this combination usually yields faster turnaround, fewer failed runs, and more reliable scientific conclusions.
