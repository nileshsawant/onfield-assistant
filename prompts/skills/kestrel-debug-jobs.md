Procedures and quirks for running short debug/interactive SLURM jobs on NLR Kestrel.

When the user is preparing or debugging a short SLURM job on Kestrel, prefer the
guidance below over generic SLURM advice.

## Partition rules

- The `debug` partition wall-clock cap is **1 hour**. Reject any
  `#SBATCH --time=` greater than `01:00:00` and propose `00:30:00` or
  `01:00:00` instead.
- `debug` nodes are shared. Always set `--ntasks` and `--cpus-per-task`
  explicitly — do not rely on partition defaults.
- For GPU debugging use the `debug` partition with `--gres=gpu:h100:1`.

## Mandatory flags on Kestrel

- `--account=<project>` is required. Debug submissions fail silently if
  it is missing — if the user hasn't provided one, ask for it before
  writing a script.
- Add `--exclusive` only when the user explicitly asks for a whole node;
  it is not the default on Kestrel.

## Interactive sessions

- Use `salloc` (not `srun --pty bash`) for an interactive shell on a
  compute node. After allocation, `ssh` to the assigned hostname rather
  than `srun bash` to keep the controlling TTY clean.
- Example: `salloc -A <project> -p debug -t 00:30:00 -N 1 --ntasks-per-node=8`.

## Common pitfalls

- `module purge` before loading the project's modulefile, otherwise
  pre-loaded system modules (especially MPI variants) collide.
- Cray `cc`/`CC` wrappers behave differently from system `gcc` — when
  the user reports a link error mentioning `cray-libsci` or `cray-mpich`,
  recommend they confirm `module list` matches the build that worked.
- Output paths in `--output=...` must already exist; SLURM does not
  create parent directories.
