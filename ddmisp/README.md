
# DDMISP: Core of the method

Folder structure:

- `daint/`: Helper scripts for running the jobs on Piz Daint supercomputer of the Swiss National Supercomputing Centre.
- `etc/`: Miscellaneous graphical files.
- `notebooks/`: IPython notebooks for analysis and figure creation.
- `scripts/`: Python scripts for figure creation.
- `stan/`: Stan files with the statistical models.
    - `ssinf.stan`: Single-seizure model
    - `msinf.stan`: Multi-seizure model
- `util/`: Python code for preparation, analysis, and visualization.


Top-level entry points are the Snakemake files:

- `main.smk`: Core of the work, including the hyperparameter learning, LOO validation, and virtual resection.
- `synth.smk`: Synthetic data validation. 
- `figs.smk`: Results visualization.
- `manifolds.smk`: Manifold figure.
