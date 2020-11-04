
# DDMISP: Preprocessing

Folder structure:

- `etc/`: Miscellaneous graphical files.
- `notebooks/`: IPython notebooks for analysis and figure creation.
- `scripts/`: Python scripts for preprocessing of the connectomes (`get_conn_matrix.py`), detection of the onset in the SEEG signals (`onsettimes.py`), mapping to onset times from channel space to region space (`mapping.py`), and extracting relevant metadata (`get_seizure_info.py`, `save_region_names.py`).


Top-level entry points is the Snakemake file `Snakefile` with the main target `all`. Run with `snakemake all`.
