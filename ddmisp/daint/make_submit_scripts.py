#!/usr/bin/env python3

import glob
import os
import sys
import shutil

NCHAINS = 2
BATCH_SIZE = 120


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def make_submit_scripts(direc, script_direc):
    """
    Create slurm submit scripts for all solo runs.
    """
    # if os.path.isdir(script_direc):
    #     shutil.rmtree(script_direc)
    #os.makedirs(script_direc)

    dirs_to_create = []

    infiles_all = []
    for sid in range(1, 26):
        infiles_all.extend(sorted(glob.glob(f"run/solo/*/vep/id{sid:03d}/input/*.R", recursive=True)))


    for i, infiles in enumerate(chunks(infiles_all, BATCH_SIZE // NCHAINS)):
        filename = os.path.join(script_direc, f"greasy_batch_{i:04d}.txt")

        with open(filename, 'w') as fh:

            for infile in infiles:
                subjdir = os.path.split(os.path.dirname(infile))[0]
                case = os.path.splitext(os.path.split(infile)[1])[0]

                dirs_to_create.append(os.path.join(subjdir, "output", case))
                dirs_to_create.append(os.path.join(subjdir, "log", case))

                for chain in range(1, NCHAINS+1):
                    outfile = os.path.join(subjdir, "output", case, f"chain_{chain}.csv")
                    logfile = os.path.join(subjdir, "log", case, f"chain_{chain}.txt")

                    fh.write("./stan/ssinf sample num_warmup=500 num_samples=500"
                             f" random seed=42 id={chain}"
                             f" data file={infile}"
                             f" output file={outfile} &> {logfile}\n")


    with open(os.path.join(script_direc, "make_dirs.sh"), 'w') as fh:
        fh.write("#!/bin/bash\n\n")
        for direc in dirs_to_create:
            fh.write(f"mkdir -p {direc}\n")


make_submit_scripts("run/solo", "run/scripts")
