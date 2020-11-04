#!/usr/bin/env python3


import os
import tempfile
import subprocess

import numpy as np


import util.io as io




def eval_lpdf(stan_binary, data, parameters):
    # Get temp directory
    tmpdirec = tempfile.TemporaryDirectory()
    tmpdir = tmpdirec.name

    # create data file
    data_file = os.path.join(tmpdir, "data.R")
    io.rdump(data_file, data)

    # Create continuation file
    param_file = os.path.join(tmpdir, "param.R")
    io.rdump(param_file, parameters)


    # Run it for one iteration
    out_file = os.path.join(tmpdir, "out.csv")
    run = subprocess.run([stan_binary, "sample", "num_warmup=0", "num_samples=1", "adapt", "engaged=0",
                          "data", f"file={data_file}", f"init={param_file}", "output", f"file={out_file}"],
                         capture_output=True)
    assert run.returncode == 0

    # Parse the results
    lp = io.parse_csv(out_file)['lp__'][0]

    return lp


data = io.rload("run/learn/ABN/vep/input/learn.R")


n_seizures = data['n_seizures']
nreg = data['nreg']

ndata = {
    'n_seizures': int(data['n_seizures']),
    'nreg': int(data['nreg']),
    'w': data['w'],

    'n_sz': np.array(data['n_sz'], dtype=int),
    'reg_sz': np.array(data['reg_sz'], dtype=int),
    't_sz': data['t_sz'],

    'n_ns': np.array(data['n_ns'], dtype=int),
    'reg_ns': np.array(data['reg_ns'], dtype=int),

    't_lim': data['t_lim'],
    'sig_t': data['sig_t'],
    'sids': np.array(data['sids'], dtype=int),
    'rids': np.array(data['rids'], dtype=int),
    't_shifts': data['t_shifts']

}

c = np.zeros((n_seizures, nreg))
c[0][0] = 1

parameters = {
    'q11': -13.71,
    'q12': -13.47,
    'qa21':  6.17,
    'qa22': 85.84,
    # 'c': np.random.normal(0, 1, size=(n_seizures, nreg))
    # 'c': np.ones((n_seizures, nreg))
    'c': c
}


lp = eval_lpdf("stan/simpropmblp", ndata, parameters)
print(lp)
