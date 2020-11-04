from collections import defaultdict
import pandas as pd

import util

configfile: "config.yml"
CMDSTANDIR = config['CMDSTANDIR']


NSZGROUP = 12
ATLAS = "vep"
ONSET = "INC"
SIDS = list(range(1, 51))
NCHAINS_SOLO = 2
NCHAINS_GRP = 4

# NSIMSOLO = defaultdict(lambda: 96, fD_nobs021=100, fD_realistic=100)
# NSIMSOLO = defaultdict(lambda: 96)

NSIMSOLO = {'random': 96, 'ez': 32}
EZ_CONFIGS = ["observed", "hidden-random", "hidden-nearmiss"]

# See notebooks/2020-01-24_FunctionF.ipynb
FUNS = {
    'A':  [-5.117447424199997,  -5.117447424199997,  1.9464059407999987, 1.9464059407999987],
    'B':  [-8., 2., 4., 13.],
    'C': [-10., 2., 5.5, 33.],
    'D':  [-12.699722249999999, 15.479622175000001,  5.527035135,       75.21215649999999]
}
NOBSS = [21, 54, 108]


localrules: genDataSoloRandom, genDataSoloEz, genDataGroup, checkChain, summarySolo, summaryGroup, postSolo, aggregSolo, joinSoloRandom, joinSoloEz

rule allSolo:
    input:
        random=expand("run/synth/solo/random/f{f}_nobs{nobs:03d}/s{sim:03d}/img/ct.pdf",
                      f=FUNS.keys(), nobs=NOBSS, sim=range(NSIMSOLO['random'])),
        ez=expand("run/synth/solo/ez/{config}/s{sim:03d}/img/ct.pdf", 
                  config=EZ_CONFIGS, sim=range(NSIMSOLO['ez']))

rule allGenDataSolo:
    input:
        random=expand("run/synth/solo/random/f{f}_nobs{nobs:03d}/s{sim:03d}/input/input.R",
                      f=FUNS.keys(), nobs=NOBSS, sim=range(NSIMSOLO['random'])),
        ez=expand("run/synth/solo/ez/{config}/s{sim:03d}/input/input.R", 
                  config=EZ_CONFIGS, sim=range(NSIMSOLO['ez']))

rule allGroup:
    input:
        expand("run/synth/group/f{f}_nobs{nobs:03d}/output/summary.csv", f=FUNS.keys(), nobs=NOBSS)

rule allGenDataGroup:
    input: expand("run/synth/group/f{f}_nobs{nobs:03d}/input/input.R", f=FUNS.keys(), nobs=NOBSS)


rule genDataSoloRandom:
    input:
        conns=[f"data/conn/{ATLAS}/id{sid:03d}.txt" for sid in SIDS]
    output:
        rfile="run/synth/solo/random/f{f}_nobs{nobs}/s{sim}/input/input.R",
        groundtruth="run/synth/solo/random/f{f}_nobs{nobs}/s{sim}/groundTruth.json"
    run:
        util.prep_synth_solo_random(input.conns, FUNS[wildcards.f], int(wildcards.nobs),
                                    int(wildcards.sim), output.rfile, output.groundtruth)

rule genDataSoloEz:
    input:
        conns=[f"data/conn/{ATLAS}/id{sid:03d}.txt" for sid in SIDS],
        onsets=[f"data/onset_regions/{ONSET}/{ATLAS}/id{sid:03d}.json" for sid in SIDS]
    output:
        rfile="run/synth/solo/ez/{config}/s{sim}/input/input.R",
        groundtruth="run/synth/solo/ez/{config}/s{sim}/groundTruth.json"
    run:
        util.prep_synth_solo_ez(input.conns, input.onsets, FUNS['D'], wildcards.config,
                                int(wildcards.sim), output.rfile, output.groundtruth)


rule genDataGroup:
    input:
        conns=[f"data/conn/{ATLAS}/id{sid:03d}.txt" for sid in SIDS],
    output:
        rfile="run/synth/group/f{f}_nobs{nobs}/input/input.R",
        groundtruth="run/synth/group/f{f}_nobs{nobs}/groundTruth.json"
    run:
        util.prep_synth_group(input.conns, FUNS[wildcards.f], int(wildcards.nobs), NSZGROUP,
                              output.rfile, output.groundtruth)


rule compile:
    input: "stan/{name}.stan"
    output: "stan/{name}"
    shell: "dir=`pwd`; cd {CMDSTANDIR}; make $dir/stan/{wildcards.name}"


rule checkChain:
    input: "{d}/chain_{chain}.csv"
    output: "{d}/chain_{chain}.status"
    run:
        summfile = f"{wildcards.d}/summary_chain_{wildcards.chain}.csv"
        shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={summfile} {input[0]}")
        util.check_chain(summfile, output[0])


rule runSolo:
    input:
        bin="stan/ssinf",
        data="run/synth/solo/{set}/{d}/s{sim}/input/input.R"
    output:  "run/synth/solo/{set}/{d}/s{sim}/output/chain_{chain}.csv"
    log:     "run/synth/solo/{set}/{d}/s{sim}/log/chain_{chain}.txt"
    threads: 1
    shell:
        """
        export STAN_NUM_THREADS=1
        ./{input.bin} sample                      \
            num_warmup=500 num_samples=500        \
            random seed=42 id={wildcards.chain}   \
            data file={input.data}                \
            output file={output[0]} &> {log}
        """

rule runGroup:
    input:
        bin="stan/msinf",
        data="run/synth/group/{d}/input/input.R"
    output:  "run/synth/group/{d}/output/chain_{chain}.csv"
    log:     "run/synth/group/{d}/log/chain_{chain}.csv"
    threads: NSZGROUP,
    shell:
        """
        export STAN_NUM_THREADS={threads}
        ./{input.bin} sample                     \
            num_warmup=500 num_samples=500       \
            random seed=42 id={wildcards.chain}  \
            data file={input.data}               \
            output file={output[0]} &> {log}
        """


rule summarySolo:
    input:
        resfiles=expand("run/synth/solo/{{d}}/chain_{chain}.csv",    chain=range(1, NCHAINS_SOLO+1)),
        statuses=expand("run/synth/solo/{{d}}/chain_{chain}.status", chain=range(1, NCHAINS_SOLO+1))
    output: "run/synth/solo/{d}/summary.csv"
    run:
        resfiles = " ".join([r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())])
        shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={output} {resfiles}")


rule summaryGroup:
    input:
        resfiles=expand("run/synth/group/{{d}}/chain_{chain}.csv",    chain=range(1, NCHAINS_GRP+1)),
        statuses=expand("run/synth/group/{{d}}/chain_{chain}.status", chain=range(1, NCHAINS_GRP+1))
    output: "run/synth/group/{d}/summary.csv"
    run:
        resfiles = " ".join([r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())])
        shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={output} {resfiles}")


rule postSolo:
    input:
        data="run/synth/solo/{d}/s{sim}/input/input.R",
        resfiles=expand("run/synth/solo/{{d}}/s{{sim}}/output/chain_{chain}.csv",    chain=range(1, NCHAINS_SOLO+1)),
        statuses=expand("run/synth/solo/{{d}}/s{{sim}}/output/chain_{chain}.status", chain=range(1, NCHAINS_SOLO+1)),
        summary="run/synth/solo/{d}/s{sim}/output/summary.csv",
        region_names="data/conn/region_names.vep.txt",
        ground_truth="run/synth/solo/{d}/s{sim}/groundTruth.json",
    output:
        "run/synth/solo/{d}/s{sim}/img/ct.pdf"
    run:
        resfiles = [r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())]
        if len(resfiles) == 0:
            open(output[0], 'w').close()
        else:
            util.post_solo(input.data, input.summary, resfiles, input.region_names, output[0],
                           ground_truth_file=input.ground_truth)


rule aggregSolo:
    input:
        lambda wildcards: [f"run/synth/solo/{{set}}/{{d}}/s{sim:03d}/output/summary.csv"
                           for sim in range(NSIMSOLO[wildcards.set])]
    output: "run/synth/solo/{set}/{d}/df.pkl"
    run: util.aggreg_synth_solo(input, wildcards.set, output[0])
    

rule joinSoloRandom:
    input: 
        expand("run/synth/solo/random/f{f}_nobs{nobs:03d}/df.pkl", f=FUNS.keys(), nobs=NOBSS)
    output:
        "run/synth/solo/random/df.pkl"
    run:
        df = pd.concat([pd.read_pickle(filename) for filename in input], ignore_index=True)
        pd.to_pickle(df, output[0])
        
rule joinSoloEz:
    input: 
        expand("run/synth/solo/ez/{config}/df.pkl", config=EZ_CONFIGS)
    output:
        "run/synth/solo/ez/df.pkl"
    run:
        df = pd.concat([pd.read_pickle(filename) for filename in input], ignore_index=True)
        pd.to_pickle(df, output[0])
