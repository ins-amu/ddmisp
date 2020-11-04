#-*- mode: snakemake -*-

import itertools

import util


configfile: "config.yml"


ONSETS = ["INC"]
ATLASES = ["vep"]
SIDS = list(range(1, 51))
FOLDS = [list(range(1, 26)), list(range(26, 51))]
NFOLDS = len(FOLDS)
NCHAINS_MULTI = 4
NCHAINS_SINGLE = 2

CMDSTANDIR = config['CMDSTANDIR']


localrules: compile, prepLearn, checkChain, summaryLearn, postLearn, prepSolo, summarySolo, postSolo,
            allLearn, allPrepSolo, allPostSolo, all


rule compile:
    input: "stan/{name}.stan"
    output: "stan/{name}"
    shell: "dir=`pwd`; cd {CMDSTANDIR}; make $dir/stan/{wildcards.name}"


rule prepLearn:
    input:
        conn=expand("data/conn/{{atlas}}/id{sid:03d}.txt", sid=SIDS),
        onsets=expand("data/onset_regions/{{onset}}/{{atlas}}/id{sid:03d}.json", sid=SIDS)
    output: "run/learn/fold{fold}/{onset}/{atlas}/input/learn.R"
    run:
        sids = FOLDS[int(wildcards.fold)]
        conns = [input.conn[SIDS.index(sid)] for sid in sids]
        onsets = [input.onsets[SIDS.index(sid)] for sid in sids]
        util.prep_learn(conns, onsets, wildcards.atlas, output[0])


rule runLearn:
    input:
        bin="stan/msinf",
        data="run/learn/fold{fold}/{onset}/{atlas}/input/learn.R"
    output:  "run/learn/fold{fold}/{onset}/{atlas}/output/chain_{chain}.csv"
    log:     "run/learn/fold{fold}/{onset}/{atlas}/log/chain_{chain}.txt"
    threads: 8
    shell:
        """
        export STAN_NUM_THREADS={threads}
        ./{input.bin} sample                     \
            num_warmup=500 num_samples=500       \
            random seed=42 id={wildcards.chain}  \
            data file={input.data}               \
            output file={output[0]} &> {log}
        """


rule checkChain:
    input: "{d}/chain_{chain}.csv"
    output: "{d}/chain_{chain}.status"
    run:
        summfile = f"{wildcards.d}/summary_chain_{wildcards.chain}.csv"
        shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={summfile} {input[0]}")
        util.check_chain(summfile, output[0])


rule summaryLearn:
    input:
        resfiles=expand("run/learn/fold{{fold}}/{{onset}}/{{atlas}}/output/chain_{chain}.csv",    chain=range(1, NCHAINS_MULTI+1)),
        statuses=expand("run/learn/fold{{fold}}/{{onset}}/{{atlas}}/output/chain_{chain}.status", chain=range(1, NCHAINS_MULTI+1))
    output: "run/learn/fold{fold}/{onset}/{atlas}/output/summary.csv"
    run:
        resfiles = " ".join([r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())])
        shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={output} {resfiles}")



rule postLearn:
    input:
        data="run/learn/fold{fold}/{onset}/{atlas}/input/learn.R",
        summary="run/learn/fold{fold}/{onset}/{atlas}/output/summary.csv",
        resfiles=expand("run/learn/fold{{fold}}/{{onset}}/{{atlas}}/output/chain_{chain}.csv",    chain=range(1, NCHAINS_MULTI+1)),
        statuses=expand("run/learn/fold{{fold}}/{{onset}}/{{atlas}}/output/chain_{chain}.status", chain=range(1, NCHAINS_MULTI+1)),
        region_names="data/conn/region_names.{atlas}.txt"
    output:
        "run/learn/fold{fold}/{onset}/{atlas}/img/params.pdf",
        directory("run/learn/fold{fold}/{onset}/{atlas}/img/ct")
    run:
        resfiles = [r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())]
        util.post_learn(input.data, input.summary, resfiles, input.region_names, output[0], output[1])


def fold_resfiles(wildcards):
    fold = '0' if int(wildcards.sid) in FOLDS[1] else '1'
    return [f"run/learn/fold{fold}/{wildcards.onset}/{wildcards.atlas}/output/chain_{ch}.csv" for ch in range(1, NCHAINS_MULTI+1)]

def fold_statuses(wildcards):
    fold = '0' if int(wildcards.sid) in FOLDS[1] else '1'
    return [f"run/learn/fold{fold}/{wildcards.onset}/{wildcards.atlas}/output/chain_{ch}.status" for ch in range(1, NCHAINS_MULTI+1)]


checkpoint prepSolo:
    input:
        conn="data/conn/{atlas}/id{sid}.txt",
        onset="data/onset_regions/{onset}/{atlas}/id{sid}.json",
        resfiles=fold_resfiles,
        statuses=fold_statuses
    output:
        direc=directory("run/solo/{onset}/{atlas}/id{sid}/input")
    run:
        resfiles = [r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())]
        util.prep_solo(input.conn, input.onset, resfiles, output.direc, create_loo_files=True)


rule runSolo:
    input:
        bin="stan/ssinf",
        data="run/solo/{onset}/{atlas}/id{sid}/input/{case}.R"
    output:  "run/solo/{onset}/{atlas}/id{sid}/output/{case}/chain_{chain}.csv"
    log:     "run/solo/{onset}/{atlas}/id{sid}/log/{case}/chain_{chain}.txt"
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


rule summarySolo:
    input:
        resfiles=expand("run/solo/{{d}}/chain_{chain}.csv",    chain=range(1, NCHAINS_SINGLE+1)),
        statuses=expand("run/solo/{{d}}/chain_{chain}.status", chain=range(1, NCHAINS_SINGLE+1))
    output: "run/solo/{d}/summary.csv"
    run:
        resfiles = " ".join([r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())])
        if len(resfiles) == 0:
            open(output[0], 'w').close()
        else:
            shell("{CMDSTANDIR}/bin/stansummary --sig_figs=3 --csv_file={output} {resfiles}")


rule postSolo:
    input:
        data="run/solo/{onset}/{atlas}/id{sid}/input/{case}.R",
        summary="run/solo/{onset}/{atlas}/id{sid}/output/{case}/summary.csv",
        resfiles=expand("run/solo/{{onset}}/{{atlas}}/id{{sid}}/output/{{case}}/chain_{chain}.csv",    chain=range(1, NCHAINS_SINGLE+1)),
        statuses=expand("run/solo/{{onset}}/{{atlas}}/id{{sid}}/output/{{case}}/chain_{chain}.status", chain=range(1, NCHAINS_SINGLE+1)),
        region_names="data/conn/region_names.{atlas}.txt",
        ez_file="data/ei-final.json"
    output: "run/solo/{onset}/{atlas}/id{sid}/img/ct_{case}.pdf"
    run:
        resfiles = [r for r, s in zip(input.resfiles, input.statuses) if int(open(s).read().strip())]
        if len(resfiles) == 0:
            open(output[0], 'w').close()
        else:
            util.post_solo(input.data, input.summary, resfiles, input.region_names, output[0], input.ez_file)


# Targets

rule allLearn:
    input: expand("run/learn/fold{fold}/{onset}/{atlas}/img/params.pdf", fold=range(NFOLDS), onset=ONSETS, atlas=ATLASES)


rule allPrepSolo:
    input: expand("run/solo/{onset}/{atlas}/id{sid:03d}/input", onset=ONSETS, atlas=ATLASES, sid=SIDS)


def allPostSoloInput(wildcards):
    s = []
    for onset, atlas, sid in itertools.product(ONSETS, ATLASES, SIDS):
        sid = f"{sid:03d}"
        direc = checkpoints.prepSolo.get(onset=onset, atlas=atlas, sid=sid).output.direc
        s.extend(expand(f"run/solo/{onset}/{atlas}/id{sid}/img/ct_{{case}}.pdf",
                        case=glob_wildcards(os.path.join(direc, "{case}.R"))[0]))

    return s


rule allPostSolo:
    input: allPostSoloInput


def aggregLooInput(wildcards):
    s = []
    for onset, atlas, sid in itertools.product(ONSETS, ATLASES, SIDS):
        sid = f"{sid:03d}"
        direc = checkpoints.prepSolo.get(onset=onset, atlas=atlas, sid=sid).output.direc
        s.extend(expand(f"run/solo/{onset}/{atlas}/id{sid}/output/{{case}}/summary.csv",
                        case=glob_wildcards(os.path.join(direc, "{case}.R"))[0]))

    return s


rule aggregLoo:
    input: aggregLooInput
    output: "run/solo/df-loo.pkl"
    run: util.aggreg_loo(input, output[0])


def virtualResectionInput(wildcards):
    s = []
    for onset, atlas, sid in itertools.product(ONSETS, ATLASES, SIDS):
        sid = f"{sid:03d}"
        direc = checkpoints.prepSolo.get(onset=onset, atlas=atlas, sid=sid).output.direc
        s.extend(expand(f"run/solo/{onset}/{atlas}/id{sid}/output/r{{rid}}_all/summary.csv",
                        rid=glob_wildcards(os.path.join(direc, "r{rid}_all.R"))[0]))
    return s

rule virtualResection:
    input: 
        surgeries="data/surgeries.vep.json",
        region_names="data/conn/region_names.vep.txt",
        summaries=virtualResectionInput,
    output: "run/solo/df-virtual-resection.pkl"
    run: util.virtual_resection(input.surgeries, input.region_names, input.summaries, output[0])


rule all:
    input:
        learn=expand("run/learn/fold{fold}/{onset}/{atlas}/img/params.pdf", fold=range(NFOLDS), onset=ONSETS, atlas=ATLASES),
        solo=allPostSoloInput,
        loo="run/solo/df-loo.pkl",
        resection="run/solo/df-virtual-resection.pkl"
        
