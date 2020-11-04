

CHAINS = [1,2]
SEIZURES = glob_wildcards("run/solo/INC/vep/id{sid}/output/r{rid}_all/")
SUBJECT_DIR = "/home/vep/RetrospectivePatients/1-Processed"

rule seizure_plots:
    input: expand("figs/solo/INC/id{sid}_r{rid}_all.pdf", zip, sid=SEIZURES.sid, rid=SEIZURES.rid)


rule plot_seizure:
    input:
        data="run/solo/{onset}/vep/id{sid}/input/r{rid}_{regs}.R",
        region_names="data/conn/region_names.vep.txt",
        resfiles=expand("run/solo/{{onset}}/vep/id{{sid}}/output/r{{rid}}_{{regs}}/chain_{ch}.csv", ch=CHAINS),
        statuses=expand("run/solo/{{onset}}/vep/id{{sid}}/output/r{{rid}}_{{regs}}/chain_{ch}.status", ch=CHAINS),
        surgeries="data/surgeries.vep.json"
    output: "figs/solo/{onset}/id{sid}_r{rid}_{regs}.pdf"
    shell:
        "util/vfb.sh python scripts/plot_compact.py one"
        "    {wildcards.sid} {input.data} '{input.resfiles}' '{input.statuses}'"
        f"    {{input.surgeries}} {SUBJECT_DIR} {{input.region_names}} {{output}}"


rule examples:
    input:
    output: directory("figs/examples")
    shell:
        "util/vfb.sh python scripts/plot_examples.py {output}"
