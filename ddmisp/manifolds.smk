

funs = ['A', 'C', 'D']

rule all:
    input: "figs/manifolds.pdf"


rule build:
    input:
    output: "run/manifolds/manifold{fun}.npz"
    shell: "python scripts/manifolds.py build {wildcards.fun} {output}"


rule plot:
    input:
        manifolds=expand("run/manifolds/manifold{fun}.npz", fun=funs),
        sketch="etc/network-3nodes.png"
    output: "img/manifolds.pdf"
    shell: "util/vfb.sh python scripts/manifolds.py plot {input.sketch} '{funs}' '{input}' {output}"
