#!python3

from vresutils.benchmark import timer
import pandas as pd
import sys
from glob import glob
from shutil import copyfile

fn, timefn, mproffn = sys.argv[1:4]

formulation ="kirchhoff"
solver = "gurobi"
solver_options = {"Method": 2, "Crossover": 0, "BarConvTol": 1e-5, 'LogToConsole': False, "Threads": 2}

times = pd.Series()

with timer("import modules") as t:
    import pypsa
times["import"] = t.usec

with timer("loading network") as t:
    n = pypsa.Network(fn)
times["loading"] = t.usec

with timer("building pyomo model") as t:
    pypsa.opf.network_lopf_build_model(n, formulation="kirchhoff")
times["building"] = t.usec

with timer("solving model") as t:
    with timer("prepare solver"):
        pypsa.opf.network_lopf_prepare_solver(n, solver)
    with timer("solving"):
        pypsa.opf.network_lopf_solve(n, formulation=formulation, solver_options=solver_options)
times["solving"] = t.usec

df = pd.DataFrame(dict(step=times.index, secs=times.values/1e6))
df.to_csv(timefn, index=False)

copyfile(glob("mprofile_*")[0], mproffn)
