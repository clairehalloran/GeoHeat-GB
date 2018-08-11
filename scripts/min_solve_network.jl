#!julia

using DataStructures, DataFrames, CSV, Base.Filesystem

fn, timefn, mproffn = ARGS

gurobi_options = Dict(:Method=>2, :Crossover=>0, :BarConvTol=>1e-5, :LogToConsole=>false, :Threads=>2)

times = OrderedDict()

let m
    times["import"] = @elapsed using EnergyModels, Gurobi
    times["loading"] = @elapsed m = EnergyModel(fn, solver=GurobiSolver(; gurobi_options...))
    times["building"] = @elapsed build(m)
    times["solving"] = @elapsed solve(m)
end # allow gc of m

let m
    times["loading 2"] = @elapsed m = EnergyModel(fn, solver=GurobiSolver(; gurobi_options...))
    times["building 2"] = @elapsed build(m)
    times["solving 2"] = @elapsed solve(m)
end

df = DataFrame(:step=>collect(keys(times)), :secs=>collect(values(times)))
CSV.write(timefn, df)

cp(first(filter(x->startswith(x, "mprofile"), readdir())), mproffn)
