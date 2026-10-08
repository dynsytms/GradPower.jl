# Single-GPU batch-size sweep to quantify polaris performance
#
#   source scripts/hpc/polaris_env.sh
#   julia --project=$GP_ENV scripts/hpc/gpu_bench_single.jl \
#         --case ieee39 --M 1,16,64,256,1024,2048 --methods schur_cudss,cpu
#
# Options (defaults in brackets):
#   --case     ieee9 | ieee39                      [ieee39]
#   --M        comma-separated batch sizes         [1,16,64,256,1024,2048]
#   --methods  schur_cudss | shared | cpu (csv)    [schur_cudss]
#   --solver   default | cudss | sds | klu         [default; env GP_SOLVER]
#              (sds = SparseDirectSolver.jl; default = CUDSS on GPU, KLU on CPU)
#   --tfinal   simulated seconds                   [1.0]
#   --dt       backward-Euler step                 [0.008333333333333333]
#   --reps     timed repetitions per point         [3]
#   --warmup   untimed repetitions per point       [1]
#   --cpu-max-M largest M to run on the CPU path   [256]
#   --out      results JSON                        [$BENCH_JSON_DIR/gpu_bench_single_<case>.json]


using Pkg
haskey(ENV, "GP_ENV") && Pkg.activate(ENV["GP_ENV"])

using CUDA
using CUDSS
# SparseDirectSolver.jl is optional: load it for `--solver sds`.
if get(ENV, "GP_SOLVER", "") == "sds" || any(a -> a == "sds" || a == "--solver=sds", ARGS)
    using SparseDirectSolver
end

include(joinpath(@__DIR__, "gpu_bench_common.jl"))

opts = parse_args(ARGS, Dict(
    "case"    => get(ENV, "GP_CASE", "ieee39"),
    "M"       => get(ENV, "GP_M", "1,16,64,256,1024,2048"),
    "solver"  => get(ENV, "GP_SOLVER", "default"),
    "methods" => get(ENV, "GP_METHODS", "schur_cudss"),
    "tfinal"  => "1.0",
    "dt"      => string(1.0 / 120.0),
    "reps"      => "3",
    "warmup"    => "1",
    "cpu-max-M" => "256",
    "out"       => "",
))

case    = opts["case"]
Ms      = parse_int_list(opts["M"])
methods = parse_str_list(opts["methods"])
tfinal  = parse(Float64, opts["tfinal"])
dt      = parse(Float64, opts["dt"])
reps    = parse(Int, opts["reps"])
warmup  = parse(Int, opts["warmup"])
cpu_max = parse(Int, opts["cpu-max-M"])
lsolver = linear_solver_backend(opts["solver"])

dev = select_device!()
info = device_info()
println("=== GradPower GPU batch sweep ===")
println("case=$case  solver=$(opts["solver"])  methods=$(join(methods, ","))  M=$(join(Ms, ","))  tfinal=$tfinal  dt=$dt  reps=$reps")
println("device[$dev] = $(info.gpu) ($(info.total_mem_gib > 0 ? round(info.total_mem_gib, digits=1) : 0) GiB), CUDA $(info.cuda_runtime)")

ps, dp = build_case(case)
println("sys_dim = $(length(dp.zvec))  (diff=$(ps.dynamic.diff_dim), alg=$(ps.dynamic.alg_dim))")
println()

print_header()
rows = []
skipped = Int[]
for m in methods, M in Ms
    if m == "cpu" && M > cpu_max
        push!(skipped, M)
        continue
    end
    r = bench_point(m, ps, dp, M; tfinal = tfinal, dt = dt, reps = reps, warmup = warmup,
                    linear_solver = lsolver)
    print_row(r)
    flush(stdout)
    push!(rows, r)
end
isempty(skipped) || println("(cpu skipped for M = $(join(skipped, ", ")) — above --cpu-max-M=$cpu_max)")

outdir = get(ENV, "BENCH_JSON_DIR", joinpath(REPO, "benchmarks", "results"))
out = isempty(opts["out"]) ? joinpath(outdir, "gpu_bench_single_$(case).json") : opts["out"]
meta = (; case, tfinal, dt, reps, warmup,
        sys_dim = length(dp.zvec),
        host = gethostname(),
        jobid = get(ENV, "PBS_JOBID", ""),
        julia = string(VERSION),
        info...)
write_json(out, meta, rows)

ok = [r for r in rows if r.ok]
if !isempty(ok)
    best = ok[argmax([r.scen_per_s_best for r in ok])]
    println()
    println("best: $(best.method) M=$(best.M) -> $(round(best.scen_per_s_best, digits=1)) scenarios/s")
end
