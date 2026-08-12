# Multi-GPU data generation timing benchmark
#
#   source scripts/hpc/polaris_env.sh
#   mpiexec -n 8 --ppn 4 --depth 8 --cpu-bind depth \
#       julia --project=$GP_ENV scripts/hpc/gpu_bench_mpi.jl --case ieee39 --M 512
#
# Each rank writes <out-dir>/gpu_scale_<case>_rank<RRRR>.json; aggregate with
#   julia --project=$GP_ENV scripts/hpc/gpu_scale_report.jl <out-dir>
#
# Options (defaults in brackets):
#   --case     ieee9 | ieee39                    [ieee39]
#   --M        scenarios per GPU                 [512]
#   --method   schur_cudss | shared | cpu        [schur_cudss]
#   --seconds  sustained load window per rank    [15.0]
#   --tfinal   simulated seconds                 [1.0]
#   --dt       backward-Euler step               [0.008333333333333333]
#   --out-dir  results directory                 [$BENCH_JSON_DIR]

using Pkg
haskey(ENV, "GP_ENV") && Pkg.activate(ENV["GP_ENV"])

using CUDA
using CUDSS
using Printf

include(joinpath(@__DIR__, "gpu_bench_common.jl"))

opts = parse_args(ARGS, Dict(
    "case"    => get(ENV, "GP_CASE", "ieee39"),
    "M"       => get(ENV, "GP_M", "512"),
    "method"  => get(ENV, "GP_METHODS", "schur_cudss"),
    "seconds" => "15.0",
    "tfinal"  => "1.0",
    "dt"      => string(1.0 / 120.0),
    "out-dir" => "",
))

rank, world, lrank = rank_size()
case    = opts["case"]
M       = parse(Int, opts["M"])
method  = opts["method"]
seconds = parse(Float64, opts["seconds"])
tfinal  = parse(Float64, opts["tfinal"])
dt      = parse(Float64, opts["dt"])

dev = select_device!(rank, lrank)
info = device_info()
host = gethostname()

if rank == 0
    println("=== GradPower multi-GPU scaling bench ===")
    println("world=$world  case=$case  method=$method  M/GPU=$M  window=$(seconds)s  tfinal=$tfinal")
    println("rank 0 on $host device[$dev] = $(info.gpu), CUDA $(info.cuda_runtime)")
    flush(stdout)
end

ps, dp = build_case(case)
r = bench_sustained(method, ps, dp, M; tfinal = tfinal, dt = dt, seconds = seconds)

@printf("[rank %4d/%d] %s dev%d  %6.1f scen/s  (%d calls / %.1fs)  %s\n",
        rank, world, host, dev, r.scen_per_s, r.calls, r.wall,
        r.ok ? "ok" : "FAIL: $(r.err)")
flush(stdout)

outdir = isempty(opts["out-dir"]) ?
    get(ENV, "BENCH_JSON_DIR", joinpath(REPO, "benchmarks", "results")) : opts["out-dir"]
meta = (; case, method, M, seconds, tfinal, dt,
        rank, world, local_rank = lrank, device = dev,
        host, jobid = get(ENV, "PBS_JOBID", ""), julia = string(VERSION), info...)
write_json(joinpath(outdir, @sprintf("gpu_scale_%s_rank%04d.json", case, rank)), meta, [r])
