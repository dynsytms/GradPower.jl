# Aggregate the per-rank JSONs written by gpu_bench_mpi.jl into one summary
#
#   julia --project=$GP_ENV scripts/hpc/gpu_scale_report.jl [<results-dir>] [--ref 537]
#
#   <results-dir>  directory holding gpu_scale_*_rank*.json  [$BENCH_JSON_DIR]
#   --ref R        single-GPU reference rate (scen/s) for the efficiency column;
#                  defaults to the fastest rank in this run.

using Printf

args = copy(ARGS)
ref = nothing
i = findfirst(==("--ref"), args)
if i !== nothing
    ref = parse(Float64, args[i + 1])
    deleteat!(args, [i, i + 1])
end

dir = isempty(args) ? get(ENV, "BENCH_JSON_DIR",
                          joinpath(abspath(joinpath(@__DIR__, "..", "..")), "benchmarks", "results")) :
      args[1]
isdir(dir) || error("no such directory: $dir")

files = sort(filter(f -> occursin(r"^gpu_scale_.*_rank\d+\.json$", f), readdir(dir)))
isempty(files) && error("no gpu_scale_*_rank*.json in $dir")

num(txt, key) = (m = match(Regex("\"$key\":(-?[\\d.eE+]+|null)"), txt);
                 m === nothing || m[1] == "null" ? NaN : parse(Float64, m[1]))
str(txt, key) = (m = match(Regex("\"$key\":\"([^\"]*)\""), txt); m === nothing ? "" : String(m[1]))
flag(txt, key) = occursin("\"$key\":true", txt)

recs = map(files) do f
    t = read(joinpath(dir, f), String)
    (; rank = Int(num(t, "rank")), host = str(t, "host"), device = Int(num(t, "device")),
     gpu = str(t, "gpu"), case = str(t, "case"), method = str(t, "method"),
     M = Int(num(t, "M")), calls = Int(num(t, "calls")), wall = num(t, "wall"),
     scen_per_s = num(t, "scen_per_s"), ok = flag(t, "ok"))
end
sort!(recs, by = r -> r.rank)

ok = filter(r -> r.ok && isfinite(r.scen_per_s), recs)
isempty(ok) && error("every rank failed — check the job log")

first_rec = first(recs)
println("=== GradPower multi-GPU scaling: $(first_rec.case) / $(first_rec.method) / M=$(first_rec.M) per GPU ===")
println("$(length(recs)) rank(s), $(length(unique(r -> r.host, recs))) node(s), $(length(recs) - length(ok)) failed")
println()
@printf("%6s %-18s %4s %12s %8s %8s\n", "rank", "host", "dev", "scen/s", "calls", "wall(s)")
for r in recs
    @printf("%6d %-18s %4d %12.1f %8d %8.1f%s\n", r.rank, first(r.host, 18), r.device,
            r.scen_per_s, r.calls, r.wall, r.ok ? "" : "  FAIL")
end

rates = [r.scen_per_s for r in ok]
total = sum(rates)
reference = something(ref, maximum(rates))
n = length(ok)
println()
@printf("aggregate       : %.1f scen/s across %d GPU(s)\n", total, n)
@printf("per-GPU mean    : %.1f  (min %.1f, max %.1f, spread %.1f%%)\n",
        total / n, minimum(rates), maximum(rates),
        100 * (maximum(rates) - minimum(rates)) / maximum(rates))
@printf("speedup vs ref  : %.2fx  (reference %.1f scen/s%s)\n",
        total / reference, reference, ref === nothing ? ", = fastest rank" : ", --ref")
@printf("parallel eff.   : %.1f%%\n", 100 * total / (n * reference))

hosts = unique(r.host for r in ok)
if length(hosts) > 1
    println()
    for h in hosts
        rs = [r.scen_per_s for r in ok if r.host == h]
        @printf("  %-18s %2d GPU  %10.1f scen/s\n", h, length(rs), sum(rs))
    end
end
