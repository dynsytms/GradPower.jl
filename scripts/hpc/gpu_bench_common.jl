# Shared helpers for the Polaris GPU throughput benchmarks
# (gpu_bench_single.jl, gpu_bench_mpi.jl).
#


using GradPower
using Printf

const REPO = get(ENV, "GRADPOWER_DIR", abspath(joinpath(@__DIR__, "..", "..")))
const EXDIR = joinpath(REPO, "examples")

# name => (raw, dyr, default fault bus)
const CASES = Dict(
    "ieee9"  => ("ieee9_v33.raw", "ieee9bus_gov.dyr", 1),
    "ieee39" => ("IEEE39.raw",    "IEEE39_gov.dyr",   16),
)

"""
    build_case(name; fault_bus=nothing, rfault=0.02, ton=0.1, toff=0.2)

Parse → power flow → dynamics init → schedule one bus fault. Returns
`(ps, dp)`. Mirrors the setup used in `test/test_gpu_backend.jl`
"""
function build_case(name::AbstractString; fault_bus = nothing, rfault = 0.02,
                    ton = 0.1, toff = 0.2)
    haskey(CASES, name) || error("unknown case '$name'; have $(sort(collect(keys(CASES))))")
    raw, dyr, defbus = CASES[name]
    ps = from_psse(joinpath(EXDIR, raw), joinpath(EXDIR, dyr))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad
            d.dtype.α = 0.5
        end
    end
    dp = DynamicProblem(ps)
    initialize_dynamics!(dp, ps)
    add_event!(ps, ContingencyEvent(Int(something(fault_bus, defbus)),
                                    Float64(rfault), Float64(ton), Float64(toff)))
    return ps, dp
end


function _gpu_mem_used(ext)
    ext === nothing && return 0
    try
        C = ext.CUDA
        if isdefined(C, :free_memory)
            return Int(C.total_memory() - C.free_memory())
        elseif isdefined(C, :available_memory)
            return Int(C.total_memory() - C.available_memory())
        elseif isdefined(C, :memory_info)
            free, total = C.memory_info()
            return Int(total - free)
        end
    catch
    end
    return 0
end

_gpu_reclaim(ext) = ext === nothing ? nothing : (try; ext.CUDA.reclaim(); catch; end; nothing)

"""
    bench_point(method, ps, dp, M; tfinal, dt, reps, warmup) -> NamedTuple

`method` is one of `"schur_cudss"`, `"shared"`, or `"cpu"`

The batch layout is rebuilt outside the timed region for every rep, so the
timing covers integration only.
"""
function bench_point(method::AbstractString, ps, dp, M::Int;
                     tfinal = 1.0, dt = 1.0 / 120.0, reps = 3, warmup = 1)
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    gpu = method != "cpu"
    gpu && ext === nothing && error("GradPowerCUDAExt not loaded — cannot run method '$method'")

    make_layout, run! = if method == "schur_cudss"
        (() -> ext.GpuBatchedLayout(dp, ps, M)),
        ((bl) -> ext.integrate_gpu_schur_cudss!(bl, ps, tfinal; dt = dt))
    elseif method == "shared"
        (() -> ext.GpuBatchedLayout(dp, ps, M)),
        ((bl) -> ext.integrate_gpu_shared!(bl, ps, tfinal; dt = dt))
    elseif method == "cpu"
        (() -> GradPower.BatchedLayout(dp, ps, M)),
        ((bl) -> GradPower.integrate_batched!(bl, ps, tfinal; dt = dt))
    else
        error("unknown method '$method' (schur_cudss | shared | cpu)")
    end

    sync = gpu ? (() -> ext.CUDA.synchronize()) : (() -> nothing)
    mem_used = () -> gpu ? _gpu_mem_used(ext) : 0

    try
        for _ in 1:warmup
            run!(make_layout())
            sync()
        end
        gpu && _gpu_reclaim(ext)
        mem0 = mem_used()

        times = Float64[]
        mem_peak = mem0
        for _ in 1:reps
            bl = make_layout()
            sync()
            t0 = time_ns()
            run!(bl)
            sync()
            push!(times, (time_ns() - t0) / 1e9)
            mem_peak = max(mem_peak, mem_used())
            bl = nothing
            gpu && _gpu_reclaim(ext)
        end

        best = minimum(times)
        mean = sum(times) / length(times)
        return (; method, M, reps, s_per_call_best = best, s_per_call_mean = mean,
                scen_per_s_best = M / best, scen_per_s_mean = M / mean,
                gpu_mem_bytes = max(0, mem_peak - mem0), ok = true, err = "")
    catch e
        msg = sprint(showerror, e)
        @warn "method=$method M=$M failed" exception = msg
        return (; method, M, reps, s_per_call_best = NaN, s_per_call_mean = NaN,
                scen_per_s_best = NaN, scen_per_s_mean = NaN,
                gpu_mem_bytes = 0, ok = false, err = first(msg, 200))
    end
end

"""
    bench_sustained(method, ps, dp, M; tfinal, dt, seconds, warmup) -> NamedTuple

"""
function bench_sustained(method::AbstractString, ps, dp, M::Int;
                         tfinal = 1.0, dt = 1.0 / 120.0, seconds = 15.0, warmup = 1)
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    gpu = method != "cpu"
    gpu && ext === nothing && error("GradPowerCUDAExt not loaded — cannot run method '$method'")

    make_layout, run! = if method == "schur_cudss"
        (() -> ext.GpuBatchedLayout(dp, ps, M)),
        ((bl) -> ext.integrate_gpu_schur_cudss!(bl, ps, tfinal; dt = dt))
    elseif method == "shared"
        (() -> ext.GpuBatchedLayout(dp, ps, M)),
        ((bl) -> ext.integrate_gpu_shared!(bl, ps, tfinal; dt = dt))
    elseif method == "cpu"
        (() -> GradPower.BatchedLayout(dp, ps, M)),
        ((bl) -> GradPower.integrate_batched!(bl, ps, tfinal; dt = dt))
    else
        error("unknown method '$method' (schur_cudss | shared | cpu)")
    end
    sync = gpu ? (() -> ext.CUDA.synchronize()) : (() -> nothing)

    try
        for _ in 1:warmup
            run!(make_layout())
            sync()
        end
        gpu && _gpu_reclaim(ext)

        calls = 0
        t0 = time_ns()
        while (time_ns() - t0) / 1e9 < seconds
            run!(make_layout())
            sync()
            calls += 1
            gpu && _gpu_reclaim(ext)
        end
        wall = (time_ns() - t0) / 1e9
        return (; method, M, calls, wall, seconds = Float64(seconds),
                scen_per_s = M * calls / wall, s_per_call = wall / calls,
                ok = true, err = "")
    catch e
        msg = sprint(showerror, e)
        @warn "method=$method M=$M failed" exception = msg
        return (; method, M, calls = 0, wall = 0.0, seconds = Float64(seconds),
                scen_per_s = NaN, s_per_call = NaN, ok = false, err = first(msg, 200))
    end
end

# Reporting format starts here
function print_header()
    @printf("%-13s %7s %13s %13s %11s  %s\n",
            "method", "M", "s/call(best)", "scen/s(best)", "GPU mem", "status")
end

function print_row(r)
    @printf("%-13s %7d %13.4f %13.1f %10.1fM  %s\n",
            r.method, r.M, r.s_per_call_best, r.scen_per_s_best,
            r.gpu_mem_bytes / 2^20, r.ok ? "ok" : "FAIL: $(r.err)")
end

_json(x::AbstractString) = '"' * replace(String(x), '\\' => "\\\\", '"' => "\\\"",
                                         '\n' => "\\n") * '"'
_json(x::Bool) = x ? "true" : "false"
_json(x::Symbol) = _json(String(x))
_json(x::Real) = isfinite(x) ? string(x) : "null"
_json(x::AbstractVector) = "[" * join(_json.(x), ",") * "]"
_json(x::NamedTuple) = "{" * join([_json(String(k)) * ":" * _json(v)
                                   for (k, v) in pairs(x)], ",") * "}"
_json(x::AbstractDict) = "{" * join([_json(String(k)) * ":" * _json(v)
                                     for (k, v) in x], ",") * "}"

"""
    write_json(path, meta::NamedTuple, rows::Vector) -> path

`{"meta": {...}, "rows": [{...}, ...]}`. Non-finite numbers become `null`.
"""
function write_json(path::AbstractString, meta, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        write(io, "{", _json("meta"), ":", _json(meta), ",",
              _json("rows"), ":[", join(_json.(rows), ","), "]}\n")
    end
    @info "wrote $path"
    return path
end

"""
    device_info() -> NamedTuple

Name / capability / memory of the currently selected CUDA device, or a
CPU-only stub when the extension is absent.
"""
function device_info()
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    (ext === nothing || !ext.CUDA.functional()) &&
        return (; gpu = "none", capability = "", total_mem_gib = 0.0, cuda_runtime = "")
    d = ext.CUDA.device()
    return (; gpu = ext.CUDA.name(d),
            capability = string(ext.CUDA.capability(d)),
            total_mem_gib = ext.CUDA.totalmem(d) / 2^30,
            cuda_runtime = string(ext.CUDA.runtime_version()))
end

"""
    select_device!(rank, local_rank) -> Int

Bind this process to one GPU set by `GP_GPU`, otherwise uses the launcher's
local rank modulo the visible device count
"""
function select_device!(rank::Int = 0, local_rank::Int = 0)
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    (ext === nothing || !ext.CUDA.functional()) && return -1
    ndev = length(collect(ext.CUDA.devices()))
    idx = haskey(ENV, "GP_GPU") ? parse(Int, ENV["GP_GPU"]) : mod(local_rank, ndev)
    ext.CUDA.device!(idx)
    return idx
end

"""
    rank_size() -> (rank, world, local_rank)

0-based rank/world/local-rank from the launcher's environment. Falls back to a serial `(0, 1, 0)`.
"""
function rank_size()
    rank = world = nothing
    for (rk, sk) in (("PMI_RANK", "PMI_SIZE"), ("PALS_RANKID", "PALS_NRANKS"),
                     ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                     ("SLURM_PROCID", "SLURM_NTASKS"), ("RANK", "WORLD_SIZE"))
        if haskey(ENV, rk)
            rank = parse(Int, ENV[rk])
            world = haskey(ENV, sk) ? parse(Int, ENV[sk]) : 1
            break
        end
    end
    rank === nothing && return (0, 1, 0)
    lrank = 0
    for k in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
              "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID")
        if haskey(ENV, k)
            lrank = parse(Int, ENV[k])
            break
        end
    end
    return (rank, world, lrank)
end

"""
    parse_args(argv, defaults::Dict{String,String}) -> Dict{String,String}

Minimal `--key value` / `--key=value` parser
"""
function parse_args(argv, defaults::Dict{String,String})
    opts = copy(defaults)
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument '$a'")
        key, val = if occursin('=', a)
            k, v = split(a[3:end], '=', limit = 2)
            (String(k), String(v))
        else
            i += 1
            i <= length(argv) || error("--$(a[3:end]) needs a value")
            (a[3:end], String(argv[i]))
        end
        haskey(opts, key) || error("unknown option --$key; known: $(sort(collect(keys(opts))))")
        opts[key] = val
        i += 1
    end
    return opts
end

parse_int_list(s::AbstractString) = [parse(Int, strip(x)) for x in split(s, ',') if !isempty(strip(x))]
parse_str_list(s::AbstractString) = [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]
