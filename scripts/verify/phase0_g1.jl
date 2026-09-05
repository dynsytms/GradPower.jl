#!/usr/bin/env julia
# Phase 0 G1 — Baseline captured.
#
# Runs the existing CPU dynamics path on every Phase 0 baseline case
# (2-bus, IEEE-9, IEEE-39, ACTIVSg200, ACTIVSg2000, ACTIVSg70k) at
# the production step h, single-sim M=1, and records:
#   - wall-clock per case,
#   - hardware spec (CPU model, RAM, OS, Julia version, Manifest SHA),
#   - SHA-256 of each PSS/E reference .npz used.
#
# Per phase-0-baseline.md#G1:
#   pass iff every case in the list has a cases[<name>] entry in
#   baseline.json with non-null wallclock_s AND a non-null
#   reference_sha256. Cases that fail to run record status: "failed"
#   with an error message — that is acceptable for Phase 0, the
#   script still exits 0.
#
# Out of scope: any solver/kernel work. This is pure measurement.
#
# Note: phase-0-baseline.md cites "scratch/cases/" and "scratch/results/".
# The actual on-disk locations in this repo are examples/ + uqgrid/data
# and examples/refs/ — recorded as an open question in the builder report.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using NPZ
using SHA
using Dates
using Printf

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_PATH = joinpath(REPO, "artifacts", "phase0", "baseline.json")
mkpath(dirname(OUT_PATH))

# ---------------- JSON helpers (no JSON.jl dep) ----------------
function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"' || c == '\\'; print(buf, '\\', c)
        elseif c == '\n'; print(buf, "\\n")
        elseif c == '\r'; print(buf, "\\r")
        elseif c == '\t'; print(buf, "\\t")
        elseif UInt32(c) < 0x20; print(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else; print(buf, c)
        end
    end
    String(take!(buf))
end
function json_value(v)
    if v === nothing; return "null"
    elseif v isa Bool; return v ? "true" : "false"
    elseif v isa AbstractString; return string('"', _json_escape(v), '"')
    elseif v isa Integer; return string(v)
    elseif v isa AbstractFloat; return isfinite(v) ? string(v) : "null"
    elseif v isa AbstractVector; return string("[", join(json_value.(v), ","), "]")
    elseif v isa AbstractDict
        ks = sort(collect(keys(v)), by=string); parts = String[]
        for k in ks
            push!(parts, string('"', _json_escape(String(string(k))), '"', ":", json_value(v[k])))
        end
        return string("{", join(parts, ","), "}")
    else; return string('"', _json_escape(string(v)), '"')
    end
end
function pretty(json::AbstractString)
    buf = IOBuffer(); indent = 0; in_string = false; prev = '\0'
    for c in json
        if in_string
            print(buf, c); (c == '"' && prev != '\\') && (in_string = false)
        else
            if c == '"'; in_string = true; print(buf, c)
            elseif c == '{' || c == '['; print(buf, c, '\n'); indent += 2; print(buf, ' '^indent)
            elseif c == '}' || c == ']'; indent -= 2; print(buf, '\n', ' '^indent, c)
            elseif c == ','; print(buf, c, '\n', ' '^indent)
            elseif c == ':'; print(buf, c, ' ')
            else; print(buf, c)
            end
        end
        prev = c
    end
    String(take!(buf))
end

function sha256_file(path::AbstractString)
    open(path, "r") do io; bytes2hex(SHA.sha256(io)); end
end

function git_sha()
    try; return strip(read(Cmd(`git rev-parse HEAD`, dir=REPO), String))
    catch; return ""; end
end

function detect_hardware()
    cpu_model = ""
    ram_bytes = 0
    if Sys.isapple()
        try cpu_model = strip(read(`sysctl -n machdep.cpu.brand_string`, String)) catch end
        try ram_bytes = parse(Int, strip(read(`sysctl -n hw.memsize`, String))) catch end
    elseif Sys.islinux()
        try
            for ln in eachline("/proc/cpuinfo")
                if startswith(ln, "model name")
                    cpu_model = strip(split(ln, ':', limit=2)[2]); break
                end
            end
        catch end
        try
            for ln in eachline("/proc/meminfo")
                if startswith(ln, "MemTotal:")
                    parts = split(ln); ram_bytes = parse(Int, parts[2]) * 1024; break
                end
            end
        catch end
    end
    os_str = ""
    try os_str = strip(read(`uname -srm`, String)) catch end
    manifest_path = joinpath(REPO, "Manifest.toml")
    manifest_sha = isfile(manifest_path) ? sha256_file(manifest_path) : ""
    return Dict{String,Any}(
        "cpu_model" => cpu_model,
        "ram_bytes" => ram_bytes,
        "os" => os_str,
        "julia_version" => string(VERSION),
        "manifest_sha256" => manifest_sha,
        "hostname" => gethostname(),
        "n_threads" => Threads.nthreads(),
    )
end

# ---------------- Case table ----------------
# Each baseline case records: raw, dyr, reference path (or nothing),
# production h, tend, fault parameters. dt/tend/fault are read from the
# reference .npz when present (the validated production values uqgrid
# generated each .npz with); when no .npz exists we still record the
# wall-clock for a reproducible default run.
struct Case
    name::String
    raw::String
    dyr::String
    reference::Union{Nothing,String}
    fault_bus::Int          # 1-based Julia bus index
    rfault::Float64
    ton::Float64
    toff::Float64
    dt::Float64
    tend::Float64
    set_zipload_alpha::Bool
    add_static_gen_stubs::Bool
end

const CASES = [
    Case("2bus",
         joinpath(REPO, "examples", "2bus.raw"),
         joinpath(REPO, "examples", "2bus.dyr"),
         joinpath(REPO, "examples", "refs", "2bus_genrou.npz"),
         1, 0.02, 0.1, 0.2, 1.0/120.0, 2.0, false, false),
    Case("ieee9",
         joinpath(REPO, "examples", "ieee9_v33.raw"),
         joinpath(REPO, "examples", "ieee9bus.dyr"),
         joinpath(REPO, "examples", "refs", "ieee9_nogov.npz"),
         7, 0.02, 0.2, 0.3, 1.0/120.0, 5.0, true, false),
    Case("ieee39",
         joinpath(REPO, "examples", "IEEE39.raw"),
         joinpath(REPO, "examples", "IEEE39.dyr"),
         joinpath(REPO, "examples", "refs", "ieee39_gov.npz"),
         1, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true, false),
    Case("activs200",
         joinpath(REPO, "uqgrid", "data", "ACTIVSg200.raw"),
         joinpath(REPO, "uqgrid", "data", "ACTIVSg200.dyr"),
         joinpath(REPO, "examples", "refs", "activs200.npz"),
         0, 0.0, 0.0, 0.0, 0.0, 0.0, true, false),  # fault/dt/tend read from ref
    Case("activs2000",
         joinpath(REPO, "uqgrid", "data", "ACTIVSg2000.raw"),
         joinpath(REPO, "uqgrid", "data", "ACTIVSg2000_genrou_only.dyr"),
         joinpath(REPO, "examples", "refs", "activs2000.npz"),
         0, 0.0, 0.0, 0.0, 0.0, 0.0, true, true),
    Case("activs70k",
         joinpath(REPO, "uqgrid", "data", "ACTIVSg70k.raw"),
         joinpath(REPO, "uqgrid", "benchmarks", "phase3_generated_dyrs",
                  "ACTIVSg70k_genrou_only.dyr"),
         joinpath(REPO, "examples", "refs", "activs70k.npz"),
         0, 0.0, 0.0, 0.0, 0.0, 0.0, true, true),
]

function run_case(c::Case)
    # Pure measurement: load case, init dynamics, integrate with fault,
    # record wall time. We do not compare to reference here (that is the
    # regression suite's job). Cases that throw record status="failed".
    record = Dict{String,Any}(
        "name" => c.name,
        "raw" => c.raw,
        "dyr" => c.dyr,
        "reference_path" => c.reference,
        "reference_sha256" => c.reference === nothing ? nothing : (isfile(c.reference) ? sha256_file(c.reference) : nothing),
        "status" => "pending",
        "wallclock_s" => nothing,
        "n_steps" => nothing,
        "h" => nothing,
        "M" => 1,
        "fault_bus_external" => nothing,
        "error" => nothing,
    )
    t_case_start = time()
    try
        ps = from_psse(c.raw, c.dyr; add_static_gen_stubs = c.add_static_gen_stubs)
        GradPower.build_network!(ps)
        GradPower.runpf!(ps)
        if c.set_zipload_alpha
            for dev in ps.dynamic.devices
                if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
            end
        end
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)

        # Determine fault parameters and (dt, tend).
        fault_bus_jl = c.fault_bus
        rfault = c.rfault; ton = c.ton; toff = c.toff
        dt = c.dt; tend = c.tend
        fault_bus_ext = 0
        if c.reference !== nothing && isfile(c.reference)
            ref = npzread(c.reference)
            if haskey(ref, "fault_bus")
                fb = Int(ref["fault_bus"][])
                ext2jl = Dict(Int64(b.i) => i for (i, b) in enumerate(ps.buses))
                # Some reference .npz files store fault_bus as the
                # external PSSE bus number; the small 2bus reference
                # stores it as the (Python 0-based) internal index.
                # Translate by external bus first; fall back to the
                # case's hard-coded 1-based Julia index if not found.
                if haskey(ext2jl, Int64(fb))
                    fault_bus_jl = ext2jl[Int64(fb)]
                    fault_bus_ext = fb
                else
                    fault_bus_jl = c.fault_bus
                    fault_bus_ext = Int(ps.buses[fault_bus_jl].i)
                end
                rfault = Float64(ref["rfault"][]); ton = Float64(ref["ton"][])
                toff = Float64(ref["toff"][]);    dt   = Float64(ref["dt"][])
                tend = Float64(ref["tend"][])
            end
        end
        if fault_bus_ext == 0
            fault_bus_ext = Int(ps.buses[fault_bus_jl].i)
        end

        GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_jl, rfault, ton, toff))
        # warm-up: a tiny solve to amortize first-call JIT cost; without
        # this the first case's wall time is dominated by JIT, not solve.
        # Use 2 steps so it is measurable but negligible.
        t_warm = time()
        _ = GradPower.integrate!(deepcopy(dp), deepcopy(ps), 2*dt; dt=dt, verbose=false)
        _ = time() - t_warm

        t0 = time()
        tvec, traj = GradPower.integrate!(dp, ps, tend; dt=dt, verbose=false)
        wall = time() - t0

        record["status"] = "ok"
        record["wallclock_s"] = wall
        record["n_steps"] = size(traj, 2)
        record["h"] = dt
        record["fault_bus_external"] = fault_bus_ext
        record["tend"] = tend
        record["n_diff"] = ps.dynamic.diff_dim
        record["n_alg"]  = ps.dynamic.alg_dim
        record["n_bus"]  = length(ps.buses)

    catch e
        record["status"] = "failed"
        record["error"] = sprint(showerror, e)
        # Per phase-0-baseline.md#G1: failed cases are acceptable for
        # Phase 0, but every case entry must still have non-null
        # wallclock_s. Record the elapsed time spent attempting the
        # case (load+pf+init+integrate, whichever stage failed).
        record["wallclock_s"] = time() - t_case_start
    end
    return record
end

function main()
    hw = detect_hardware()
    cases = Dict{String,Any}()
    for c in CASES
        @printf "running case: %s ..." c.name
        rec = run_case(c)
        cases[c.name] = rec
        if rec["status"] == "ok"
            @printf " ok  wall=%.2fs  n_steps=%d  h=%g\n" rec["wallclock_s"] rec["n_steps"] rec["h"]
        else
            @printf " FAILED: %s\n" rec["error"]
        end
    end

    # Every claimed baseline must have completed successfully.
    criteria = Any[]
    missing_wall = String[]; missing_sha = String[]; failed_cases = String[]
    for c in CASES
        e = cases[c.name]
        if e["wallclock_s"] === nothing; push!(missing_wall, c.name); end
        if e["reference_sha256"] === nothing; push!(missing_sha, c.name); end
        if e["status"] != "ok"; push!(failed_cases, c.name); end
    end
    push!(criteria, Dict("name" => "all_cases_have_wallclock",
                         "value" => length(CASES) - length(missing_wall),
                         "threshold" => length(CASES),
                         "passed" => isempty(missing_wall)))
    push!(criteria, Dict("name" => "all_cases_have_reference_sha",
                         "value" => length(CASES) - length(missing_sha),
                         "threshold" => length(CASES),
                         "passed" => isempty(missing_sha)))
    push!(criteria, Dict("name" => "all_cases_completed",
                         "value" => length(CASES) - length(failed_cases),
                         "threshold" => length(CASES),
                         "passed" => isempty(failed_cases)))

    passed = all(c["passed"] for c in criteria)

    baseline = Dict{String,Any}(
        "phase" => 0,
        "gate"  => "G1",
        "passed" => passed,
        "criteria" => criteria,
        "hardware" => hw,
        "metadata" => Dict{String,Any}(
            "git_sha" => git_sha(),
            "captured_at" => string(Dates.now(Dates.UTC)),
            "n_cases" => length(CASES),
            "missing_wallclock" => missing_wall,
            "missing_reference_sha" => missing_sha,
            "failed_cases" => failed_cases,
        ),
        "cases" => cases,
    )

    open(OUT_PATH, "w") do io
        write(io, pretty(json_value(baseline))); write(io, "\n")
    end
    @printf "\nbaseline: passed=%s  cases=%d  artifact=%s\n" passed length(CASES) OUT_PATH
    return passed
end

exit(main() ? 0 : 1)
