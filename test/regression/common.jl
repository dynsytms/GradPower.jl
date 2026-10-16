# Shared helpers for Phase 0 regression scripts (test/regression/*).
# Each per-case script writes artifacts/regression/<case>.json with the
# shape from docs/plan/conventions.md (phase=0, gate="regression:<case>")
# and exits 0 iff passed.

using SHA
using Dates
using NPZ

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const FROZEN_REF_DIR = joinpath(REPO_ROOT, "artifacts", "phase0", "references")
const REG_OUT_DIR    = joinpath(REPO_ROOT, "artifacts", "regression")

mkpath(REG_OUT_DIR)

# Minimal JSON encoder so we do not depend on JSON.jl in Project.toml.
function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"' || c == '\\'
            print(buf, '\\', c)
        elseif c == '\n'; print(buf, "\\n")
        elseif c == '\r'; print(buf, "\\r")
        elseif c == '\t'; print(buf, "\\t")
        elseif UInt32(c) < 0x20
            print(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else
            print(buf, c)
        end
    end
    String(take!(buf))
end

function json_value(v)
    if v === nothing;            return "null"
    elseif v isa Bool;           return v ? "true" : "false"
    elseif v isa AbstractString; return string('"', _json_escape(v), '"')
    elseif v isa Integer;        return string(v)
    elseif v isa AbstractFloat;  return isfinite(v) ? string(v) : "null"
    elseif v isa AbstractVector; return string("[", join(json_value.(v), ","), "]")
    elseif v isa AbstractDict
        ks = sort(collect(keys(v)), by=string)
        parts = String[]
        for k in ks
            push!(parts, string('"', _json_escape(String(string(k))), '"',
                                ":", json_value(v[k])))
        end
        return string("{", join(parts, ","), "}")
    else
        return string('"', _json_escape(string(v)), '"')
    end
end

function pretty_json(json::AbstractString)
    buf = IOBuffer()
    indent = 0
    in_string = false
    prev = '\0'
    for c in json
        if in_string
            print(buf, c)
            if c == '"' && prev != '\\'; in_string = false; end
        else
            if c == '"';                 in_string = true; print(buf, c)
            elseif c == '{' || c == '['; print(buf, c, '\n'); indent += 2; print(buf, ' '^indent)
            elseif c == '}' || c == ']'; indent -= 2; print(buf, '\n', ' '^indent, c)
            elseif c == ','; print(buf, c, '\n', ' '^indent)
            elseif c == ':'; print(buf, c, ' ')
            else;            print(buf, c)
            end
        end
        prev = c
    end
    String(take!(buf))
end

function write_case_json(case::AbstractString, criteria::Vector, metadata::Dict)
    passed = all(c["passed"] for c in criteria)
    out = Dict{String,Any}(
        "phase" => 0,
        "gate"  => string("regression:", case),
        "passed" => passed,
        "criteria" => criteria,
        "metadata" => metadata,
    )
    path = joinpath(REG_OUT_DIR, string(case, ".json"))
    open(path, "w") do io
        write(io, pretty_json(json_value(out)))
        write(io, "\n")
    end
    return passed, path
end

function frozen_ref(name::AbstractString)
    p = joinpath(FROZEN_REF_DIR, name)
    isfile(p) || error("frozen reference not found: $p (run scripts/verify/phase0_g2.jl first)")
    return p
end

function sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function git_sha()
    try
        return strip(read(Cmd(`git rev-parse HEAD`, dir=REPO_ROOT), String))
    catch
        return ""
    end
end

function parse_json(s::AbstractString)
    p = Ref(firstindex(s))
    skip() = while p[] <= lastindex(s) && isspace(s[p[]]); p[] = nextind(s, p[]); end
    function value()
        skip()
        c = s[p[]]
        if c == '"'
            p[] = nextind(s, p[])
            out = IOBuffer()
            while s[p[]] != '"'
                c = s[p[]]
                if c == '\\'
                    p[] = nextind(s, p[])
                    c = s[p[]]
                    print(out, c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' : c)
                else
                    print(out, c)
                end
                p[] = nextind(s, p[])
            end
            p[] = nextind(s, p[])
            return String(take!(out))
        elseif c == '{'
            p[] = nextind(s, p[]); out = Dict{String,Any}(); skip()
            while s[p[]] != '}'
                key = value(); skip(); s[p[]] == ':' || error("invalid JSON object")
                p[] = nextind(s, p[]); out[key] = value(); skip()
                s[p[]] == ',' && (p[] = nextind(s, p[]); skip())
            end
            p[] = nextind(s, p[]); return out
        elseif c == '['
            p[] = nextind(s, p[]); out = Any[]; skip()
            while s[p[]] != ']'
                push!(out, value()); skip()
                s[p[]] == ',' && (p[] = nextind(s, p[]); skip())
            end
            p[] = nextind(s, p[]); return out
        elseif startswith(SubString(s, p[]), "true")
            p[] += 4; return true
        elseif startswith(SubString(s, p[]), "false")
            p[] += 5; return false
        elseif startswith(SubString(s, p[]), "null")
            p[] += 4; return nothing
        else
            start = p[]
            while p[] <= lastindex(s) && (isdigit(s[p[]]) || s[p[]] in ('-', '+', '.', 'e', 'E'))
                p[] = nextind(s, p[])
            end
            token = s[start:prevind(s, p[])]
            return occursin(r"[.eE]", token) ? parse(Float64, token) : parse(Int, token)
        end
    end
    result = value(); skip()
    p[] > lastindex(s) || error("trailing JSON data")
    return result
end

function load_reference(path::AbstractString)
    arrays = npzread(path)
    metadata = if haskey(arrays, "metadata_json")
        parse_json(String(UInt8.(vec(arrays["metadata_json"]))))
    else
        nothing
    end
    return (arrays=arrays, metadata=metadata)
end

function build_state_manifest(ps::GradPower.PowerSystem)
    dyn = ps.dynamic
    dyn === nothing && error("dynamic system is not initialized")
    manifest = Vector{Dict{String,Any}}()
    for (device_index, device) in enumerate(dyn.devices)
        dtype = device.dtype
        bus_internal = dyn.map.bus[device_index]
        bus_external = ps.buses[bus_internal].i
        # StaticGenerator has no id; uqgrid names it static@<internal bus, 0-based>.
        id = dtype isa GradPower.StaticGenerator ? "static@$(bus_internal - 1)" :
             hasproperty(dtype, :id) ? GradPower._normalize_id(getproperty(dtype, :id)) : ""
        model = lowercase(get_device_name(dtype))
        for (offset, name) in enumerate(get_diff_names(dtype))
            push!(manifest, Dict("index" => device.diff_ptr + offset - 1,
                                 "partition" => "differential", "name" => name,
                                 "device_type" => model, "canonical_model" => model,
                                 "device_id" => id,
                                 "bus_external" => bus_external))
        end
        for (offset, name) in enumerate(get_alg_names(dtype))
            push!(manifest, Dict("index" => dyn.diff_dim + device.alg_ptr + offset - 1,
                                 "partition" => "algebraic", "name" => name,
                                 "device_type" => model, "canonical_model" => model,
                                 "device_id" => id,
                                 "bus_external" => bus_external))
        end
    end
    net = dyn.diff_dim + dyn.alg_dim
    for (bus_internal, bus) in enumerate(ps.buses)
        for (offset, name) in enumerate(("voltage_real", "voltage_imaginary"))
            push!(manifest, Dict("index" => net + 2*(bus_internal - 1) + offset,
                                 "partition" => "network", "name" => name,
                                 "device_type" => "bus", "canonical_model" => "bus",
                                 "device_id" => string(bus.i),
                                 "bus_external" => bus.i))
        end
    end
    sort!(manifest, by=x -> x["index"])
    length(manifest) == dyn.diff_dim + dyn.alg_dim + 2length(ps.buses) ||
        error("GradPower state manifest is incomplete")
    return manifest
end

_state_key(entry) = entry["partition"] == "network" ?
    ("network", "bus", Int(entry["bus_external"]), "", String(entry["name"])) :
    (String(entry["partition"]), String(entry["canonical_model"]),
     Int(entry["bus_external"]), GradPower._normalize_id(String(entry["device_id"])),
     String(entry["name"]))

const UQGRID_ONLY_STATES = Set(["p_m0", "e_fd0", "p_m_out", "e_fd_out"])

function map_state_manifests(julia_manifest, reference_manifest;
                             reference_index_base::Int=0,
                             allowed_reference_names=UQGRID_ONLY_STATES)
    julia_by_key = Dict{Tuple{String,String,Int,String,String},Int}()
    for entry in julia_manifest
        key = _state_key(entry)
        haskey(julia_by_key, key) && error("duplicate GradPower state key $key")
        julia_by_key[key] = Int(entry["index"])
    end
    reference_by_key = Dict{Tuple{String,String,Int,String,String},Int}()
    for entry in reference_manifest
        key = _state_key(entry)
        haskey(reference_by_key, key) && error("duplicate reference state key $key")
        reference_by_key[key] = Int(entry["index"]) + 1 - reference_index_base
    end
    rows = Tuple{String,Int,Int}[]
    unmatched_reference = String[]
    for (key, reference_index) in reference_by_key
        if haskey(julia_by_key, key)
            label = join((key[2], key[3], key[4], key[5]), ":")
            push!(rows, (label, julia_by_key[key], reference_index))
        elseif !(key[5] in allowed_reference_names)
            push!(unmatched_reference, string(key))
        end
    end
    isempty(unmatched_reference) || error("unmatched reference states: $(join(sort(unmatched_reference), ", "))")
    length(rows) == length(julia_by_key) || error("only $(length(rows)) of $(length(julia_by_key)) GradPower states matched")
    isempty(rows) && error("state manifests produced no matches")
    sort!(rows, by=x -> x[2])
    return rows
end

function compare_trajectories(tvec, traj, ref_tvec, history, rows;
                              threshold::Float64, event_times=Float64[], mask_steps::Int=0)
    length(tvec) == length(ref_tvec) || error("time-vector length mismatch")
    maximum(abs, tvec .- ref_tvec) <= 10eps(Float64) || error("time grids do not align")
    mask = trues(length(tvec))
    for event_time in event_times
        center = argmin(abs.(tvec .- event_time))
        mask[max(1, center-mask_steps):min(length(mask), center+mask_steps)] .= false
    end
    any(mask) || error("event mask excludes every sample")
    worst = 0.0; worst_state = ""; worst_time = 0.0
    first_state = nothing; first_time = nothing; first_error = nothing
    initial_error = 0.0
    per_state = Dict{String,Float64}()
    for (name, jr, rr) in rows
        errors = abs.(traj[jr, :] .- history[rr, :])
        initial_error = max(initial_error, errors[1])
        masked = findall(mask)
        local_pos = argmax(errors[masked]); sample = masked[local_pos]
        per_state[name] = errors[sample]
        if errors[sample] > worst
            worst = errors[sample]; worst_state = name; worst_time = tvec[sample]
        end
        crossing = findfirst(i -> mask[i] && errors[i] > threshold, eachindex(errors))
        if crossing !== nothing && (first_time === nothing || tvec[crossing] < first_time)
            first_state = name; first_time = tvec[crossing]; first_error = errors[crossing]
        end
    end
    return Dict{String,Any}(
        "max_abs_error" => worst, "worst_state" => worst_state,
        "worst_time" => worst_time, "initial_state_error" => initial_error,
        "first_divergence_state" => first_state,
        "first_divergence_time" => first_time,
        "first_divergence_error" => first_error,
        "n_samples_total" => length(tvec), "n_samples_compared" => count(mask),
        "per_state_max_abs" => per_state,
    )
end

function comparison_rows(ps, reference; legacy_rows=nothing)
    if reference.metadata === nothing
        legacy_rows === nothing && error("legacy reference requires explicit row mapping")
        isempty(legacy_rows) && error("legacy row mapping cannot be empty")
        return legacy_rows
    end
    manifest = reference.metadata["state_manifest"]
    length(manifest) == size(reference.arrays["history"], 1) ||
        error("reference state manifest length does not match history")
    index_base = Int(get(reference.metadata, "state_index_base", 0))
    return map_state_manifests(build_state_manifest(ps), manifest;
                               reference_index_base=index_base)
end

function setup_regression_case(raw, dyr; zipload_alpha=1.0, add_static_gen_stubs=true,
                               enforce_q_limits=true)
    ps = GradPower.from_psse(raw, dyr; add_static_gen_stubs=add_static_gen_stubs)
    GradPower.build_network!(ps)
    GradPower.runpf!(ps; enforce_q_limits=enforce_q_limits)
    for device in ps.dynamic.devices
        device.dtype isa GradPower.ZIPLoad && (device.dtype.α = zipload_alpha)
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    return ps, dp
end

function run_reference_regression(case, raw, dyr, reference_name;
                                  fault_bus_external::Int, rfault::Float64,
                                  ton::Float64, toff::Float64, dt::Float64,
                                  tend::Float64, zipload_alpha::Float64,
                                  tolerance::Float64, legacy_rows=nothing,
                                  event_mask_steps::Int=0,
                                  add_static_gen_stubs::Bool=true,
                                  enforce_q_limits::Bool=true)
    reference_path = frozen_ref(reference_name)
    reference = load_reference(reference_path)
    history = reference.arrays["history"]
    ref_tvec = vec(reference.arrays["tvec"])

    ps_flat, dp_flat = setup_regression_case(
        raw, dyr;
        zipload_alpha=zipload_alpha,
        add_static_gen_stubs=add_static_gen_stubs,
        enforce_q_limits=enforce_q_limits,
    )
    initial = copy(dp_flat.zvec)
    _, flat = GradPower.integrate!(dp_flat, ps_flat, min(1.0, tend); dt=dt, verbose=false)
    flat_line_drift = maximum(abs, flat .- initial)

    ps, dp = setup_regression_case(
        raw, dyr;
        zipload_alpha=zipload_alpha,
        add_static_gen_stubs=add_static_gen_stubs,
        enforce_q_limits=enforce_q_limits,
    )
    fault_bus = get(ps.busmap, fault_bus_external, 0)
    fault_bus == 0 && error("fault bus $fault_bus_external is not present")
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, rfault, ton, toff))
    started = time()
    tvec, trajectory = GradPower.integrate!(dp, ps, tend; dt=dt, verbose=false)
    wallclock = time() - started

    rows = comparison_rows(ps, reference; legacy_rows=legacy_rows)
    comparison = compare_trajectories(
        tvec, trajectory, ref_tvec, history, rows;
        threshold=tolerance, event_times=[ton, toff], mask_steps=event_mask_steps,
    )
    criteria = Any[
        Dict("name" => "max_traj_err_inf", "value" => comparison["max_abs_error"],
             "threshold" => tolerance, "passed" => comparison["max_abs_error"] <= tolerance),
        Dict("name" => "initial_state_error", "value" => comparison["initial_state_error"],
             "threshold" => tolerance, "passed" => comparison["initial_state_error"] <= tolerance),
        Dict("name" => "flat_line_drift", "value" => flat_line_drift,
             "threshold" => 1.0e-6, "passed" => flat_line_drift <= 1.0e-6),
    ]
    metadata = Dict{String,Any}(
        "case" => case, "wallclock_s" => wallclock,
        "reference_path" => reference_path,
        "reference_sha256" => sha256_file(reference_path),
        "reference_schema_version" => reference.metadata === nothing ? nothing : reference.metadata["schema_version"],
        "mapping_count" => length(rows),
        "flat_line_drift" => flat_line_drift,
        "comparison" => comparison,
        "limit_activation_summary" => Dict("available" => false),
        "git_sha" => git_sha(),
    )
    return write_case_json(case, criteria, metadata)
end
