# PERC2 case studies 1-5 as V/f playback (no network). All parameters are
# ASSUMED STUDY PARAMETERS; nothing here is facility or equipment validation.
#
#   julia +1.13 --project=. scripts/perc2/run_playback.jl [--selfcheck]
#
# Convention: t_k = k*dt, perc2_step!(s, p, v(t_k), f(t_k), t_k, dt), snapshot
# after step k is the response at t_k (same as test/perc2_helpers.jl).
# Output: artifacts/perc2/study<k>.json (artifacts/ is git-ignored).

using GradPower
include(joinpath(@__DIR__, "json_io.jl"))
using .PERC2JSON
include(joinpath(@__DIR__, "params_sets.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "..", "artifacts", "perc2")
const DT = 1 / 240
const TRACE_DT = 0.01
const TOL = 1e-12

# ---------------------------------------------------------------- playback --
profile_at(bps, t) = (i = findlast(b -> b[1] <= t + 1e-12, bps); (bps[i][2], bps[i][3]))

function playback(p, bps, t_end, dt; P0 = 1.0, Q0 = 0.0)
    N = round(Int, t_end / dt)
    v0, _ = profile_at(bps, 0.0)
    s = GradPower.perc2_init(p, P0, Q0, v0)
    snaps = Any[]
    for k in 0:N
        t = k * dt
        v, f = profile_at(bps, t)
        GradPower.perc2_step!(s, p, v, f, t, dt)
        push!(snaps, merge(GradPower.perc2_snapshot(s, v), (; t, V = v, f)))
    end
    return snaps
end

# Event extraction from the FracOn trace (same convention as the tests):
#   trip_time        first step with FracOn < 1
#   reconnect_start  step before FracOn first rises (the ramp start)
#   reconnect_end    first step with FracOn at its final value after rising
function fracon_events(frac, dt)
    ev = Dict{Symbol,Union{Float64,Nothing}}(:trip_time => nothing, :reconnect_start => nothing,
                                              :reconnect_end => nothing)
    kt = findfirst(x -> x < 1 - TOL, frac)
    kt === nothing && return ev
    ev[:trip_time] = (kt - 1) * dt
    kr = findfirst(k -> frac[k] > frac[k-1] + TOL, kt+1:length(frac))
    kr === nothing && return ev
    kr += kt
    ev[:reconnect_start] = (kr - 2) * dt
    kend = findfirst(k -> frac[k] >= frac[end] - TOL, kr:length(frac))
    kend === nothing && return ev
    ev[:reconnect_end] = (kend + kr - 2) * dt
    return ev
end

# Number of ceases = number of falling edges of FracOn (re-cease during a ramp
# counts as a new cease).
function count_ceases(frac)
    n = 0
    for k in 2:length(frac)
        frac[k] < frac[k-1] - TOL && (k == 2 || frac[k-1] >= frac[k-2] - TOL) && (n += 1)
    end
    return n
end

first_cross(t, x, thr, dir) = (k = dir == :below ? findfirst(<(thr), x) : findfirst(>(thr), x);
                               k === nothing ? nothing : t[k])

function downsample(snaps, dt, t_end)
    idx = unique([round(Int, j * TRACE_DT / dt) + 1 for j in 0:floor(Int, t_end / TRACE_DT + 1e-9)])
    idx = filter(i -> i <= length(snaps), idx)
    pick(f) = [getfield(snaps[i], f) for i in idx]
    return Dict{String,Any}("t" => pick(:t), "V" => pick(:V), "Vtf" => pick(:Vtf),
                            "P_pu_of_P0" => pick(:Ppu0), "Q_sysbase_pu" => pick(:Q),
                            "FracOn" => pick(:FracOn))
end

# --------------------------------------------------------------- one run ----
"""
    run_case(name, set, p, bps, t_end; P0, Q0, extra_assumptions) -> Dict

Returns a run record. `validate` or runtime errors are recorded as
"error: ...", so a failing run never aborts the study.
"""
function run_case(name, set, p, bps, t_end; P0 = 1.0, Q0 = 0.0,
                  description = "", extra_assumptions = String[], dt = DT)
    rec = Dict{String,Any}("name" => name, "description" => description,
                           "parameter_set" => set_metadata(set, p),
                           "profile" => Dict{String,Any}("breakpoints_t_V_f" => [collect(b) for b in bps],
                                                         "t_end" => t_end, "dt" => dt, "P0" => P0, "Q0" => Q0))
    try
        GradPower.validate(p)
        snaps = playback(p, bps, t_end, dt; P0, Q0)
        frac = [s.FracOn for s in snaps]
        ppu = [s.Ppu0 for s in snaps]
        vtf = [s.Vtf for s in snaps]
        t = [s.t for s in snaps]
        ev = fracon_events(frac, dt)
        ramp = ev[:reconnect_start] === nothing || ev[:reconnect_end] === nothing ? nothing :
               ev[:reconnect_end] - ev[:reconnect_start]
        nf = count(!isfinite, ppu) + count(!isfinite, frac) + count(!isfinite, vtf)
        floor_applied = p.tt_floor_steps * dt > p.Tt
        rec["status"] = nf == 0 ? "ok" : "error: non-finite values in trace"
        rec["metrics"] = Dict{String,Any}(
            "trip_time_s" => ev[:trip_time], "reconnect_start_s" => ev[:reconnect_start],
            "reconnect_end_s" => ev[:reconnect_end], "ramp_duration_s" => ramp,
            "TrampUsed_s" => p.Tramp * (p.Ldfact > 0 ? p.Ldfact : 0.8),
            "min_P_pu_of_P0" => minimum(ppu), "max_P_pu_of_P0" => maximum(ppu),
            "min_Vtf" => minimum(vtf), "FracOn_final" => frac[end], "FracOn_min" => minimum(frac),
            "n_ceases" => count_ceases(frac), "nonfinite" => nf)
        rec["assumptions"] = vcat(["frequency estimate: n/a playback (f supplied by the profile)",
                                   "Tt floor applied: $(floor_applied) (tt_floor_steps = $(p.tt_floor_steps), dt = $(round(dt; digits = 6)) s, Tt = $(p.Tt) s)",
                                   "parameters are ASSUMED STUDY PARAMETERS, not facility values"],
                                  extra_assumptions)
        rec["trace"] = downsample(snaps, dt, t_end)
        rec["_events"] = ev
        rec["_t"] = t; rec["_vtf"] = vtf; rec["_frac"] = frac
    catch e
        rec["status"] = "error: " * sprint(showerror, e)
    end
    return rec
end

function strip_private!(rec)
    for k in collect(keys(rec))
        startswith(k, "_") && delete!(rec, k)
    end
    return rec
end

function write_study(k, title, runs; extra = Dict{String,Any}())
    status = all(r -> startswith(r["status"], "ok"), runs) ? "ok" :
             any(r -> startswith(r["status"], "ok"), runs) ? "partial" : "failed"
    doc = Dict{String,Any}("study" => k, "title" => title, "status" => status,
                           "label" => "assumed study parameters; playback, no network; no facility validation",
                           "runs" => [strip_private!(r) for r in runs])
    merge!(doc, extra)
    json_string(doc)
    doc["nonfinite_values_written"] = count_nonfinite()
    path = joinpath(OUTDIR, "study$k.json")
    write_json(path, doc)
    println("study $k: $status -> $(relpath(path, pwd()))")
    return doc
end

fmt(x) = x === nothing ? "  n/a " : string(round(x; digits = 4))

# ------------------------------------------------------------------ studies --
# Study-specific assumed parameter variants are built on a base set and marked
# in the set name; every one carries the set's `source`.
variant(set, suffix, label; kw...) = (; set..., name = set.name * suffix, label,
                                      kwargs = merge(set.kwargs, NamedTuple(kw)))

function study1()
    # 1.0 -> 0.7 -> 1.0 pu, 50 ms. default_spec (VLcease = 0.5) must NOT cease:
    # the dip never gets below VLcease, and Tv = 0.05 s filters it anyway.
    # short_dip raises VLcease/VLrecon and shortens Tcease/Tv so that the
    # dip is a small dip with a fast reconnection.
    bps = [(0.0, 1.0, 60.0), (0.1, 0.7, 60.0), (0.15, 1.0, 60.0)]
    sdip = variant(perc1_like(), "+short_dip",
                  "assumed study parameters (PERC1-like with raised VLcease/VLrecon for a 0.7 pu dip)";
                  VLcease = 0.8, VLrecon = 0.9, Tcease = 0.02, Tv = 0.01, Trecon = 0.05, Tramp = 1.25)
    runs = Any[]
    for set in (perc1_like(), sdip)
        p = perc2_params(set)
        push!(runs, run_case(set.name, set, p, bps, 4.0;
                             description = "short dip 1.0->0.7->1.0 pu, 50 ms"))
    end
    println("\nStudy 1 timeline (A: Vfilt<VLcease, B: cease, W: Vfilt>VLrecon, X: ramp starts):")
    for r in runs
        startswith(r["status"], "ok") || (println("  $(r["name"]): $(r["status"])"); continue)
        p = perc2_params(get_set_by_name(r["parameter_set"]["name"], sdip))
        t, vtf, ev = r["_t"], r["_vtf"], r["_events"]
        A = first_cross(t, vtf, p.VLcease, :below)
        W = ev[:trip_time] === nothing ? nothing : first_cross(t[t .> ev[:trip_time]], vtf[t .> ev[:trip_time]], p.VLrecon, :above)
        X = ev[:reconnect_start]
        println("  $(rpad(r["name"], 22)) A=$(fmt(A))  B(trip)=$(fmt(ev[:trip_time]))  W=$(fmt(W))  X(ramp start)=$(fmt(X))  ",
                "ramp end=$(fmt(ev[:reconnect_end]))  min Vtf=$(fmt(r["metrics"]["min_Vtf"]))")
        if A !== nothing && ev[:trip_time] !== nothing
            println("    expected B-A ~ Tcease+Tdelay = $(p.Tcease + p.Tdelay) s: got $(fmt(ev[:trip_time] - A)) s (step ", round(DT; digits = 5), " s)")
        else
            println("    no cease (as expected if Vfilt never goes below VLcease = $(p.VLcease))")
        end
        if W !== nothing && X !== nothing
            println("    expected X-W ~ Trecon = $(p.Trecon) s: got $(fmt(X - W)) s")
        end
    end
    write_study(1, "Short dip, fast reconnect", runs)
end

get_set_by_name(name, extra) = name == extra.name ? extra : get_set(name)

function study2()
    # 1.0 -> 0.45 pu for 300 ms. Escalation to the long cease: slow reconnect
    # TLreconL, ramp TLrampL*Ldfact. VLrecon = 0.9 = highest active
    # long-cease voltage (VLC1) satisfies the safety rule.
    bps = [(0.0, 1.0, 60.0), (0.1, 0.45, 60.0), (0.4, 1.0, 60.0)]
    base = perc1_like()
    long = variant(base, "+long_cease", "assumed study parameters (PERC2 long-cease levels, slow reconnect)";
                   VLLflag = true, VLrecon = 0.9, TLreconL = 2.0, TLrampL = 10.0)
    runs = Any[]
    for set in (long,)
        p = perc2_params(set)
        push!(runs, run_case(set.name, set, p, bps, 16.0;
                             description = "long dip 1.0->0.45->1.0 pu, 300 ms; escalation to TLreconL/TLrampL",
                             extra_assumptions = ["TLreconL = 2 s, TLrampL = 10 s (x Ldfact) are assumed values; PERC2 defaults would equal the momentary ones"]))
    end
    r = runs[1]
    if startswith(r["status"], "ok")
        m = r["metrics"]; p = perc2_params(long)
        println("\nStudy 2: trip $(fmt(m["trip_time_s"])) s, ramp start $(fmt(m["reconnect_start_s"])) s, ",
                "ramp end $(fmt(m["reconnect_end_s"])) s; expected ramp ~ TLrampL*Ldfact = $(p.TLrampL * p.Ldfact) s, ",
                "got $(fmt(m["ramp_duration_s"])) s")
    end
    write_study(2, "Long dip, slow reconnect", runs)
end

function study3()
    bps = [(0.0, 1.0, 60.0), (0.1, 1.15, 60.0), (1.6, 1.0, 60.0)]
    base = perc1_like()
    sw = variant(base, "+high_voltage", "assumed study parameters (PERC2 high-voltage cease, slow reconnect)";
                 VHMflag = true, VLHflag = true, TLreconH = 1.0, TLrampH = 5.0)
    runs = Any[]
    p = perc2_params(sw)
    push!(runs, run_case(sw.name, sw, p, bps, 12.0;
                         description = "swell 1.0->1.15->1.0 pu, 1.5 s",
                         extra_assumptions = ["TLreconH = 1 s, TLrampH = 5 s (x Ldfact) are assumed values"]))
    write_study(3, "Swell", runs)
end

function study4()
    # Staircase sag 1.0 -> 0.9 ... 0.5 pu (0.4 s per step), no ceasing (fcease = 0).
    steps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    bps = [(0.0, 1.0, 60.0)]
    for (i, v) in enumerate(steps[2:end]); push!(bps, (0.1 + 0.4 * (i - 1), v, 60.0)); end
    push!(bps, (0.1 + 0.4 * (length(steps) - 1), 1.0, 60.0))
    tend = 0.1 + 0.4 * length(steps) + 0.4
    base = variant(perc1_like(), "+nocease", "assumed study parameters (current limiting, ceasing disabled by fcease = 0)";
                   fcease = 0.0, QPratio = 0.3)
    off = variant(base, "_imax_off", "assumed study parameters (Imax = 1.5 does not bind: Ip max 1.0, Iq max 0.7)";
                  Imax = 1.5)
    # Imax = 1.0 caps |I|: with Q0 = 0.3, Ip=1 at V <= 0.8 leaves no room for Iq.
    on = variant(base, "_imax_on", "assumed study parameters (Imax = 1.0 binds total current)";
                 Imax = 1.0)
    runs = Any[]
    for (set, bind) in ((off, false), (on, true))
        p = perc2_params(set)
        push!(runs, run_case(set.name, set, p, bps, tend; Q0 = 0.3,
                             description = "sag staircase 0.9..0.5 pu, 0.4 s per step, P0 = 1, Q0 = 0.3 (system pu)",
                             extra_assumptions = ["QPratio = Q0/P0 = 0.3 (all reactive consumption on the dynamic Q path)"; bind ? ["total-current limit Imax binds"] :
                                                         ["P-side limits only; Imax does not bind"]]))
    end
    for r in runs
        startswith(r["status"], "ok") &&
            println("Study 4 $(r["name"]): min P/P0 = $(fmt(r["metrics"]["min_P_pu_of_P0"])) (nP = 0, Ipmax = 1, Ldfact = 0.8: P/P0 -> V/0.8 once Ip hits Ipmax)")
    end
    write_study(4, "Current limiting", runs)
end

function study5()
    bps = [(0.0, 1.0, 60.0), (0.1, 0.3, 60.0), (0.25, 1.0, 60.0)]
    runs = Any[]
    sweep = Any[]
    println("\nStudy 5: Ldfact sweep, ramp duration vs TrampUsed = Tramp*Ldfact (Tramp = 1.25 s)")
    for ld in (0.3, 0.6, 0.8, 1.0)
        set = variant(perc1_like(), "_Ldfact$(ld)", "assumed study parameters (PERC1-like, Ldfact = $ld)"; Ldfact = ld)
        p = perc2_params(set)
        r = run_case(set.name, set, p, bps, 4.0;
                     description = "dip 1.0->0.3->1.0 pu (0.15 s), Ldfact = $ld")
        push!(runs, r)
        if startswith(r["status"], "ok")
            m = r["metrics"]
            err = m["ramp_duration_s"] === nothing ? nothing : m["ramp_duration_s"] - m["TrampUsed_s"]
            push!(sweep, Dict{String,Any}("Ldfact" => ld, "TrampUsed_s" => m["TrampUsed_s"],
                                          "ramp_duration_s" => m["ramp_duration_s"], "error_s" => err,
                                          "error_steps" => err === nothing ? nothing : err / DT))
            println("  Ldfact=$ld  Tramp*Ldfact=$(fmt(m["TrampUsed_s"]))  measured ramp=$(fmt(m["ramp_duration_s"]))  diff=$(fmt(err)) s")
        else
            println("  Ldfact=$ld  $(r["status"])")
        end
    end
    write_study(5, "Loading factor", runs; extra = Dict{String,Any}("ldfact_sweep" => sweep))
end

# ----------------------------------------------------------------- selfcheck --
# Check the event/trace extraction against the committed reference CSVs
# (no PERC2 model involved).
function selfcheck()
    refdir = joinpath(@__DIR__, "..", "..", "test", "perc2_reference")
    cases = read_json(joinpath(refdir, "cases.json"))
    dt = cases["dt"]
    bad = 0
    for (name, c) in sort!(collect(cases["cases"]); by = first)
        lines = filter(!isempty, strip.(readlines(joinpath(refdir, name * ".csv"))))
        hdr = split(lines[1], ',')
        col(n) = [parse(Float64, split(l, ',')[findfirst(==(n), hdr)]) for l in lines[2:end]]
        ev = fracon_events(col("FracOn"), dt)
        for (k, kk) in ((:trip_time, "trip_time"), (:reconnect_start, "reconnect_start"), (:reconnect_end, "reconnect_end"))
            want = get(c["events"], kk, nothing)
            got = ev[k]
            ok = (want === nothing && got === nothing) ||
                 (want !== nothing && got !== nothing && abs(want - got) <= dt + 1e-9)
            ok || (bad += 1; println("MISMATCH $name $kk: ref=$want got=$got"))
        end
    end
    println("selfcheck: $(bad == 0 ? "all reference events reproduced" : "$bad mismatches")")
    # json roundtrip
    d = Dict{String,Any}("a" => [1, 2.5, nothing], "b" => "x\"y", "c" => Dict("z" => true), "n" => NaN)
    pth = joinpath(OUTDIR, "_selfcheck.json"); write_json(pth, d)
    r = read_json(pth); rm(pth)
    @assert r["a"] == Any[1, 2.5, nothing] && r["b"] == "x\"y" && r["c"]["z"] === true && r["n"] === nothing
    println("selfcheck: json roundtrip ok")
    return bad == 0
end

function main(args)
    mkpath(OUTDIR)
    "--selfcheck" in args && return selfcheck()
    println("PERC2 playback studies (all parameters: assumed study parameters)")
    for f in (study1, study2, study3, study4, study5)
        try
            f()
        catch e
            @error "study failed" f exception = (e, catch_backtrace())
        end
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main(ARGS)
