# PERC2 network case studies 6 and 7 on the bundled IEEE 9-bus case.
# All parameters are ASSUMED STUDY PARAMETERS; no facility or equipment validation.
#
#   julia +1.13 --project=. scripts/perc2/run_network.jl [6|7|8|9]
#
# Study 6: PERC2 = 80 % of the largest load bus, 3-phase (shunt) fault at an
#          adjacent bus at t = 1.0 s, cleared at 1.075 s (4.5 cycles at 60 Hz);
#          ZIP baseline vs each parameter set.
# Study 7: PERC2 (80 %) at all three load buses; identical vs heterogeneous sets.
# Study 8: VLcease x Ldfact sweep and a batched-integrator check.
# Study 9: frequency ceasing after a generator trip.
# Output: artifacts/perc2/study<k>.json (git-ignored).

using GradPower
include(joinpath(@__DIR__, "run_playback.jl"))     # json_io, params_sets, fracon_events, ...

const EX = joinpath(@__DIR__, "..", "..", "examples")
const NET_DT = 1 / 120
const NET_TF = 10.0
const T_FAULT_ON = 1.0
const T_FAULT_OFF = 1.075            # 9 steps of 1/120 s = 4.5 cycles
const FAULT_R = 0.01                 # pu, resistive shunt fault (assumed); see sag check in output
const PERC_FRACTION = 0.8
const FNOM = 60.0

function setup_system()
    ps = from_psse(joinpath(EX, "ieee9_v33.raw"), joinpath(EX, "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    return ps
end

# external numbers of load buses sorted by P (descending), with P in MW
function load_table(ps)
    ls = [(bus = ps.buses[l.bus].i, id = strip(l.id), P_MW = l.pd * ps.baseMVA, Q_MW = l.qd * ps.baseMVA)
          for l in ps.loads]
    return sort(ls; by = l -> -l.P_MW)
end

neighbours(ps, bus_ext) = begin
    ib = GradPower.bus_index(ps, bus_ext)
    out = Int[]
    for b in ps.branches
        b.fr == ib && push!(out, ps.buses[b.to].i)
        b.to == ib && push!(out, ps.buses[b.fr].i)
    end
    sort!(unique(out))
end

# --- log access (NamedTuple per accepted step: t, device, snapshot fields) ----
lf(e, k, default = nothing) = hasproperty(e, k) ? getproperty(e, k) : default

function split_log(log)
    d = Dict{Any,Vector{Any}}()
    for e in log
        push!(get!(d, lf(e, :device, 0), Any[]), e)
    end
    for v in values(d); sort!(v; by = e -> e.t); end
    return d
end

"""
    simulate(assign; fault_bus, ...) -> Dict

`assign` is a Vector of `(bus_ext, set)`; empty = ZIP baseline. Returns a record
with the generator, bus-voltage and PERC2 traces.
"""
function simulate(assign, fault_bus_ext; tf = NET_TF, dt = NET_DT, integrator = :backward_euler,
                  trip_gen = nothing)
    ps = setup_system()
    ldmap = Dict(l.bus => l for l in load_table(ps))
    params = Dict{Int,Any}()
    for (bus, set) in assign
        p = perc2_params(set)
        params[bus] = p
        GradPower.add_perc2_load!(ps, bus, ldmap[bus].id; fraction = PERC_FRACTION, params = p)
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    if fault_bus_ext !== nothing
        GradPower.add_event!(ps, GradPower.ContingencyEvent(GradPower.bus_index(ps, fault_bus_ext),
                                                           FAULT_R, T_FAULT_ON, T_FAULT_OFF))
    end
    trip_gen === nothing || GradPower.add_generator_trip!(ps, trip_gen.bus, trip_gen.id, trip_gen.t)
    log = Any[]
    kw = isempty(assign) ? (;) : (; perc2_log = log)
    r = integrate!(dp, ps, tf; dt = dt, result = true, integrator = integrator, kw...)
    t = collect(Float64, r.times)
    gens = [(d.dtype.bus, strip(string(d.dtype.id))) for d in ps.dynamic.devices if d.dtype isa GradPower.Genrou]
    w = [collect(Float64, speed(r, ps.buses[b].i, id)) for (b, id) in gens]
    vm = Dict(l.bus => collect(Float64, bus_voltage(r, l.bus; component = :magnitude)) for l in load_table(ps))
    return (; ps, t, w, gens, vm, log, params, baseMVA = ps.baseMVA, status = r.status, events = r.events)
end

# generator metrics: speed deviation in Hz (w is pu deviation of nominal speed)
function speed_metrics(sim)
    W = reduce(hcat, sim.w)                   # nt x ngen
    dev = abs.(W) .* FNOM
    mean_w = vec(sum(W; dims = 2) ./ size(W, 2)) .* FNOM
    return Dict{String,Any}("max_abs_speed_dev_Hz" => maximum(dev),
                            "max_speed_dev_Hz_by_gen" => vec(maximum(dev; dims = 1)),
                            "mean_speed_dev_Hz_min" => minimum(mean_w),
                            "mean_speed_dev_Hz_max" => maximum(mean_w),
                            "mean_speed_dev_Hz_final" => mean_w[end],
                            "time_of_max_abs_speed_dev_s" => sim.t[argmax(vec(maximum(dev; dims = 2)))])
end

function perc_device_series(sim, dev_entries, bus_ext)
    t = [e.t for e in dev_entries]
    P_MW = [lf(e, :P, NaN) * sim.baseMVA for e in dev_entries]
    Q_MVAr = [lf(e, :Q, NaN) * sim.baseMVA for e in dev_entries]
    frac = [lf(e, :FracOn, NaN) for e in dev_entries]
    vtf = [lf(e, :Vtf, NaN) for e in dev_entries]
    f_Hz = [FNOM * (1 + lf(e, :ftf, NaN)) for e in dev_entries]
    return (; bus = bus_ext, t, P_MW, Q_MVAr, frac, vtf, f_Hz)
end

function sample_idx(t, dt = TRACE_DT)
    idx = unique([clamp(round(Int, j * dt / (t[2] - t[1])) + 1, 1, length(t)) for j in 0:floor(Int, t[end] / dt + 1e-9)])
    return idx
end

function trace_dict(sim, perc_series)
    idx = sample_idx(sim.t)
    d = Dict{String,Any}("t" => sim.t[idx])
    for (b, v) in sim.vm; d["V_bus$b"] = v[idx]; end
    for (k, (b, id)) in enumerate(sim.gens)
        d["speed_dev_Hz_gen$(b)"] = sim.w[k][idx] .* FNOM
    end
    for s in perc_series
        # log entries are on the same step grid (k*dt); align by nearest time
        tt = s.t
        ii = [clamp(round(Int, (x - tt[1]) / (tt[2] - tt[1])) + 1, 1, length(tt)) for x in sim.t[idx]]
        d["perc2_bus$(s.bus)_P_MW"] = s.P_MW[ii]
        d["perc2_bus$(s.bus)_Q_MVAr"] = s.Q_MVAr[ii]
        d["perc2_bus$(s.bus)_FracOn"] = s.frac[ii]
        d["perc2_bus$(s.bus)_Vtf"] = s.vtf[ii]
        d["perc2_bus$(s.bus)_f_Hz"] = s.f_Hz[ii]
    end
    return d
end

function perc_metrics(s, p, P0_MW)
    ev = fracon_events(s.frac, s.t[2] - s.t[1])
    # fracon_events uses index*dt; shift if the log does not start at t = 0
    off = s.t[1] - 0.0
    sh(x) = x === nothing ? nothing : x + off
    ramp = ev[:reconnect_start] === nothing || ev[:reconnect_end] === nothing ? nothing :
           ev[:reconnect_end] - ev[:reconnect_start]
    return Dict{String,Any}("bus" => s.bus, "P0_MW" => P0_MW,
        "trip_time_s" => sh(ev[:trip_time]), "reconnect_start_s" => sh(ev[:reconnect_start]),
        "reconnect_end_s" => sh(ev[:reconnect_end]), "ramp_duration_s" => ramp,
        "TrampUsed_s" => p.Tramp * (p.Ldfact > 0 ? p.Ldfact : 0.8),
        "min_P_MW" => minimum(s.P_MW), "P_final_MW" => s.P_MW[end],
        "min_Vtf" => minimum(s.vtf), "min_FracOn" => minimum(s.frac), "FracOn_final" => s.frac[end],
        "n_ceases" => count_ceases(s.frac),
        "nonfinite" => count(!isfinite, s.P_MW) + count(!isfinite, s.frac) + count(!isfinite, s.vtf))
end

function bus_v_metrics(sim, bus)
    v = sim.vm[bus]
    i1 = findfirst(>=(T_FAULT_ON - 1e-9), sim.t); i2 = findfirst(>=(T_FAULT_OFF + 1.0), sim.t)
    return Dict{String,Any}("bus" => bus, "min_V" => minimum(v), "max_V" => maximum(v),
                            "min_V_during_fault" => minimum(v[i1:findfirst(>=(T_FAULT_OFF - 1e-9), sim.t)]),
                            "max_V_after_clearing" => maximum(v[findfirst(>=(T_FAULT_OFF - 1e-9), sim.t):end]),
                            "V_final" => v[end])
end

common_assumptions(extra = String[]) = vcat([
    "QPratio = Q0/P0 of each host load (power-flow consumption) in every PERC2 set (Mvarinit = QPratio*P0 on the dynamic Q path, the rest is constant current following FracOn)",
    "all parameters are assumed study parameters (NERC Appendix H example settings / PERC2 defaults); not facility values",
    "fault: shunt resistive, rfault = $(FAULT_R) pu on the system base at the adjacent bus, on at $(T_FAULT_ON) s, off at $(T_FAULT_OFF) s (4.5 cycles at 60 Hz)",
    "dt = 1/120 s, backward Euler, monolithic CPU solver; Tt floor = tt_floor_steps*dt = $(round(2 * NET_DT; digits = 5)) s <= Tt = 0.02 s: Tt floor applied: false",
    "partitioned PERC2: load current held during each Newton solve, perc2_step! once per accepted step (one-step lag)",
    "frequency estimate: bus-angle derivative inside the PERC2 device; only used by the droop (Kdroop) path here, FL/FH cease flags are off in every set",
    "speed deviation = generator :w (pu of nominal) x 60 Hz; frequency proxy only, not a system frequency measurement"], extra)

function run_one(label, set_by_bus, fault_bus; note = "", integrator = :backward_euler, trip_gen = nothing,
                 tf = NET_TF)
    # QPratio = Q0/P0 of the host load (power-flow consumption); Q0 = -qd (qd has injection sign)
    qpr = Dict(l.bus => -l.Q_MW / l.P_MW for l in load_table(setup_system()))
    set_by_bus = [(b, (; s..., kwargs = merge(s.kwargs, (; QPratio = qpr[b])))) for (b, s) in set_by_bus]
    rec = Dict{String,Any}("name" => label, "note" => note, "integrator" => string(integrator),
        "QPratio_by_bus" => Dict{String,Any}(string(b) => qpr[b] for (b, _) in set_by_bus),
        "assignment" => [Dict{String,Any}("bus" => b, "parameter_set" => set_metadata(s, perc2_params(s))) for (b, s) in set_by_bus])
    try
        sim = simulate(set_by_bus, fault_bus; integrator, trip_gen, tf)
        rec["status"] = all(isfinite, reduce(hcat, sim.w)) && all(v -> all(isfinite, v), values(sim.vm)) ? "ok" : "error: non-finite generator or voltage trace"
        rec["solver_status"] = string.(sim.status)
        rec["speed"] = speed_metrics(sim)
        rec["bus_voltage"] = [bus_v_metrics(sim, b) for b in sort!(collect(keys(sim.vm)))]
        series = Any[]; pm = Any[]
        if !isempty(set_by_bus)
            groups = split_log(sim.log)
            isempty(groups) && error("perc2_log returned no entries")
            devs = sort!(collect(keys(groups)))
            length(devs) == length(set_by_bus) || error("expected $(length(set_by_bus)) PERC2 devices in the log, got $(length(devs))")
            # PERC2 devices are added in the order of `set_by_bus`
            ldp = Dict(l.bus => l for l in load_table(sim.ps))
            for (k, (b, set)) in enumerate(set_by_bus)
                s = perc_device_series(sim, groups[devs[k]], b)
                push!(series, s)
                push!(pm, perc_metrics(s, sim.params[b], PERC_FRACTION * ldp[b].P_MW))
            end
            rec["perc2"] = pm
            tot_t = series[1].t
            total = sum(s.P_MW for s in series)
            P0 = sum(m["P0_MW"] for m in pm)
            i1 = findfirst(>=(T_FAULT_ON - 1e-9), tot_t)
            rec["total_perc2"] = Dict{String,Any}("P0_MW" => P0, "P_before_fault_MW" => total[i1 - 1],
                "min_total_P_MW" => minimum(total), "max_loss_MW" => total[i1 - 1] - minimum(total),
                "time_of_min_total_P_s" => tot_t[argmin(total)], "P_final_MW" => total[end],
                "restored_MW_at_end" => total[end] - minimum(total))
            rec["ramp_overlap"] = ramp_overlap(series)
            rec["_total"] = (tot_t, total)
        end
        rec["trace"] = trace_dict(sim, series)
        if !isempty(series)
            tt, total = rec["_total"]
            ii = [clamp(round(Int, (x - tt[1]) / (tt[2] - tt[1])) + 1, 1, length(tt)) for x in sim.t[sample_idx(sim.t)]]
            rec["trace"]["perc2_total_P_MW"] = total[ii]
            delete!(rec, "_total")
        end
        rec["load_table"] = [Dict{String,Any}("bus" => l.bus, "P_MW" => l.P_MW, "Q_MW" => l.Q_MW) for l in load_table(sim.ps)]
        rec["fault_events_applied"] = [string(e) for e in sim.events]
    catch e
        rec["status"] = "error: " * first(sprint(showerror, e), 400)
        delete!(rec, "_total")
    end
    return rec
end

# time during which >= 2 PERC2 devices ramp up simultaneously (FracOn increasing)
function ramp_overlap(series)
    dt = series[1].t[2] - series[1].t[1]
    n = minimum(length(s.frac) for s in series)
    ramping = [vcat(false, [s.frac[k] > s.frac[k-1] + 1e-9 for k in 2:n]) for s in series]
    cnt = [count(r[k] for r in ramping) for k in 1:n]
    spans = [Dict{String,Any}("bus" => s.bus,
                              "ramp_start_s" => (i = findfirst(r -> r, ramping[j]); i === nothing ? nothing : series[j].t[i]),
                              "ramp_end_s" => (i = findlast(r -> r, ramping[j]); i === nothing ? nothing : series[j].t[i]))
             for (j, s) in enumerate(series)]
    return Dict{String,Any}("time_with_2plus_ramping_s" => count(>=(2), cnt) * dt,
                            "max_simultaneous_ramping" => maximum(cnt), "ramp_spans" => spans)
end

# compact metrics used for the backward-Euler vs Crank-Nicolson comparison
function key_metrics(r)
    m = Dict{String,Any}("max_abs_speed_dev_Hz" => r["speed"]["max_abs_speed_dev_Hz"],
                         "min_V_any_load_bus" => minimum(b["min_V"] for b in r["bus_voltage"]))
    if haskey(r, "perc2")
        for pm in r["perc2"]
            b = pm["bus"]
            for k in ("trip_time_s", "reconnect_start_s", "reconnect_end_s", "min_P_MW", "P_final_MW", "FracOn_final")
                m["bus$(b)_$k"] = pm[k]
            end
        end
        m["min_total_P_MW"] = r["total_perc2"]["min_total_P_MW"]
    end
    return m
end

# Run backward Euler (stored in full) and Crank-Nicolson; attach CN metrics and differences.
function run_pair(label, asg, fault_bus; note = "")
    be = run_one(label, asg, fault_bus; note)
    cn = run_one(label * "_cn", asg, fault_bus; note, integrator = :crank_nicolson)
    if startswith(be["status"], "ok") && startswith(cn["status"], "ok")
        mb, mc = key_metrics(be), key_metrics(cn)
        be["crank_nicolson"] = Dict{String,Any}("status" => "ok", "metrics" => mc,
            "diff_cn_minus_be" => Dict{String,Any}(k => (mb[k] === nothing || mc[k] === nothing ? nothing : mc[k] - mb[k]) for k in keys(mb)))
        be["crank_nicolson"]["trace_perc2_total_P_MW"] = get(cn["trace"], "perc2_total_P_MW", nothing)
    else
        be["crank_nicolson"] = Dict{String,Any}("status" => cn["status"])
    end
    return be
end

function print_cn_diff(runs)
    println("  Crank-Nicolson minus backward-Euler (key metrics; times in s, speed in Hz):")
    for r in runs
        c = get(r, "crank_nicolson", nothing)
        c === nothing && continue
        if c["status"] != "ok"
            println("    ", rpad(r["name"], 24), c["status"]); continue
        end
        d = c["diff_cn_minus_be"]
        tk = sort([k for k in keys(d) if endswith(k, "trip_time_s") || endswith(k, "reconnect_start_s") || endswith(k, "reconnect_end_s")])
        println("    ", rpad(r["name"], 24), "d(max|dw|)=", fmt5(d["max_abs_speed_dev_Hz"]), "  d(minV)=", fmt5(d["min_V_any_load_bus"]),
                haskey(d, "min_total_P_MW") ? "  d(minPtot MW)=" * fmt5(d["min_total_P_MW"]) : "",
                join(["  d($(replace(k, r"_s$" => "")))=" * fmt5(d[k]) for k in tk]))
    end
end
fmt5(x) = x === nothing ? "n/a" : string(round(x; sigdigits = 3))

function sag_check(fault_bus_ext, load_bus)
    sim = simulate(Tuple[], fault_bus_ext; tf = 2.0)
    m = bus_v_metrics(sim, load_bus)
    println("  ZIP baseline: min V at bus $load_bus during the fault = $(round(m["min_V_during_fault"]; digits = 3)) pu")
    return sim
end

function study6()
    ps = setup_system()
    lt = load_table(ps)
    host = lt[1]
    nb = neighbours(ps, host.bus)
    # fault bus: the adjacent bus (first neighbour); choice reported below
    fault_bus = first(nb)
    println("\nStudy 6: largest load bus = $(host.bus) ($(host.P_MW) MW + j$(host.Q_MW) MVAr); neighbours $(nb); fault at bus $fault_bus")
    println("  PERC2 share = $(PERC_FRACTION) -> $(PERC_FRACTION * host.P_MW) MW")
    sag_check(fault_bus, host.bus)

    sets = [s for s in all_sets() if s.name != "default_spec"]     # identical to perc1_like
    runs = Any[run_pair("zip_baseline", Tuple[], fault_bus; note = "no PERC2: all loads stay ZIP")]
    for s in sets
        push!(runs, run_pair(s.name, [(host.bus, s)], fault_bus))
    end

    println(rpad("  run", 22), rpad("status", 8), rpad("minV", 8), rpad("trip s", 9), rpad("rec.start", 11),
            rpad("rec.end", 9), rpad("minP MW", 9), rpad("Pend MW", 9), rpad("FracEnd", 9), "max|dw| Hz")
    for r in runs
        if !startswith(get(r, "status", ""), "ok")
            println("  ", rpad(r["name"], 20), r["status"]); continue
        end
        vb = only(filter(x -> x["bus"] == host.bus, r["bus_voltage"]))
        pm = haskey(r, "perc2") ? r["perc2"][1] : nothing
        println("  ", rpad(r["name"], 20), rpad("ok", 8), rpad(round(vb["min_V"]; digits = 3), 8),
                rpad(pm === nothing ? "-" : fmt(pm["trip_time_s"]), 9), rpad(pm === nothing ? "-" : fmt(pm["reconnect_start_s"]), 11),
                rpad(pm === nothing ? "-" : fmt(pm["reconnect_end_s"]), 9),
                rpad(pm === nothing ? "-" : round(pm["min_P_MW"]; digits = 2), 9),
                rpad(pm === nothing ? "-" : round(pm["P_final_MW"]; digits = 2), 9),
                rpad(pm === nothing ? "-" : round(pm["FracOn_final"]; digits = 3), 9),
                round(r["speed"]["max_abs_speed_dev_Hz"]; digits = 5))
    end
    print_cn_diff(runs)
    # sensitivity: PERC2-default VHrecon = 1.1 pu instead of 1.2 pu
    ext = Any[]
    println("  sensitivity VHrecon = 1.1 pu (PERC2 default):")
    for s in sets
        s.name in ("perc1_like",) && continue
        sv = variant(s, "_vhrecon1.1", "assumed study parameters ($(s.name) with PERC2-default VHrecon = 1.1 pu)"; VHrecon = 1.1)
        e = run_one(sv.name, [(host.bus, sv)], fault_bus; note = "sensitivity: VHrecon = 1.1 pu")
        delete!(e, "trace")
        push!(ext, e)
        if startswith(e["status"], "ok")
            q = e["perc2"][1]
            println("    ", rpad(sv.name, 26), "trip=", fmt(q["trip_time_s"]), " rec.start=", fmt(q["reconnect_start_s"]),
                    " FracOn end=", round(q["FracOn_final"]; digits = 3), " max|dw|=", round(e["speed"]["max_abs_speed_dev_Hz"]; digits = 3), " Hz")
        end
    end
    status = all(r -> startswith(r["status"], "ok"), runs) ? "ok" : "partial"
    doc = Dict{String,Any}("study" => 6, "vhrecon_sensitivity_runs" => ext, "title" => "Nearby fault, IEEE 9-bus", "status" => status,
        "label" => "NERC Appendix H example settings (assumed); no facility validation",
        "host_bus" => host.bus, "host_load_MW" => host.P_MW, "perc2_fraction" => PERC_FRACTION,
        "fault_bus" => fault_bus, "fault" => Dict{String,Any}("rfault_pu" => FAULT_R, "t_on" => T_FAULT_ON, "t_off" => T_FAULT_OFF),
        "dt" => NET_DT, "tf" => NET_TF, "assumptions" => common_assumptions(
            ["default_spec is identical to perc1_like (not run separately)",
             "VHrecon = 1.1 pu (PERC2 default) is used by every set: after the load ceases the bus voltage can rise above it, which blocks the reconnect window (strict VLrecon < Vtf < VHrecon)",
             "sets with Trecon + ramp longer than the remaining simulation time do not return within tf = $(NET_TF) s"]),
        "runs" => runs)
    doc["nonfinite_values_written"] = 0
    n = write_json(joinpath(OUTDIR, "study6.json"), doc)
    println("study 6: $status -> artifacts/perc2/study6.json (non-finite values written as null: $n)")
    return doc
end

function study7()
    ps = setup_system()
    lt = load_table(ps)
    host = lt[1]
    fault_bus = first(neighbours(ps, host.bus))
    buses = [l.bus for l in lt]
    println("\nStudy 7: PERC2 at load buses $(buses) ($(PERC_FRACTION*100)% each); fault at bus $fault_bus")
    hetero_sets = [vertiv_frt(), eaton_frt(), equinix()]
    ident = [(b, perc1_like()) for b in buses]
    hetero = [(b, s) for (b, s) in zip(buses, hetero_sets)]
    plan = [("zip_baseline", Tuple[], "no PERC2"),
            ("identical_perc1_like", ident, "perc1_like at every PERC2 bus"),
            ("identical_vertiv_frt", [(b, vertiv_frt()) for b in buses], "vertiv_frt at every PERC2 bus"),
            ("heterogeneous", hetero, "largest bus: vertiv_frt, second: eaton_frt, third: equinix (assumed assignment)")]
    runs = Any[]
    for (nm, asg, note) in plan
        push!(runs, run_pair(nm, asg, fault_bus; note))
    end
    println(rpad("  run", 26), rpad("status", 8), rpad("P0 MW", 9), rpad("max loss", 10), rpad("t(min)", 8),
            rpad("P end", 9), rpad("overlap s", 10), rpad("max conc.", 10), "max|dw| Hz")
    for r in runs
        if !startswith(get(r, "status", ""), "ok")
            println("  ", rpad(r["name"], 24), r["status"]); continue
        end
        tp = get(r, "total_perc2", nothing); ro = get(r, "ramp_overlap", nothing)
        println("  ", rpad(r["name"], 24), rpad("ok", 8),
                rpad(tp === nothing ? "-" : round(tp["P0_MW"]; digits = 1), 9),
                rpad(tp === nothing ? "-" : round(tp["max_loss_MW"]; digits = 1), 10),
                rpad(tp === nothing ? "-" : round(tp["time_of_min_total_P_s"]; digits = 2), 8),
                rpad(tp === nothing ? "-" : round(tp["P_final_MW"]; digits = 1), 9),
                rpad(ro === nothing ? "-" : round(ro["time_with_2plus_ramping_s"]; digits = 2), 10),
                rpad(ro === nothing ? "-" : ro["max_simultaneous_ramping"], 10),
                round(r["speed"]["max_abs_speed_dev_Hz"]; digits = 5))
    end
    print_cn_diff(runs)
    status = all(r -> startswith(r["status"], "ok"), runs) ? "ok" : "partial"
    doc = Dict{String,Any}("study" => 7, "title" => "Clustered facilities, IEEE 9-bus", "status" => status,
        "label" => "NERC Appendix H example settings (assumed); no facility validation",
        "perc2_buses" => buses, "perc2_fraction" => PERC_FRACTION, "fault_bus" => fault_bus,
        "fault" => Dict{String,Any}("rfault_pu" => FAULT_R, "t_on" => T_FAULT_ON, "t_off" => T_FAULT_OFF),
        "dt" => NET_DT, "tf" => NET_TF, "baseMVA" => ps.baseMVA,
        "assumptions" => common_assumptions(), "runs" => runs)
    n = write_json(joinpath(OUTDIR, "study7.json"), doc)
    println("study 7: $status -> artifacts/perc2/study7.json (non-finite values written as null: $n)")
    return doc
end

# ---------------------------------------------------------------------------
# Study 8: parameter sweep VLcease x Ldfact (12 points) on the nearby-fault case,
# plus a batched-integrator check (integrate_batched!) against single runs.
# NOTE: integrate_batched! shares ONE PERC2Params table across scenarios
# (table.params), so VLcease/Ldfact cannot vary per scenario in one batched call;
# the sweep therefore uses sequential single runs, and the batched check varies
# generator inertia H per scenario (a parameter-vector entry) with fixed PERC2 params.
# ---------------------------------------------------------------------------
function study8()
    ps0 = setup_system()
    host = load_table(ps0)[1]
    fault_bus = first(neighbours(ps0, host.bus))
    base = perc1_like()
    vlc = (0.5, 0.6, 0.7, 0.8); lds = (0.5, 0.8, 1.0)
    println("\nStudy 8: sweep VLcease x Ldfact at bus $(host.bus) (assumed: VLrecon = 0.9, Tcease = 0.017, Trecon = 0.5, Tramp = 2, fcease = frecon = 1, VHrecon = 1.2)")
    println("  ", rpad("VLcease", 9), rpad("Ldfact", 8), rpad("trip s", 9), rpad("rec.start", 11), rpad("rec.end", 9), rpad("ramp s", 8),
            rpad("TrampUsed", 10), rpad("min V", 8), rpad("minP MW", 9), rpad("max|dw| Hz", 11))
    table = Any[]
    for v in vlc, l in lds
        st = variant(base, "_vlc$(v)_ld$(l)", "assumed study parameters (sweep point VLcease = $v, Ldfact = $l)";
                     VLcease = v, Ldfact = l, VLrecon = 0.9, Tcease = 0.017, Trecon = 0.5, Tramp = 2.0,
                     fcease = 1.0, frecon = 1.0, VHrecon = 1.2)
        e = run_one(st.name, [(host.bus, st)], fault_bus)
        delete!(e, "trace")
        row = Dict{String,Any}("VLcease" => v, "Ldfact" => l, "status" => e["status"])
        if startswith(e["status"], "ok")
            q = e["perc2"][1]
            row["metrics"] = Dict{String,Any}("trip_time_s" => q["trip_time_s"], "reconnect_start_s" => q["reconnect_start_s"],
                "reconnect_end_s" => q["reconnect_end_s"], "ramp_duration_s" => q["ramp_duration_s"], "TrampUsed_s" => q["TrampUsed_s"],
                "min_V_host" => only(filter(x -> x["bus"] == host.bus, e["bus_voltage"]))["min_V"],
                "min_P_MW" => q["min_P_MW"], "P_final_MW" => q["P_final_MW"], "FracOn_final" => q["FracOn_final"],
                "max_abs_speed_dev_Hz" => e["speed"]["max_abs_speed_dev_Hz"])
            m = row["metrics"]
            println("  ", rpad(v, 9), rpad(l, 8), rpad(fmt(m["trip_time_s"]), 9), rpad(fmt(m["reconnect_start_s"]), 11),
                    rpad(fmt(m["reconnect_end_s"]), 9), rpad(fmt(m["ramp_duration_s"]), 8), rpad(m["TrampUsed_s"], 10),
                    rpad(round(m["min_V_host"]; digits = 3), 8), rpad(round(m["min_P_MW"]; digits = 1), 9),
                    round(m["max_abs_speed_dev_Hz"]; digits = 4))
        else
            println("  ", v, " ", l, " ", e["status"])
        end
        push!(table, row)
    end
    batched = batched_check(host, fault_bus)
    status = all(r -> startswith(r["status"], "ok"), table) && get(batched, "status", "") == "ok" ? "ok" : "partial"
    doc = Dict{String,Any}("study" => 8, "title" => "VLcease x Ldfact sweep and batched check, IEEE 9-bus",
        "status" => status,
        "label" => "assumed study parameters; no facility validation",
        "host_bus" => host.bus, "fault_bus" => fault_bus, "dt" => NET_DT, "tf" => NET_TF,
        "sweep_assumed_fixed_parameters" => Dict{String,Any}("VLrecon" => 0.9, "Tcease" => 0.017, "Trecon" => 0.5, "Tramp" => 2.0,
                                                           "fcease" => 1.0, "frecon" => 1.0, "VHrecon" => 1.2),
        "sweep" => table, "batched_check" => batched,
        "assumptions" => common_assumptions([
            "integrate_batched! shares one PERC2Params table across scenarios: VLcease/Ldfact cannot vary per scenario in one batched call, so the 12-point sweep uses single runs",
            "batched check: scenarios differ by generator inertia H (x0.8, 1.0, 1.2, 1.5) with the same PERC2 set; 2 scenarios are compared with single integrate! runs"]),
        )
    n = write_json(joinpath(OUTDIR, "study8.json"), doc)
    println("study 8: $status -> artifacts/perc2/study8.json (non-finite values written as null: $n)")
    return doc
end

function batched_check(host, fault_bus)
    try
        ps = setup_system()
        ldid = only(filter(l -> l.bus == host.bus, load_table(ps))).id
        qpr = -host.Q_MW / host.P_MW
        st = variant(perc1_like(), "_batched", "assumed study parameters (batched check: PERC1-like + fast reconnect)";
                     VLcease = 0.7, VLrecon = 0.9, Tcease = 0.017, Trecon = 0.5, Tramp = 2.0,
                     fcease = 1.0, frecon = 1.0, VHrecon = 1.2, QPratio = qpr)
        p = perc2_params(st)
        GradPower.add_perc2_load!(ps, host.bus, ldid; fraction = PERC_FRACTION, params = p)
        dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(GradPower.bus_index(ps, fault_bus), FAULT_R, T_FAULT_ON, T_FAULT_OFF))
        scales = (0.8, 1.0, 1.2, 1.5)
        M = length(scales)
        bl = GradPower.BatchedLayout(dp, ps, M)
        gdevs = [i for (i, d) in enumerate(ps.dynamic.devices) if d.dtype isa GradPower.Genrou]
        hidx = [GradPower.parameter_index(ps, gi, :H) for gi in gdevs]
        for m in 1:M, h in hidx
            bl.p[m, h] *= scales[m]
        end
        p_in = copy(bl.p); z_in = copy(bl.z)
        tf = 6.0
        t0 = time()
        rb = GradPower.integrate_batched!(bl, ps, tf; dt = NET_DT, result = true, scenario_ids = collect(1:M))
        tb = time() - t0
        nT = length(rb.times)
        out = Dict{String,Any}("status" => "ok", "M" => M, "H_scales" => collect(scales), "threads" => Threads.nthreads(),
                               "batched_wall_s" => tb, "scenarios" => Any[], "compare_with_single" => Any[])
        gens = [(d.dtype.bus, strip(string(d.dtype.id))) for d in ps.dynamic.devices if d.dtype isa GradPower.Genrou]
        for m in 1:M
            W = reduce(hcat, [collect(Float64, speed(rb, ps.buses[b].i, id; scenario = m)) for (b, id) in gens])
            vm5 = collect(Float64, bus_voltage(rb, host.bus; component = :magnitude, scenario = m))
            push!(out["scenarios"], Dict{String,Any}("scenario" => m, "H_scale" => scales[m],
                "max_abs_speed_dev_Hz" => maximum(abs.(W)) * FNOM, "min_V_host" => minimum(vm5)))
        end
        for m in (1, 4)
            dpm = deepcopy(dp)
            dpm.pvec .= p_in[m, :]; dpm.zvec .= z_in[m, :]
            log = Any[]
            r1 = integrate!(dpm, ps, tf; dt = NET_DT, result = true, perc2_log = log)
            W1 = reduce(hcat, [collect(Float64, speed(r1, ps.buses[b].i, id)) for (b, id) in gens])
            Wb = reduce(hcat, [collect(Float64, speed(rb, ps.buses[b].i, id; scenario = m)) for (b, id) in gens])
            v1 = collect(Float64, bus_voltage(r1, host.bus; component = :magnitude))
            vb = collect(Float64, bus_voltage(rb, host.bus; component = :magnitude, scenario = m))
            fr = [e.FracOn for e in log]
            push!(out["compare_with_single"], Dict{String,Any}("scenario" => m, "H_scale" => scales[m],
                "max_abs_diff_speed_pu" => maximum(abs.(W1 .- Wb)), "max_abs_diff_Vhost_pu" => maximum(abs.(v1 .- vb)),
                "single_min_FracOn" => minimum(fr), "single_ceased" => minimum(fr) < 0.999))
        end
        println("  batched check (M = $M, H scales $(collect(scales)), $(Threads.nthreads()) threads, $(round(tb; digits = 2)) s):")
        for c in out["compare_with_single"]
            println("    scenario $(c["scenario"]): max|dw batched - single| = $(c["max_abs_diff_speed_pu"]), max|dV| = $(c["max_abs_diff_Vhost_pu"]), PERC2 ceased in single run: $(c["single_ceased"])")
        end
        for sc in out["scenarios"]
            println("    scenario $(sc["scenario"]) H x$(sc["H_scale"]): max|dw| = $(round(sc["max_abs_speed_dev_Hz"]; digits = 4)) Hz")
        end
        return out
    catch e
        println("  batched check failed: ", first(sprint(showerror, e), 300))
        return Dict{String,Any}("status" => "error: " * first(sprint(showerror, e), 400))
    end
end

# ---------------------------------------------------------------------------
# Study 9: frequency ceasing after a generator trip (the bus-frequency
# estimate must move after the trip).
# ---------------------------------------------------------------------------
const T_GEN_TRIP = 1.0
const S9_TF = 20.0

function study9()
    ps = setup_system()
    host = load_table(ps)[1]
    gens = [(d.dtype.bus, strip(string(d.dtype.id))) for d in ps.dynamic.devices if d.dtype isa GradPower.Genrou]
    gbus = ps.buses[gens[3][1]].i             # smallest machine (bus 3, 85 MW in the case)
    trip = (bus = gbus, id = gens[3][2], t = T_GEN_TRIP)
    println("\nStudy 9: generator trip at bus $gbus (id $(trip.id)) at t = $T_GEN_TRIP s; PERC2 at bus $(host.bus) ($(PERC_FRACTION * host.P_MW) MW)")
    base = perc1_like()
    # observation run: FL check off (FLflag = false) -> bus frequency estimate nadir without ceasing
    obs_set = variant(base, "+obs", "assumed study parameters (frequency observation, FLflag off)")
    zip = run_one("zip_baseline", Tuple[], nothing; trip_gen = trip, tf = S9_TF)
    obs = run_one("perc2_no_freq_cease", [(host.bus, obs_set)], nothing; trip_gen = trip, tf = S9_TF,
                  note = "FLflag off; used to find the frequency nadir seen by the PERC2 estimator")
    runs = Any[zip, obs]
    if !startswith(obs["status"], "ok")
        write_study9(runs, nothing, "partial", host, trip); return
    end
    f = obs["trace"]["perc2_bus$(host.bus)_f_Hz"]; t = obs["trace"]["t"]
    after = t .>= T_GEN_TRIP
    nadir = minimum(f[after]); fmax_early = maximum(f[after])
    println("  estimated frequency at bus $(host.bus): min $(round(nadir; digits = 4)) Hz, max after trip $(round(fmax_early; digits = 4)) Hz")
    if maximum(abs.(f .- FNOM)) < 1e-6
        println("  frequency estimate did not move: skipping the FL runs")
        write_study9(runs, nothing, "skipped: frequency estimate did not move (ftf stays 0)", host, trip); return
    end
    margin = 0.02
    for (nm, thr, d) in (("FLC1_below_nadir", nadir - margin, "FLC1 = nadir - $margin Hz: no cease expected"),
                         ("FLC1_above_nadir", nadir + margin, "FLC1 = nadir + $margin Hz: permanent cease expected"))
        st = variant(base, "+" * nm, "assumed study parameters (FL cease, $d)";
                     FLflag = true, FLC1 = thr, TFLC1 = 0.1)
        push!(runs, run_one(nm, [(host.bus, st)], nothing; trip_gen = trip, tf = S9_TF, note = d))
    end
    write_study9(runs, (; nadir, margin), "ok", host, trip)
end

function write_study9(runs, nadir, status0, host, trip)
    for r in runs
        startswith(get(r, "status", ""), "ok") || (println("  ", r["name"], ": ", get(r, "status", "?")); continue)
        sp = r["speed"]; pm = haskey(r, "perc2") ? r["perc2"][1] : nothing
        println("  ", rpad(r["name"], 22), "max|dw|=", round(sp["max_abs_speed_dev_Hz"]; digits = 4), " Hz",
                "  mean-speed min/final = ", round(sp["mean_speed_dev_Hz_min"]; digits = 4), "/", round(sp["mean_speed_dev_Hz_final"]; digits = 4),
                pm === nothing ? "" : "  trip=" * fmt(pm["trip_time_s"]) * "  reconnect=" * fmt(pm["reconnect_start_s"]) *
                "  FracOn end=" * string(round(pm["FracOn_final"]; digits = 3)) * "  P end=" * string(round(pm["P_final_MW"]; digits = 2)) * " MW")
    end
    status = status0 != "ok" ? status0 : all(r -> startswith(r["status"], "ok"), runs) ? "ok" : "partial"
    doc = Dict{String,Any}("study" => 9, "title" => "Frequency ceasing after generator trip, IEEE 9-bus",
        "status" => status,
        "label" => "assumed study parameters; no facility validation",
        "perc2_bus" => host.bus, "perc2_fraction" => PERC_FRACTION,
        "generator_trip" => Dict{String,Any}("bus" => trip.bus, "id" => trip.id, "t" => trip.t),
        "estimated_frequency_nadir_Hz" => nadir === nothing ? nothing : nadir.nadir,
        "FLC1_margin_Hz" => nadir === nothing ? nothing : nadir.margin,
        "dt" => NET_DT, "tf" => S9_TF,
        "assumptions" => [
            "all parameters are assumed study parameters; FLC1 chosen relative to the simulated nadir (no real setting)",
            "TFLC1 = 0.1 s (assumed; PERC2 default 100 s would never trip within the run)",
            "frequency estimate: the bus-frequency estimate used by the PERC2 device at the PERC2 bus",
            "Tt floor applied: false (dt = 1/120 s, Tt = 0.02 s)",
            "frequency cease never reconnects (PERC2 level 2)",
            "no fault is applied; the only disturbance is the generator trip"],
        "runs" => runs)
    n = write_json(joinpath(OUTDIR, "study9.json"), doc)
    println("study 9: $status -> artifacts/perc2/study9.json (non-finite values written as null: $n)")
end

function main_network(args)
    mkpath(OUTDIR)
    which = isempty(args) ? ["6", "7", "8", "9"] : args
    for k in which
        try
            k == "6" && study6()
            k == "7" && study7()
            k == "8" && study8()
            k == "9" && study9()
        catch e
            @error "study $k failed" exception = (e, catch_backtrace())
        end
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && main_network(ARGS)
