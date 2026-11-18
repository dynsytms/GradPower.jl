# Compare GradPower against stored PSS/E trajectories.
#
# A case is `data/<tag>.toml` (network, dynamic data, event, power-flow options,
# machine ids, provenance) plus `data/<tag>.npz` (the PSS/E trajectories); both
# are written by `validation/psse/export_reference.py`. `check_case` runs
# GradPower on the case and tests the error metrics against per-case ceilings.
#
# The metrics are those of the offline cross-check (`validation/psse/README.md`):
# - angle: largest difference in the change of each machine's angle relative
#   to machine 1 from its pre-event value, as % of PSS/E's largest such change;
# - speed, Efd, Pm: largest difference in the change from the initial value, as
#   % of PSS/E's largest change, leaving out 0.05 s around each switching instant
#   (the two codes resolve a switching instant differently);
# - bus voltage: largest |V| difference after the event, same exclusion;
# - power flow: largest |V| and angle difference of the initial solution.
module PSSEReference

using GradPower
using NPZ
using TOML
using Test
using Printf
using Logging

export check_case, load_case, run_gradpower, compare_runs, ANGLE_TARGET_PCT

const DATA = joinpath(@__DIR__, "data")

"Angle error bound the project validates against (the ANDES-level bound, 1/1920 s)."
const ANGLE_TARGET_PCT = 4.5

"Steps per second GradPower is run at; the references were made at the same rate."
const RATE = 1920

function load_case(tag::AbstractString; dir=DATA)
    case = TOML.parsefile(joinpath(dir, "$tag.toml"))
    z = npzread(joinpath(dir, "$tag.npz"))
    r = case["reference"]
    ref = (t=Float64.(z["t"]), bus_i=Int.(z["bus_i"]), vm_bus=Int.(z["vm_bus"]),
           pf_vm=Float64.(z["pf_vm"]), pf_va=Float64.(z["pf_va"]),
           gens=collect(zip(Int.(r["gen_bus"]), String.(r["gen_id"]))),
           delta=Float64.(z["delta"]), omega=Float64.(z["omega"]), efd=Float64.(z["efd"]),
           pm=Float64.(z["pmech"]), vm=Float64.(z["vm"]))
    return case, ref
end

"""
    run_gradpower(case; dir=DATA, rate=RATE)

Run the case's event in GradPower and return the same quantities the reference
holds: per machine angle (rad), speed deviation, Efd and Pm (machine base), and
every bus voltage magnitude.
"""
function run_gradpower(case; dir=DATA, rate=RATE)
    pf, ev = case["powerflow"], case["event"]
    sys = with_logger(NullLogger()) do
        s = GradPower.from_psse(joinpath(dir, case["raw"]), joinpath(dir, case["dyr"]);
                                static_gen_mode=Symbol(pf["static_gen_mode"]))
        GradPower.build_network!(s)
        GradPower.runpf!(s; verbose=false, enforce_q_limits=pf["enforce_q_limits"])
        s
    end
    dp = GradPower.DynamicProblem(sys)
    GradPower.initialize_dynamics!(dp, sys)
    if ev["kind"] == "trip"
        GradPower.add_trip_event!(sys, GradPower.create_trip_line_event(sys, ev["a"], ev["b"], ev["t_on"]))
    elseif ev["kind"] == "fault"
        GradPower.add_event!(sys, GradPower.ContingencyEvent(sys.busmap[ev["a"]], ev["r"], ev["t_on"], ev["t_off"]))
    else
        error("unknown event kind $(ev["kind"])")
    end
    u0 = copy(dp.uvec)                     # control values of machines with no controller
    tv, tr = GradPower.integrate!(dp, sys, Float64(ev["t_final"]); dt=1 / rate)

    dyn = sys.dynamic
    gi = [i for (i, d) in enumerate(dyn.devices) if d.dtype isa GradPower.Genrou]
    dev = dyn.devices[gi]
    nt, nb = length(tv), length(sys.buses)
    off = dyn.diff_dim + dyn.alg_dim
    # ctrl slots of a machine are [e_fd, p_m]; uvec_idx is the state driving a slot, 0 if none
    ctrl(d, slot) = (src = dyn.uvec_idx[d.ctrl_ptr + slot];
                     src > 0 ? tr[src, :] : fill(u0[d.ctrl_ptr + slot], nt))
    mbase = [sys.gens[dyn.map.gen[i]].mbase for i in gi]
    # Genrou differential block: e_qp, e_dp, phi_1d, phi_2q, w, delta
    delta = permutedims(tr[[d.diff_ptr + 5 for d in dev], :])
    omega = permutedims(tr[[d.diff_ptr + 4 for d in dev], :])
    efd = reduce(hcat, [ctrl(d, 0) for d in dev])
    pm = reduce(hcat, [ctrl(d, 1) for d in dev]) .* (sys.baseMVA ./ mbase')   # system -> machine base
    vr = tr[[off + 2i - 1 for i in 1:nb], :]
    vi = tr[[off + 2i for i in 1:nb], :]
    bus_i = [b.i for b in sys.buses]
    return (t=collect(Float64, tv), bus_i=bus_i, vm_bus=bus_i,
            pf_vm=[b.v0m for b in sys.buses], pf_va=[b.v0a for b in sys.buses],
            gens=[(sys.buses[dyn.map.bus[i]].i, String(strip(replace(d.dtype.id, "'" => ""))))
                  for (i, d) in zip(gi, dev)],
            delta=delta, omega=omega, efd=efd, pm=pm, vm=permutedims(sqrt.(vr .^ 2 .+ vi .^ 2)))
end

# linear interpolation onto tt, constant beyond the ends (numpy.interp)
function interp(t, y, tt)
    out = similar(tt, Float64)
    n = length(t)
    for (j, x) in enumerate(tt)
        if x <= t[1]
            out[j] = y[1]
        elseif x >= t[n]
            out[j] = y[n]
        else
            k = searchsortedlast(t, x)
            w = (x - t[k]) / (t[k + 1] - t[k])
            out[j] = (1 - w) * y[k] + w * y[k + 1]
        end
    end
    return out
end

function on_grid(t, Y, tt)
    t = Float64.(t)
    for k in findall(<=(0), diff(t))        # PSS/E writes two rows at a switching instant
        t[k + 1:end] .+= 1e-9
    end
    return reduce(hcat, [interp(t, Y[:, k], tt) for k in 1:size(Y, 2)]; init=zeros(length(tt), 0))
end

function median_(x)
    s = sort(x); n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

function sample_hz(t)
    d = filter(>(1e-7), diff(Float64.(t)))
    return isempty(d) ? 240.0 : 1 / median_(d)
end

wrap(a) = mod(a + pi, 2pi) - pi

"""
    compare_runs(gp, ref, event) -> Dict{String,Float64}

Error metrics of `gp` against `ref` (both as returned by `run_gradpower` /
`load_case`). Percentages are of the reference's own excursion; a metric whose
reference excursion is below 1e-6 is NaN (nothing to compare: e.g. Efd on a
machine with no exciter).
"""
function compare_runs(gp, ref, event)
    m = Dict{String,Float64}()
    # power flow, on the buses both sides have
    cb = [b for b in gp.bus_i if b in ref.bus_i]
    gi = [findfirst(==(b), gp.bus_i) for b in cb]
    ri = [findfirst(==(b), ref.bus_i) for b in cb]
    m["pf_dvm"] = maximum(abs.(gp.pf_vm[gi] .- ref.pf_vm[ri]))
    da = (gp.pf_va[gi] .- gp.pf_va[gi][1]) .- (ref.pf_va[ri] .- ref.pf_va[ri][1])
    m["pf_dva_deg"] = rad2deg(maximum(abs.(wrap.(da))))
    # machines, matched by (bus, id)
    cg = [g for g in gp.gens if g in ref.gens]
    m["n_gen"] = length(cg)
    length(cg) == length(gp.gens) == length(ref.gens) ||
        error("machines differ: GradPower $(gp.gens), reference $(ref.gens)")
    mg = [findfirst(==(g), gp.gens) for g in cg]
    mr = [findfirst(==(g), ref.gens) for g in cg]
    t_end = min(gp.t[end], ref.t[end])
    hz = round(Int, min(240.0, sample_hz(gp.t), sample_hz(ref.t)))
    tt = collect(range(0, t_end; length=round(Int, t_end * hz) + 1))
    t_sw = event["kind"] == "fault" ? [event["t_on"], event["t_off"]] : [event["t_on"]]
    quiet = trues(length(tt))
    for ts in t_sw
        quiet .&= abs.(tt .- ts) .> 0.05
    end
    G = Dict(k => on_grid(gp.t, getfield(gp, k)[:, mg], tt) for k in (:delta, :omega, :efd, :pm))
    R = Dict(k => on_grid(ref.t, getfield(ref, k)[:, mr], tt) for k in (:delta, :omega, :efd, :pm))
    rel(D) = rad2deg.(D .- D[:, 1:1])
    ag = rel(G[:delta]); ag = ag .- ag[1:1, :]
    ar = rel(R[:delta]); ar = ar .- ar[1:1, :]
    m["swing_gp_deg"] = maximum(abs.(ag))
    m["swing_ref_deg"] = maximum(abs.(ar))
    m["angle_pct"] = 100 * maximum(abs.(ag .- ar)) / max(m["swing_ref_deg"], 1e-9)
    m["sync_gp"] = Float64(maximum(abs.(ag[end, :])) < 180)
    m["sync_ref"] = Float64(maximum(abs.(ar[end, :])) < 180)
    for (k, name) in ((:omega, "speed"), (:efd, "efd"), (:pm, "pm"))
        a = G[k] .- G[k][1:1, :]
        b = R[k] .- R[k][1:1, :]
        ex = maximum(abs.(b))
        e = maximum(abs.(a .- b)[quiet, :])
        m[name * "_pct"] = ex > 1e-6 ? 100 * e / ex : NaN
        m[name * "_abs"] = e
    end
    # bus voltage after the event, on the buses both sides recorded
    cv = [b for b in gp.vm_bus if b in ref.vm_bus]
    Vg = on_grid(gp.t, gp.vm[:, [findfirst(==(b), gp.vm_bus) for b in cv]], tt)
    Vr = on_grid(ref.t, ref.vm[:, [findfirst(==(b), ref.vm_bus) for b in cv]], tt)
    post = quiet .& (tt .> event["t_on"])
    m["v_err"] = maximum(abs.(Vg .- Vr)[post, :])
    return m
end

const DEFAULT_TOL = Dict("pf_dvm" => 1e-4, "pf_dva_deg" => 1e-3)

"""
    check_case(tag; tol, broken=Dict(), dir=DATA)

Run `tag` and test each metric in `tol` (plus the power-flow defaults) against
its ceiling, the angle error against `ANGLE_TARGET_PCT`, and that both codes
stay in synchronism. Metrics named in `broken` (metric => ceiling) are known
failures, tested with `@test_broken`: when a fix makes one pass, the test
reports an unexpected pass and the entry should move to `tol`.
A metric that is NaN (no reference excursion) must not be given a ceiling.
"""
function check_case(tag; tol::AbstractDict, broken::AbstractDict=Dict{String,Float64}(), dir=DATA)
    case, ref = load_case(tag; dir)
    @testset "$tag" begin
        @test case["reference"]["init_suspect"] == 0     # PSS/E started from equilibrium
        gp = run_gradpower(case; dir)
        m = compare_runs(gp, ref, case["event"])
        @info @sprintf("%-22s angle %5.2f%%  speed %s  Efd %s  Pm %s  dV %.4f  PF dVm %.1e",
                       tag, m["angle_pct"], pct(m["speed_pct"]), pct(m["efd_pct"]), pct(m["pm_pct"]),
                       m["v_err"], m["pf_dvm"])
        @test m["sync_gp"] == m["sync_ref"] == 1.0
        @test m["angle_pct"] <= ANGLE_TARGET_PCT
        for (k, ceil_) in merge(DEFAULT_TOL, tol)
            @test isfinite(m[k]) && m[k] <= ceil_
        end
        for (k, ceil_) in broken
            @test_broken m[k] <= ceil_
        end
    end
end

pct(x) = isnan(x) ? "  n/a " : @sprintf("%5.2f%%", x)

end # module
