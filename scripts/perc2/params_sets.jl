# PERC2 parameter sets for the case studies.
#
# ALL sets are ASSUMED STUDY PARAMETERS. The Appendix H sets are NERC Appendix H
# example settings (assumed): a mapping of the PERC1-style tables in the NERC
# Data Center Load Modeling Technical Reference (July 30, 2026), Appendix H,
# onto PERC2 parameter names. They are not facility, manufacturer-certified or
# validated settings.
#
# Mapping (Appendix H -> PERC2 CON names):
#   Vcease      -> VLcease
#   Tcease      -> Tcease
#   Tdelay      -> Tdelay
#   Vrecon      -> VLrecon
#   Trecon      -> Trecon
#   Tramp_OEM   -> Tramp          (the model ramps for TrampUsed = Tramp * Ldfact;
#                                   with the default Ldfact = 0.8 the effective ramp
#                                   is 0.8 * Tramp_OEM, so Ldfact = 1.0 reproduces
#                                   the published ramp time)
#   fcease/frecon -> given only in some sources; otherwise 1.0 / 1.0
#                    (all load ceases, all load returns)
# Everything else is the PERC2 default (VLMflag only; long/high/frequency checks
# inactive).
#
# Safety rules: VLrecon <= 0.95 and VLrecon >= the highest active
# long-cease voltage, so the presets cannot cycle between cease and reconnect.
# Where a table gives a range ("0.6-0.75"), the arithmetic midpoint is used.

const PERC2_SET_LABEL = "NERC Appendix H example settings (assumed)"
const NERC_SOURCE = "NERC Data Center Load Modeling Technical Reference, July 30, 2026, Appendix H"

# PERC1 has no upper reconnect bound, so every Appendix H mapping sets
# VHrecon = VHcease (= 1.2 pu, the PERC2 default VHcease): the reconnect window
# is then not narrowed by the PERC2 default VHrecon = 1.1 pu. With 1.1 pu the
# IEEE 9 study strands the load after it ceases (post-trip bus voltage ~1.13 pu
# > VHrecon), see study 6 sensitivity runs ("_vhrecon1.1").
_set(name, kwargs; label = PERC2_SET_LABEL, source, mapping = "", notes = String[],
     published = (;)) =
    (; name, label, source, published,
       mapping = mapping * "; VHrecon = VHcease = 1.2 (PERC1 has no upper reconnect bound)",
       notes = vcat(notes, ["VHrecon = 1.2 pu assumed (PERC2 default 1.1 would block reconnect when the bus voltage rises above 1.1 pu after the cease)"]),
       kwargs = merge(kwargs, (; VHrecon = 1.2)))

# ---- non-Appendix sets ------------------------------------------------------

default_spec() = (; name = "default_spec",
    label = "assumed study parameters (PERC2 spec defaults)",
    source = "PERC2Params() defaults (PERC2 specification table)",
    mapping = "none", notes = ["PERC2Params() defaults, VLM check only"],
    published = (;), kwargs = (;))

perc1_like() = (; name = "perc1_like",
    label = "assumed study parameters (PERC1-equivalent)",
    source = "PERC2 with only the momentary low-voltage cease enabled (PERC1-equivalent)",
    mapping = "none", notes = ["identical to default_spec by construction (VLM only)"],
    published = (;), kwargs = (;))

# ---- Appendix H mappings ----------------------------------------------------

vertiv_today() = _set("vertiv_today",
    (; VLcease = 0.70, Tcease = 0.017, Tdelay = 0.0, VLrecon = 0.85, Trecon = 5.0,
       Tramp = 15.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.1 (Today's UPS, typical column)",
    mapping = "Vcease->VLcease, Tcease, Tdelay, Vrecon->VLrecon, Trecon, Tramp_OEM->Tramp; fcease=frecon=1.0 (not given)",
    notes = ["Tramp_OEM = 15 s; TrampUsed = Tramp*Ldfact = 15*0.8 = 12 s at default Ldfact",
             "published ranges: Vcease 0.70-0.90, Vrecon 0.85-0.95, Trecon 2.25-10, Tramp 1.5-30"],
    published = (; Vcease = 0.70, Tcease = 0.017, Tdelay = 0.0, Vrecon = 0.85, Trecon = 5.0, Tramp_OEM = 15.0))

vertiv_frt() = _set("vertiv_frt",
    (; VLcease = 0.70, Tcease = 0.017, Tdelay = 0.0, VLrecon = 0.85, Trecon = 5.0,
       Tramp = 15.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.2 (Fault Ride Through UPS, suggested default column)",
    mapping = "as vertiv_today; fcease=frecon=1.0 (not given)",
    notes = ["Table H.2 suggested defaults equal Table H.1 typical values; the FRT 'goal' (requalify, restart, ramp in <1 s) is only reached at the minimum column, see vertiv_frt_goal",
             "TrampUsed = Tramp*Ldfact = 12 s at default Ldfact"],
    published = (; Vcease = 0.70, Tcease = 0.017, Tdelay = 0.0, Vrecon = 0.85, Trecon = 5.0, Tramp_OEM = 15.0))

vertiv_frt_goal() = _set("vertiv_frt_goal",
    (; VLcease = 0.30, Tcease = 0.004, Tdelay = 0.0, VLrecon = 0.60, Trecon = 0.80,
       Tramp = 0.10, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.2 (Fault Ride Through UPS, MINIMUM column = <1 s requalify/restart/ramp goal)",
    mapping = "as vertiv_today but every parameter at the table minimum; fcease=frecon=1.0",
    notes = ["not a typical value: all entries at the published Minimum column, representing the 'goal' behaviour",
             "TrampUsed = Tramp*Ldfact = 0.08 s at default Ldfact; Tcease = 0.004 s is about one 1/240 s step (4.17 ms): timer resolved at step granularity"],
    published = (; Vcease = 0.30, Tcease = 0.004, Tdelay = 0.0, Vrecon = 0.60, Trecon = 0.80, Tramp_OEM = 0.10))

eaton() = _set("eaton",
    (; VLcease = 0.70, Tcease = 0.001, Tdelay = 0.001, VLrecon = 0.90, Trecon = 5.0,
       Tramp = 5.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.4 (typical Eaton UPS without fault ride through)",
    mapping = "Vcease->VLcease, Tcease, Tdelay, Vrecon->VLrecon, Trecon, Tramp_OEM->Tramp; fcease=frecon=1.0",
    notes = ["Tcease = Tdelay = 1 ms is below one step: the timers act at step granularity",
             "TrampUsed = Tramp*Ldfact = 4 s at default Ldfact",
             "published ranges: Vcease 0.7-0.9, Trecon 0.2-*, Tramp 1-120"],
    published = (; Vcease = 0.7, Tcease = 0.001, Tdelay = 0.001, Vrecon = 0.9, Trecon = 5.0, Tramp_OEM = 5.0))

eaton_frt() = _set("eaton_frt",
    (; VLcease = 0.50, Tcease = 0.001, Tdelay = 0.001, VLrecon = 0.90, Trecon = 0.2,
       Tramp = 0.05, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.5 (typical Eaton UPS with fault ride through)",
    mapping = "as eaton; fcease=frecon=1.0",
    notes = ["Tcease = Tdelay = 1 ms is below one step: the timers act at step granularity",
             "TrampUsed = Tramp*Ldfact = 0.04 s at default Ldfact (about a step response at dt = 1/120 s)",
             "published ranges: Vcease 0.5-0.9, Trecon 0.1-*, Tramp 0.02-120"],
    published = (; Vcease = 0.5, Tcease = 0.001, Tdelay = 0.001, Vrecon = 0.9, Trecon = 0.2, Tramp_OEM = 0.05))

# Schneider Table H.6: the typical Tdelay (0.001 s) lies OUTSIDE the published
# [Minimum, Maximum] = [0, 0] of the same row. Kept as published (typical value),
# not "fixed"; flagged in the `inconsistency` metadata field.
schneider() = merge(_set("schneider",
    (; VLcease = 0.50, Tcease = 0.003, Tdelay = 0.001, VLrecon = 0.50, Trecon = 1.0,
       Tramp = 1.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.6 (Schneider UPS, typical column)",
    mapping = "Vcease->VLcease, Tcease, Tdelay, Vrecon->VLrecon, Trecon, Tramp_OEM->Tramp; fcease=frecon=1.0",
    notes = ["INCONSISTENCY (as published): Tdelay typical = 0.001 s but Minimum = Maximum = 0 in the same row; typical kept as published",
             "VLrecon = VLcease = 0.5: reconnect window is strict (Vtf > VLrecon); the model raises VLrecon to VLcease anyway",
             "TrampUsed = Tramp*Ldfact = 0.8 s at default Ldfact",
             "published ranges: Vcease 0.5-0.85, Tcease 0.003-0.2, Trecon 0-30, Tramp 1-300"],
    published = (; Vcease = 0.5, Tcease = 0.003, Tdelay = 0.001, Vrecon = 0.5, Trecon = 1.0, Tramp_OEM = 1.0)),
    (; inconsistency = "Table H.6 Tdelay: typical 0.001 s is outside the published min/max (0, 0); typical value kept as published"))

rack_bbu() = _set("rack_bbu",
    (; VLcease = 0.675, Tcease = 0.175, Tdelay = 0.001, VLrecon = 0.90, Trecon = 15.5,
       Tramp = 5.5, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.7 (BBU, OCP rack-level distributed design; ranges -> midpoints)",
    mapping = "ranges replaced by arithmetic midpoints: Vcease 0.6-0.75 -> 0.675, Tcease 0.1-0.25 -> 0.175, Vrecon 0.85-0.95 -> 0.90, Trecon 1-30 -> 15.5, Tramp 1-10 -> 5.5",
    notes = ["midpoints are an ASSUMPTION (the table gives only ranges)",
             "TrampUsed = Tramp*Ldfact = 4.4 s at default Ldfact"],
    published = (; Vcease = "0.6-0.75", Tcease = "0.1-0.25", Tdelay = 0.001, Vrecon = "0.85-0.95", Trecon = "1-30", Tramp_OEM = "1-10"))

toshiba() = _set("toshiba",
    (; VLcease = 0.80, Tcease = 0.02, Tdelay = 0.0005, VLrecon = 0.90, Trecon = 3.0,
       Tramp = 10.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.8 (Toshiba UPS, suggested generic value)",
    mapping = "Vcease->VLcease, Tcease, Tdelay, Vrecon->VLrecon, Trecon, Tramp_OEM->Tramp; fcease=frecon=1.0",
    notes = ["Tdelay = 0.5 ms is below one step: timer acts at step granularity",
             "TrampUsed = Tramp*Ldfact = 8 s at default Ldfact",
             "published ranges: Vcease 0.78-0.8, Vrecon 0.9-0.92, Trecon 1-3600, Tramp 1-30"],
    published = (; Vcease = 0.8, Tcease = 0.02, Tdelay = 0.0005, Vrecon = 0.9, Trecon = 3.0, Tramp_OEM = 10.0))

equinix() = _set("equinix",
    (; VLcease = 0.90, Tcease = 0.1, Tdelay = 0.0, VLrecon = 0.95, Trecon = 10.0,
       Tramp = 16.0, fcease = 1.0, frecon = 1.0);
    source = NERC_SOURCE * ", Table H.10 (Equinix, suggested generic value)",
    mapping = "Vcease->VLcease, Tcease, Tdelay, Vrecon->VLrecon, Trecon, Tramp_OEM->Tramp; fcease=frecon=1.0",
    notes = ["table note: some entries are hardware response, not settings (e.g. Tcease)",
             "VLrecon = 0.95 is at the safety limit: the load only returns if the post-fault voltage exceeds 0.95 pu",
             "TrampUsed = Tramp*Ldfact = 12.8 s at default Ldfact"],
    published = (; Vcease = 0.9, Tcease = 0.1, Tdelay = 0.0, Vrecon = 0.95, Trecon = 10.0, Tramp_OEM = 16.0))

const SET_BUILDERS = (default_spec, perc1_like, vertiv_today, vertiv_frt, vertiv_frt_goal,
                      eaton, eaton_frt, schneider, rack_bbu, toshiba, equinix)

all_sets() = [b() for b in SET_BUILDERS]
get_set(name) = only(filter(s -> s.name == String(name), all_sets()))

"""Check the safety rules (no cease/reconnect cycling); returns a Vector of violation strings (empty = ok)."""
function safety_violations(p)
    v = String[]
    p.VLrecon <= 0.95 || push!(v, "VLrecon = $(p.VLrecon) > 0.95")
    if p.VLLflag
        act = [c for (c, t) in ((p.VLC1, p.TVLC1), (p.VLC2, p.TVLC2), (p.VLC3, p.TVLC3), (p.VLC4, p.TVLC4)) if t > 0]
        !isempty(act) && p.VLrecon < maximum(act) &&
            push!(v, "VLrecon = $(p.VLrecon) < highest active long-cease voltage $(maximum(act))")
    end
    return v
end

"""Build the `PERC2Params` for a set (extra kwargs override). Throws on safety violations."""
function perc2_params(set; kwargs...)
    p = GradPower.PERC2Params(; set.kwargs..., kwargs...)
    bad = safety_violations(p)
    isempty(bad) || error("parameter set $(set.name) violates safety rules: " * join(bad, "; "))
    return p
end

"""Plain-Dict view of the effective parameters (all fields) for JSON output."""
params_dict(p) = Dict{String,Any}(string(f) => getfield(p, f) for f in fieldnames(typeof(p)))

function set_metadata(set, p)
    d = Dict{String,Any}(
        "name" => set.name, "label" => set.label, "source" => set.source,
        "mapping" => set.mapping, "notes" => collect(set.notes),
        "TrampUsed_s" => p.Tramp * (p.Ldfact > 0 ? p.Ldfact : 0.8),
        "params" => params_dict(p),
        "published_typical" => Dict{String,Any}(string(k) => v for (k, v) in pairs(set.published)))
    haskey(set, :inconsistency) && (d["inconsistency"] = set.inconsistency)
    return d
end
