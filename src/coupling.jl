# Phase 1.5b of ROADMAP.md: device coupling graph.
#
# Replaces the hand-matched governor wiring in src/GradPower.jl:289-318
# with a generic `wire_controls!` that uses traits to find each
# controller's target and resolve which ctrl slots / state indices to
# wire together.
#
# The traits below are the contract; any new controller type (TGOV1, SEXS,
# PSS, ...) drops in by adding three method definitions and registering
# its table in src/tables/<dev>.jl. No changes to set_dynamics! needed.

# ------------------------------------------------------------------------
# Trait fallbacks
#
# Generators / loads define `attaches_to(T) === nothing` because they are
# not controllers — `wire_controls!` skips devices whose trait returns
# nothing. Concrete controllers override.
# ------------------------------------------------------------------------

attaches_to(::Type) = nothing
attaches_to(::T) where {T} = attaches_to(T)

# Signals a controller writes into its target's ctrl slots.
# Returns a tuple of `(slot_index_on_target, value_kind)` per signal.
# `value_kind` is `:alg_first` (governor: write generator's ctrl slot to
# point at governor's alg state) or `:diff_first` (exciter: write target
# ctrl slot to point at exciter's first diff state — currently unused;
# kept for future controllers whose output is a diff state).
produces_signals(::Type) = ()
produces_signals(::T) where {T} = produces_signals(T)

# Signals a controller reads from its target's state.
# Returns a tuple of `(slot_index_on_controller, state_kind)` per signal.
# `state_kind` is one of `:w` (rotor speed, target's diff state 5 in Genrou)
# or `:v_t` (terminal voltage, target's alg outputs).
consumes_signals(::Type) = ()
consumes_signals(::T) where {T} = consumes_signals(T)

# Strip whitespace and apostrophes from a PSS/E device id so e.g.
# "1" matches "'1 '". PSS/E .dyr files vary in their quoting convention
# even within a single file (IEESGO records often unquote what GENROU
# records quote), so normalization is required for cross-device matching.
_normalize_id(s::AbstractString) = strip(replace(replace(s, "'" => ""), " " => ""))

# Helper: position (1-based) of the generator's `w` state within its diff block.
# Genrou layout: diff_ptr+0..5 → e_qp, e_dp, phi_1d, phi_2q, w, delta.
# So w is at offset 4 (zero-based) → 5 (one-based).
w_offset(::Type{Genrou}) = 4  # zero-based offset added to diff_ptr

# ------------------------------------------------------------------------
# IEESGO traits
#
# IEESGO is a governor:
#   - attaches to Genrou (matched by bus + id)
#   - produces p_m → writes into the generator's ctrl slot 2 (the p_m slot)
#   - consumes w  → reads from the generator's diff state 5 (w)
# ------------------------------------------------------------------------

attaches_to(::Type{IEESGO}) = Genrou

# Each entry: (target_ctrl_slot_offset_zero_based, governor_signal_source)
# The governor's `p_m` lives at index `diff_dim + alg_ptr` in the global
# z vector (it's the IEESGO's first algebraic state). The target slot is
# offset 1 from the generator's ctrl_ptr (since Genrou ctrl is [e_fd, p_m]).
produces_signals(::Type{IEESGO}) = (
    (target_ctrl_offset = 1,  # write to generator's ctrl_ptr + 1 (p_m slot)
     source_kind        = :alg_first),  # source: governor's first alg state
)

# Each entry: (controller_ctrl_slot_offset_zero_based, target_state_kind)
consumes_signals(::Type{IEESGO}) = (
    (ctrl_offset = 0,        # write to governor's ctrl_ptr + 0 (w slot)
     state_kind  = :w),      # source: generator's w state
)

# ------------------------------------------------------------------------
# Generic wire_controls!
#
# One pass over psd.devices. For each device that is a controller
# (`attaches_to !== nothing`), find its target by (bus, id), then walk
# its produced and consumed signals to populate `uvec_idx`.
#
# This replaces the hand-coded governor block in src/GradPower.jl and
# fixes the off-by-one bug (Phase 2.0a) by construction: the target ctrl
# slot is looked up via `produces_signals(...).target_ctrl_offset` instead
# of being hard-coded.
# ------------------------------------------------------------------------

"""
    wire_controls!(psd, dmap)

Walk `psd.devices`; for each controller device (one whose
`attaches_to(typeof(d.dtype))` returns a concrete generator type), locate
its target generator by (bus, id), and populate `psd.uvec_idx` for both
the produced signal (target reads from controller) and the consumed
signal (controller reads from target).

Side effects:
  - mutates `psd.uvec_idx`
  - mutates `dmap.gen[i]` so the controller's init routine can read pg/qg
    from the same static generator as its target

Returns nothing.
"""
function wire_controls!(psd, dmap)
    for (i, device) in enumerate(psd.devices)
        target_class = attaches_to(typeof(device.dtype))
        target_class === nothing && continue

        # Find target device by (bus, id). Bus comparison uses the raw bus
        # number from the .dyr (NOT the mapped internal index), matching
        # the prior hand-coded logic at src/GradPower.jl:301-308 (which
        # compared `dev.dtype.bus == bus_idx` where `bus_idx = device.dtype.bus`).
        # IDs are normalized via _normalize_id (strip spaces + apostrophes)
        # so e.g. "1" (IEESGO) matches "'1 '" (Genrou) — the .dyr quoting
        # convention differs between record types in some PSS/E files.
        bus_match = device.dtype.bus
        id_match  = _normalize_id(device.dtype.id)
        target_idx = 0
        for (j, candidate) in enumerate(psd.devices)
            if candidate.dtype isa target_class &&
               candidate.dtype.bus == bus_match &&
               _normalize_id(candidate.dtype.id) == id_match
                target_idx = j
                break
            end
        end

        if target_idx == 0
            @warn "No $target_class target found for $(typeof(device.dtype)) device $i (bus=$bus_match, id=$id_match)"
            continue
        end

        target = psd.devices[target_idx]

        # Propagate parent-generator index so the controller's init
        # routine can read pg/qg from the same static generator.
        dmap.gen[i] = dmap.gen[target_idx]

        # Wire produced signals: target reads controller's output.
        for sig in produces_signals(typeof(device.dtype))
            target_slot = target.ctrl_ptr + sig.target_ctrl_offset
            if sig.source_kind === :alg_first
                # source: controller's first alg state in the global z vector.
                source_z = psd.diff_dim + device.alg_ptr
            elseif sig.source_kind === :diff_first
                source_z = device.diff_ptr
            else
                error("Unknown source_kind $(sig.source_kind) in produces_signals($(typeof(device.dtype)))")
            end
            psd.uvec_idx[target_slot] = source_z
        end

        # Wire consumed signals: controller reads target's state.
        for sig in consumes_signals(typeof(device.dtype))
            ctrl_slot = device.ctrl_ptr + sig.ctrl_offset
            if sig.state_kind === :w
                # Generator's w state at target.diff_ptr + w_offset(target_class).
                source_z = target.diff_ptr + w_offset(target_class)
            else
                error("Unknown state_kind $(sig.state_kind) in consumes_signals($(typeof(device.dtype)))")
            end
            psd.uvec_idx[ctrl_slot] = source_z
        end
    end
    return nothing
end
