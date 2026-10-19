# -----------------------------------------------------------------------
# Trajectory recording (selective output) for the CPU integrators.
#
# `integrate!` and `integrate_batched!` store the state at the output steps
# `1, 1 + save_every, 1 + 2 save_every, ...` of the time grid (1 = the initial
# state), restricted to the rows `save_idxs`, converted to `save_eltype`. The
# defaults store every row of every step in Float64, i.e. the full trajectory.
# The last step is NOT added when it is not on the stride: the output grid is
# exactly `1:save_every:nsteps+1`, as in `dynamics_channels(...; downsample)`.
#
# `step_callback(k, t, z)` is called with every recorded-or-not accepted state
# (k = 1 for the initial state, then k = 2, ..., nsteps+1), which lets callers
# accumulate full-resolution statistics without storing the full trajectory.
# At an event step, the state passed (and stored) is the one before the
# event's algebraic re-solve, as in the stored trajectory.
# -----------------------------------------------------------------------

_check_save_every(s::Integer) =
    s >= 1 || throw(ArgumentError("save_every must be >= 1, got $s"))

function _resolve_save_idxs(save_idxs, n::Int)
    save_idxs === Colon() && return Colon()
    idx = collect(Int, save_idxs)
    all(i -> 1 <= i <= n, idx) ||
        throw(ArgumentError("save_idxs must be within 1:$n"))
    return idx
end

_nsaved(nsteps::Int, save_every::Int) = length(1:save_every:nsteps+1)

struct _Recorder{T,I}
    idxs::I                 # Colon() or Vector{Int}
    every::Int
    traj::Matrix{T}         # (nrows, nsaved)
end

function _Recorder(n::Int, nsteps::Int, save_idxs, save_every::Integer, ::Type{T}) where {T}
    _check_save_every(save_every)
    idxs = _resolve_save_idxs(save_idxs, n)
    nrows = idxs === Colon() ? n : length(idxs)
    return _Recorder{T,typeof(idxs)}(idxs, Int(save_every),
                                     zeros(T, nrows, _nsaved(nsteps, Int(save_every))))
end

_saved_times(tvec::AbstractVector, every::Int) = tvec[1:every:end]

# Store state `z` of grid step `k` (1-based; 1 = initial state) if `k` is an
# output step.
@inline function _record!(r::_Recorder, k::Int, z::AbstractVector)
    (k - 1) % r.every == 0 || return nothing
    j = (k - 1) ÷ r.every + 1
    _store_column!(r.traj, j, z, r.idxs)
    return nothing
end

@inline function _store_column!(traj::AbstractMatrix, j::Int, z::AbstractVector, ::Colon)
    @inbounds for i in eachindex(z)
        traj[i, j] = z[i]
    end
end
@inline function _store_column!(traj::AbstractMatrix, j::Int, z::AbstractVector, idxs::Vector{Int})
    @inbounds for (r, i) in enumerate(idxs)
        traj[r, j] = z[i]
    end
end

# Output up to grid step `kstop` (early termination): the stored columns of
# steps 1:every:kstop.
function _truncated_output(r::_Recorder, tvec::AbstractVector, kstop::Int)
    j = (kstop - 1) ÷ r.every + 1
    return tvec[1:r.every:kstop], r.traj[:, 1:j]
end
