# Generic batch drivers for the `AbstractStdExciter` models
# (src/exciters_psse.jl, tables in src/tables/std_exciters.jl), plus the
# umbrella entry points that src/dynamics.jl calls once for all of them.
#
# Each model supplies:
#   _<m>_jac_coords(dp, vr_idx, vs_idx) -> NTuple of (row, col), in slot order
#                                          (col == 0 marks an unused vs slot)
#   _<m>_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx, vs_idx, k)
#   _<m>_jacobian_one!(nz, z, p, par_ptr, vr_idx, vs_idx, diff_ptr, jac_pos, k)
# (the same leaf signatures as ESDC1A/ESST4B, so the KA and CUDA wrappers are
# mechanical). Driving preallocation AND jac_pos from one coordinate list keeps
# the sparsity pattern and the position cache consistent by construction.

function _std_exc_preallocate!(coords, coord_list::Vector{Vector{Int}}, t::StdExcTable)
    for k in 1:t.n
        for (row, col) in coords(Int(t.diff_ptr[k]), Int(t.vr_idx[k]), Int(t.vs_idx[k]))
            col > 0 && push!(coord_list[row], col)
        end
    end
    return nothing
end

function _std_exc_jac_positions!(coords, t::StdExcTable, J::SparseMatrixCSC)
    t.n == 0 && return nothing
    rows = rowvals(J)
    for k in 1:t.n
        s = 0
        for (row, col) in coords(Int(t.diff_ptr[k]), Int(t.vr_idx[k]), Int(t.vs_idx[k]))
            s += 1
            t.jac_pos[k, s] = col > 0 ? _find_pos(J, rows, row, col) : Int32(0)
        end
        @assert s == size(t.jac_pos, 2)
    end
    return nothing
end

@inline function _std_exc_residual_batch!(leaf::F, f, z, p, t::StdExcTable) where {F}
    t.n == 0 && return nothing
    @inbounds for k in 1:t.n
        t.online[k] || continue
        leaf(f, z, p, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx, k)
    end
    return nothing
end

@inline function _std_exc_jacobian_batch!(leaf::F, J::SparseMatrixCSC, z, p,
                                          t::StdExcTable) where {F}
    t.n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:t.n
        t.online[k] || continue
        leaf(nz, z, p, t.par_ptr, t.vr_idx, t.vs_idx, t.diff_ptr, t.jac_pos, k)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Umbrella entry points (called from src/dynamics.jl). One explicit line per
# model so every call site is concrete. Add new models HERE.
# ---------------------------------------------------------------------------

@inline function std_exc_residual_batch!(f, z, p, L)
    ieeet1_residual_batch!(f, z, p, L.ieeet1)
    expic1_residual_batch!(f, z, p, L.expic1)
    exac2_residual_batch!(f, z, p, L.exac2)
    exac1_residual_batch!(f, z, p, L.exac1)
    esac1a_residual_batch!(f, z, p, L.esac1a)
    scrx_residual_batch!(f, z, p, L.scrx)
    esac6a_residual_batch!(f, z, p, L.esac6a)
    return nothing
end

@inline function std_exc_jacobian_batch!(J, z, p, L)
    ieeet1_jacobian_batch!(J, z, p, L.ieeet1)
    expic1_jacobian_batch!(J, z, p, L.expic1)
    exac2_jacobian_batch!(J, z, p, L.exac2)
    exac1_jacobian_batch!(J, z, p, L.exac1)
    esac1a_jacobian_batch!(J, z, p, L.esac1a)
    scrx_jacobian_batch!(J, z, p, L.scrx)
    esac6a_jacobian_batch!(J, z, p, L.esac6a)
    return nothing
end

function std_exc_preallocate!(coord_list, L)
    ieeet1_preallocate!(coord_list, L.ieeet1)
    expic1_preallocate!(coord_list, L.expic1)
    exac2_preallocate!(coord_list, L.exac2)
    exac1_preallocate!(coord_list, L.exac1)
    esac1a_preallocate!(coord_list, L.esac1a)
    scrx_preallocate!(coord_list, L.scrx)
    esac6a_preallocate!(coord_list, L.esac6a)
    return nothing
end

function std_exc_jac_positions!(L, J::SparseMatrixCSC)
    ieeet1_jac_positions!(L.ieeet1, J)
    expic1_jac_positions!(L.expic1, J)
    exac2_jac_positions!(L.exac2, J)
    exac1_jac_positions!(L.exac1, J)
    esac1a_jac_positions!(L.esac1a, J)
    scrx_jac_positions!(L.scrx, J)
    esac6a_jac_positions!(L.esac6a, J)
    return nothing
end
