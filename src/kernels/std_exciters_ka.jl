# KernelAbstractions wrappers for the AbstractStdExciter models (IEEET1, ...).
# Mirrors esdc1a_/esst4b_residual_ka! / _jacobian_ka! in ka_wrappers.jl: each
# @kernel calls the model's branch-free `_one!` leaf. Included AFTER
# ka_wrappers.jl (needs `using KernelAbstractions`). `_rhs_fun_ka_cpu!` calls
# `std_exc_residual_ka!` once for all models.

@kernel function ieeet1_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _ieeet1_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
    end
end

@kernel function ieeet1_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _ieeet1_jacobian_one!(nz, z, p, par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

@kernel function expic1_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _expic1_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
    end
end

@kernel function expic1_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _expic1_jacobian_one!(nz, z, p, par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

@kernel function exac2_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _exac2_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
    end
end

@kernel function exac2_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _exac2_jacobian_one!(nz, z, p, par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

@kernel function scrx_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _scrx_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
    end
end

@kernel function scrx_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _scrx_jacobian_one!(nz, z, p, par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

@kernel function esac6a_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _esac6a_residual_one!(f, z, p, diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
    end
end

@kernel function esac6a_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _esac6a_jacobian_one!(nz, z, p, par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

function std_exc_residual_ka!(backend, f, z, p, L)
    t = L.ieeet1
    if t.n > 0
        kernel = ieeet1_residual_ka!(backend)
        kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
    end
    t = L.expic1
    if t.n > 0
        kernel = expic1_residual_ka!(backend)
        kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
    end
    t = L.exac2
    if t.n > 0
        kernel = exac2_residual_ka!(backend)
        kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
    end
    # EXAC1 / ESAC1A run the EXAC2 leaf (shared pvec layout)
    for t in (L.exac1, L.esac1a)
        if t.n > 0
            kernel = exac2_residual_ka!(backend)
            kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
        end
    end
    t = L.scrx
    if t.n > 0
        kernel = scrx_residual_ka!(backend)
        kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
    end
    t = L.esac6a
    if t.n > 0
        kernel = esac6a_residual_ka!(backend)
        kernel(f, z, p, t.online, t.diff_ptr, t.par_ptr, t.vr_idx, t.vs_idx; ndrange=t.n)
    end
    return nothing
end
