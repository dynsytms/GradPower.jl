# StaticGenerator batched residual & Jacobian.
#
# Three bus-type variants share the current-injection equations into the
# voltage rows; PV adds one alg state (q), SLACK adds two (p, q) and pins
# (vr, vi) to (vset*cos(aset), vset*sin(aset)) via additional algebraic
# residuals on the alg rows.

const SG_PQ    = 1
const SG_PV    = 2
const SG_SLACK = 3

# --------------------------------------------------------------------
# Sparsity preallocation
# --------------------------------------------------------------------

function static_gen_preallocate!(coord_list::Vector{Vector{Int}},
                                  table::StaticGenTable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        vr = Int(table.vr_idx[k])
        vi = vr + 1
        ap = Int(table.alg_ptr[k])
        bt = Int(table.bus_type[k])

        # Current-injection block (all bus types): vr/vi rows × vr/vi cols.
        # Already exists in the network adjacency block — adjacency walk
        # makes (bus, bus) entries; these are duplicates that the
        # `sparse(row,col,data)` combiner dedups.
        push!(coord_list[vr], vr)
        push!(coord_list[vr], vi)
        push!(coord_list[vi], vr)
        push!(coord_list[vi], vi)

        if bt == SG_PV
            # voltage-regulation row: vr^2 + vi^2 - vset^2
            push!(coord_list[ap], vr)
            push!(coord_list[ap], vi)
            # power-injection columns: q enters vr/vi rows via the alg state
            push!(coord_list[vr], ap)
            push!(coord_list[vi], ap)
        elseif bt == SG_SLACK
            # angle pins: vr = vset*cos(aset), vi = vset*sin(aset)
            push!(coord_list[ap],     vr)
            push!(coord_list[ap + 1], vi)
            # both p and q enter vr/vi rows
            push!(coord_list[vr], ap)
            push!(coord_list[vr], ap + 1)
            push!(coord_list[vi], ap)
            push!(coord_list[vi], ap + 1)
        end
    end
    return nothing
end

# --------------------------------------------------------------------
# Jacobian position cache
# --------------------------------------------------------------------

function static_gen_jac_positions!(table::StaticGenTable, J::SparseMatrixCSC)
    n = table.n
    n == 0 && return nothing
    rows = rowvals(J)
    @inbounds for k in 1:n
        vr = Int(table.vr_idx[k])
        vi = vr + 1
        ap = Int(table.alg_ptr[k])
        bt = Int(table.bus_type[k])

        table.jac_pos[k, J_SG_VR_VR] = _find_pos(J, rows, vr, vr)
        table.jac_pos[k, J_SG_VR_VI] = _find_pos(J, rows, vr, vi)
        table.jac_pos[k, J_SG_VI_VR] = _find_pos(J, rows, vi, vr)
        table.jac_pos[k, J_SG_VI_VI] = _find_pos(J, rows, vi, vi)

        if bt == SG_PV
            table.jac_pos[k, J_SG_VR_AP] = _find_pos(J, rows, vr, ap)
            table.jac_pos[k, J_SG_VI_AP] = _find_pos(J, rows, vi, ap)
            table.jac_pos[k, J_SG_AP_VR] = _find_pos(J, rows, ap, vr)
            table.jac_pos[k, J_SG_AP_VI] = _find_pos(J, rows, ap, vi)
        elseif bt == SG_SLACK
            table.jac_pos[k, J_SG_VR_AP]  = _find_pos(J, rows, vr, ap)
            table.jac_pos[k, J_SG_VI_AP]  = _find_pos(J, rows, vi, ap)
            table.jac_pos[k, J_SG_VR_AP1] = _find_pos(J, rows, vr, ap + 1)
            table.jac_pos[k, J_SG_VI_AP1] = _find_pos(J, rows, vi, ap + 1)
            table.jac_pos[k, J_SG_AP_VR]  = _find_pos(J, rows, ap, vr)
            table.jac_pos[k, J_SG_AP1_VI] = _find_pos(J, rows, ap + 1, vi)
        end
    end
    return nothing
end

# --------------------------------------------------------------------
# Residual batch
# --------------------------------------------------------------------

@inline function _sg_power(z::AbstractArray, p::AbstractArray, bt::Int,
                            ap::Int, pp::Int)
    if bt == SG_PV
        return p[pp], z[ap]          # p is param; q is alg state
    elseif bt == SG_SLACK
        return z[ap], z[ap + 1]      # both alg
    else
        return p[pp], p[pp + 1]      # both param
    end
end

@inline function _static_gen_residual_one!(f, z, p,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr,
        k::Int, inj=nothing, inj_slot::Int=0)
    @inbounds begin
    vr_idx = Int(vr_idx_arr[k])
    vi_idx = vr_idx + 1
    ap = Int(alg_ptr[k])
    pp = Int(par_ptr[k])
    bt = Int(bus_type_arr[k])

    vr = z[vr_idx]
    vi = z[vi_idx]
    vm2_raw = vr*vr + vi*vi
    vm2_tld = 0.2
    vm2 = vm2_raw > vm2_tld ? vm2_raw : vm2_tld

    pp_val, qq_val = _sg_power(z, p, bt, ap, pp)

    # Current injection into the network voltage rows.
    if inj === nothing
        # Plain-loop path: accumulate directly into f.
        f[vr_idx] += (pp_val*vr + qq_val*vi) / vm2
        f[vi_idx] += (pp_val*vi - qq_val*vr) / vm2
    else
        # KA path: write to per-device injection buffer.
        inj[2*inj_slot - 1] = (pp_val*vr + qq_val*vi) / vm2
        inj[2*inj_slot]     = (pp_val*vi - qq_val*vr) / vm2
    end

    # Alg-row residuals.
    if bt == SG_PV
        vset = p[pp + 2]
        f[ap] = vr*vr + vi*vi - vset*vset
    elseif bt == SG_SLACK
        vset = p[pp + 2]
        aset = p[pp + 3]
        f[ap]     = vr - vset*cos(aset)
        f[ap + 1] = vi - vset*sin(aset)
    end
    end
    return nothing
end

@inline function static_gen_residual_batch!(f::AbstractArray, z::AbstractArray,
                                             p::AbstractArray,
                                             table::StaticGenTable)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        table.online[k] || continue
        _static_gen_residual_one!(f, z, p,
            table.vr_idx, table.alg_ptr, table.par_ptr, table.bus_type,
            k)
    end
    return nothing
end

# --------------------------------------------------------------------
# Jacobian batch
# --------------------------------------------------------------------

@inline function _static_gen_jacobian_one!(nz, z, p,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr, jac_pos,
        k::Int)
    @inbounds begin
    vr_idx = Int(vr_idx_arr[k])
    vi_idx = vr_idx + 1
    ap = Int(alg_ptr[k])
    pp = Int(par_ptr[k])
    bt = Int(bus_type_arr[k])

    vr = z[vr_idx]
    vi = z[vi_idx]
    vm2_raw = vr*vr + vi*vi
    vm2_tld = 0.2
    vm2 = vm2_raw > vm2_tld ? vm2_raw : vm2_tld
    pp_val, qq_val = _sg_power(z, p, bt, ap, pp)
    ir_num = pp_val*vr + qq_val*vi
    ii_num = pp_val*vi - qq_val*vr

    if vm2_raw > vm2_tld
        vm4 = vm2*vm2
        dir_dvr = pp_val/vm2 - 2.0*vr*ir_num/vm4
        dir_dvi = qq_val/vm2 - 2.0*vi*ir_num/vm4
        dii_dvr = -qq_val/vm2 - 2.0*vr*ii_num/vm4
        dii_dvi = pp_val/vm2 - 2.0*vi*ii_num/vm4
    else
        dir_dvr =  pp_val/vm2
        dir_dvi =  qq_val/vm2
        dii_dvr = -qq_val/vm2
        dii_dvi =  pp_val/vm2
    end

    nz[jac_pos[k, J_SG_VR_VR]] += dir_dvr
    nz[jac_pos[k, J_SG_VR_VI]] += dir_dvi
    nz[jac_pos[k, J_SG_VI_VR]] += dii_dvr
    nz[jac_pos[k, J_SG_VI_VI]] += dii_dvi

    if bt == SG_PV
        nz[jac_pos[k, J_SG_AP_VR]] = 2.0*vr
        nz[jac_pos[k, J_SG_AP_VI]] = 2.0*vi
        nz[jac_pos[k, J_SG_VR_AP]] += vi / vm2
        nz[jac_pos[k, J_SG_VI_AP]] += -vr / vm2
    elseif bt == SG_SLACK
        nz[jac_pos[k, J_SG_AP_VR]]  = 1.0
        nz[jac_pos[k, J_SG_AP1_VI]] = 1.0
        nz[jac_pos[k, J_SG_VR_AP]]  += vr / vm2
        nz[jac_pos[k, J_SG_VI_AP]]  += vi / vm2
        nz[jac_pos[k, J_SG_VR_AP1]] += vi / vm2
        nz[jac_pos[k, J_SG_VI_AP1]] += -vr / vm2
    end
    end
    return nothing
end

@inline function static_gen_jacobian_batch!(J::SparseMatrixCSC,
                                             z::AbstractArray,
                                             p::AbstractArray,
                                             table::StaticGenTable)
    n = table.n
    n == 0 && return nothing
    nz = nonzeros(J)
    @inbounds for k in 1:n
        table.online[k] || continue
        _static_gen_jacobian_one!(nz, z, p,
            table.vr_idx, table.alg_ptr, table.par_ptr, table.bus_type, table.jac_pos,
            k)
    end
    return nothing
end
