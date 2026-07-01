# KernelAbstractions wrappers for device residual and Jacobian kernels.
#
# Each KA @kernel calls the existing `_*_one!` leaf for its device type
# with an injection buffer argument. Network current injections go through
# a per-device buffer `inj[2*k-1:2*k]`, then a reduction kernel sums
# them per bus into f.
#
# `dispatch_residual_kernels!` and `dispatch_jacobian_kernels!` select
# between the plain-loop path (untouched) and the KA path.

using KernelAbstractions

# Sentinel type for KA CPU backend selection (wraps KernelAbstractions.CPU()).
struct KA_CPU end

# -----------------------------------------------------------------------
# KA @kernel wrappers — each calls the existing _*_one! leaf.
# -----------------------------------------------------------------------

@kernel function genrou_residual_ka!(f, z, u, p, inj, online,
        diff_ptr, alg_ptr, ctrl_ptr, par_ptr, bus_arr,
        @Const(diff_dim), @Const(net_ptr), @Const(twopi60))
    k = @index(Global)
    if @inbounds online[k]
        _genrou_residual_one!(f, z, u, p,
            diff_ptr, alg_ptr, ctrl_ptr, par_ptr, bus_arr,
            k, diff_dim, net_ptr, twopi60, inj, k)
    end
end

@kernel function zipload_residual_ka!(f, z, p, inj, online,
        bus_arr, par_ptr,
        @Const(net_ptr), @Const(inj_offset))
    k = @index(Global)
    if @inbounds online[k]
        _zipload_residual_one!(f, z, p,
            bus_arr, par_ptr,
            k, net_ptr, inj, inj_offset + k)
    end
end

@kernel function static_gen_residual_ka!(f, z, p, inj, online,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr,
        @Const(inj_offset))
    k = @index(Global)
    if @inbounds online[k]
        _static_gen_residual_one!(f, z, p,
            vr_idx_arr, alg_ptr, par_ptr, bus_type_arr,
            k, inj, inj_offset + k)
    end
end

# -----------------------------------------------------------------------
# Controller KA @kernel wrappers (phase 14c D1)
# -----------------------------------------------------------------------

@kernel function ieesgo_residual_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim))
    k = @index(Global)
    if @inbounds online[k]
        _ieesgo_residual_one!(f, z, p,
            diff_ptr, alg_ptr, par_ptr, w_idx_arr,
            k, diff_dim)
    end
end

@kernel function tgov1_residual_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim))
    k = @index(Global)
    if @inbounds online[k]
        _tgov1_residual_one!(f, z, p,
            diff_ptr, alg_ptr, par_ptr, w_idx_arr,
            k, diff_dim)
    end
end

@kernel function sexs_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _sexs_residual_one!(f, z, p,
            diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr,
            k)
    end
end

@kernel function esdc1a_residual_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr)
    k = @index(Global)
    if @inbounds online[k]
        _esdc1a_residual_one!(f, z, p,
            diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr,
            k)
    end
end

@kernel function ieeest_residual_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim))
    k = @index(Global)
    if @inbounds online[k]
        _ieeest_residual_one!(f, z, p,
            diff_ptr, alg_ptr, par_ptr, w_idx_arr,
            diff_dim, k)
    end
end

# -----------------------------------------------------------------------
# Jacobian KA @kernel wrappers (phase 14c D2)
# -----------------------------------------------------------------------

@kernel function genrou_jacobian_ka!(nz, z, p, online,
        diff_ptr, alg_ptr, par_ptr, bus_arr, jac_pos, has_gov, has_exc,
        @Const(diff_dim), @Const(net_ptr), @Const(twopi60))
    k = @index(Global)
    if @inbounds online[k]
        _genrou_jacobian_one!(nz, z, p,
            diff_ptr, alg_ptr, par_ptr, bus_arr, jac_pos, has_gov, has_exc,
            k, diff_dim, net_ptr, twopi60)
    end
end

@kernel function ieesgo_jacobian_ka!(nz, p, online,
        par_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _ieesgo_jacobian_one!(nz, p,
            par_ptr, jac_pos, k)
    end
end

@kernel function tgov1_jacobian_ka!(nz, p, online,
        par_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _tgov1_jacobian_one!(nz, p,
            par_ptr, jac_pos, k)
    end
end

@kernel function sexs_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _sexs_jacobian_one!(nz, z, p,
            par_ptr, vr_idx_arr, vs_idx_arr, jac_pos, k)
    end
end

@kernel function esdc1a_jacobian_ka!(nz, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _esdc1a_jacobian_one!(nz, z, p,
            par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
    end
end

@kernel function ieeest_jacobian_ka!(nz, z, p, online,
        par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos,
        @Const(diff_dim))
    k = @index(Global)
    if @inbounds online[k]
        _ieeest_jacobian_one!(nz, z, p,
            par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos,
            diff_dim, k)
    end
end

@kernel function zipload_jacobian_ka!(nz, z, p, online,
        bus_arr, par_ptr, jac_pos,
        @Const(net_ptr))
    k = @index(Global)
    if @inbounds online[k]
        _zipload_jacobian_one!(nz, z, p,
            bus_arr, par_ptr, jac_pos,
            k, net_ptr)
    end
end

@kernel function static_gen_jacobian_ka!(nz, z, p, online,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr, jac_pos)
    k = @index(Global)
    if @inbounds online[k]
        _static_gen_jacobian_one!(nz, z, p,
            vr_idx_arr, alg_ptr, par_ptr, bus_type_arr, jac_pos, k)
    end
end

@kernel function bus_injection_reduce_ka!(f, inj, bus_map, @Const(net_ptr))
    ik = @index(Global)
    @inbounds begin
        bus = Int(bus_map[ik])
        vr_idx = net_ptr + 2*(bus - 1) + 1
        f[vr_idx] += inj[2*ik - 1]
        f[vr_idx + 1] += inj[2*ik]
    end
end

# -----------------------------------------------------------------------
# Bus injection reduction (plain-loop fallback)
# -----------------------------------------------------------------------

function bus_injection_reduce!(f::AbstractArray, inj::AbstractArray,
                                bus_map::AbstractVector{Int32},
                                n_inj::Int, net_ptr::Int)
    @inbounds for ik in 1:n_inj
        bus = Int(bus_map[ik])
        vr_idx = net_ptr + 2*(bus - 1) + 1
        vi_idx = vr_idx + 1
        f[vr_idx] += inj[2*ik - 1]
        f[vi_idx] += inj[2*ik]
    end
    return nothing
end

# -----------------------------------------------------------------------
# Injection metadata: which devices inject and their bus mapping
# -----------------------------------------------------------------------

struct InjectionMeta
    n_genrou::Int
    n_zipload::Int
    n_static_gen::Int
    n_total::Int
    # bus_map[ik] = internal bus index for injector ik
    # ordering: genrou 1..n_genrou, zipload n_genrou+1..n_genrou+n_zipload,
    # static_gen n_genrou+n_zipload+1..n_total
    bus_map::Vector{Int32}
end

function InjectionMeta(L::SimulationLayout)
    ng = L.genrou.n
    nz = L.zipload.n
    ns = L.static_gen.n
    nt = ng + nz + ns
    bus_map = Vector{Int32}(undef, nt)
    @inbounds for k in 1:ng
        bus_map[k] = L.genrou.bus[k]
    end
    @inbounds for k in 1:nz
        bus_map[ng + k] = L.zipload.bus[k]
    end
    @inbounds for k in 1:ns
        # static_gen bus is stored as internal index (after fix_static_gen_bus_idx!)
        bus_map[ng + nz + k] = L.static_gen.bus[k]
    end
    return InjectionMeta(ng, nz, ns, nt, bus_map)
end

# -----------------------------------------------------------------------
# KA residual dispatch (KA_CPU path)
# -----------------------------------------------------------------------

function _rhs_fun_ka_cpu!(f::AbstractArray, z::AbstractArray, u::AbstractArray,
                           p::AbstractArray, inj::AbstractArray,
                           dyn::PowerSystemDynamics,
                           ybus::SparseMatrixCSC, L::SimulationLayout,
                           inj_meta::InjectionMeta)
    backend = KernelAbstractions.CPU()
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    net_ptr  = diff_dim + alg_dim
    v  = @view z[net_ptr+1:end]
    fv = @view f[net_ptr+1:end]
    mul!(fv, ybus, v, -1.0, 0.0)

    _apply_uvec_routing!(u, z, dyn.uvec_idx)

    # Zero injection buffer
    fill!(inj, 0.0)

    twopi60 = 2.0 * π * 60.0

    # Genrou: inj slots 1..n_genrou
    gt = L.genrou
    if gt.n > 0
        kernel = genrou_residual_ka!(backend)
        kernel(f, z, u, p, inj, gt.online,
               gt.diff_ptr, gt.alg_ptr, gt.ctrl_ptr, gt.par_ptr, gt.bus,
               diff_dim, net_ptr, twopi60; ndrange=gt.n)
    end

    # Controllers (KA kernels — no network injection)
    ig = L.ieesgo
    if ig.n > 0
        kernel = ieesgo_residual_ka!(backend)
        kernel(f, z, p, ig.online,
               ig.diff_ptr, ig.alg_ptr, ig.par_ptr, ig.w_idx,
               diff_dim; ndrange=ig.n)
    end

    tg = L.tgov1
    if tg.n > 0
        kernel = tgov1_residual_ka!(backend)
        kernel(f, z, p, tg.online,
               tg.diff_ptr, tg.alg_ptr, tg.par_ptr, tg.w_idx,
               diff_dim; ndrange=tg.n)
    end

    pss = L.ieeest
    if pss.n > 0
        kernel = ieeest_residual_ka!(backend)
        kernel(f, z, p, pss.online,
               pss.diff_ptr, pss.alg_ptr, pss.par_ptr, pss.w_idx,
               diff_dim; ndrange=pss.n)
    end

    sx = L.sexs
    if sx.n > 0
        kernel = sexs_residual_ka!(backend)
        kernel(f, z, p, sx.online,
               sx.diff_ptr, sx.par_ptr, sx.vr_idx, sx.vs_idx; ndrange=sx.n)
    end

    ex = L.esdc1a
    if ex.n > 0
        kernel = esdc1a_residual_ka!(backend)
        kernel(f, z, p, ex.online,
               ex.diff_ptr, ex.par_ptr, ex.vr_idx, ex.vs_idx; ndrange=ex.n)
    end

    # ZIPLoad: inj slots n_genrou+1..n_genrou+n_zipload
    zt = L.zipload
    if zt.n > 0
        kernel = zipload_residual_ka!(backend)
        kernel(f, z, p, inj, zt.online,
               zt.bus, zt.par_ptr,
               net_ptr, inj_meta.n_genrou; ndrange=zt.n)
    end

    # StaticGen: inj slots n_genrou+n_zipload+1..n_total
    st = L.static_gen
    if st.n > 0
        kernel = static_gen_residual_ka!(backend)
        kernel(f, z, p, inj, st.online,
               st.vr_idx, st.alg_ptr, st.par_ptr, st.bus_type,
               inj_meta.n_genrou + inj_meta.n_zipload; ndrange=st.n)
    end

    # Reduce injection buffer into f
    if inj_meta.n_total > 0
        kernel = bus_injection_reduce_ka!(backend)
        kernel(f, inj, inj_meta.bus_map, net_ptr; ndrange=inj_meta.n_total)
    end

    _apply_events_fun!(f, v, dyn.events, net_ptr)
    return nothing
end

# -----------------------------------------------------------------------
# KA jacobian dispatch (KA_CPU path)
# -----------------------------------------------------------------------
# The Jacobian network-row entries for genrou are per-device unique (row,col)
# pairs (each generator has its own delta/iq/id columns). No bus-level
# collision. For zipload and static_gen, entries target (vr,vr)/(vr,vi)/
# (vi,vr)/(vi,vi) of the same bus — collision when devices share a bus.
# But on CPU, the sequential loop handles this correctly via +=.
# GPU atomics or buffer+reduce for Jacobian are deferred to phase 14b.

function _rhs_jac_ka_cpu!(jac::SparseMatrixCSC, z::AbstractArray, u::AbstractArray,
                           p::AbstractArray, dyn::PowerSystemDynamics,
                           ybus::SparseMatrixCSC, L::SimulationLayout)
    # Reuse the existing batched Jacobian path — it's already correct on CPU.
    _rhs_jac_batched!(jac, z, u, p, dyn, ybus, L)
    return nothing
end

# -----------------------------------------------------------------------
# Top-level dispatch
# -----------------------------------------------------------------------

"""
    dispatch_residual_kernels!(f, z, u, p, inj, L, dyn, ybus, inj_meta, backend)

Backend-selected residual evaluation.
- `Nothing`: existing plain-loop path (untouched).
- `KA_CPU()`: KA kernel path with injection buffer + reduction on CPU.
"""
function dispatch_residual_kernels!(f, z, u, p, inj, L, dyn, ybus, inj_meta, ::KA_CPU;
                                    log=nothing)
    _rhs_fun_ka_cpu!(f, z, u, p, inj, dyn, ybus, L, inj_meta)
    return nothing
end

function dispatch_residual_kernels!(f, z, u, p, inj, L, dyn, ybus, inj_meta, ::Nothing;
                                    log=nothing)
    # Plain-loop path — delegates to existing _rhs_fun_batched!
    _rhs_fun_batched!(f, z, u, p, dyn, ybus, L, log)
    return nothing
end

"""
    dispatch_jacobian_kernels!(J, z, u, p, L, dyn, ybus, backend)

Backend-selected Jacobian evaluation.
- `Nothing` (CPU): existing plain-loop path.
- `KA_CPU()`: KA CPU path (currently same as plain-loop for Jacobian).
"""
function dispatch_jacobian_kernels!(J, z, u, p, L, dyn, ybus, ::KA_CPU)
    _rhs_jac_ka_cpu!(J, z, u, p, dyn, ybus, L)
    return nothing
end

function dispatch_jacobian_kernels!(J, z, u, p, L, dyn, ybus, ::Nothing)
    _rhs_jac_batched!(J, z, u, p, dyn, ybus, L)
    return nothing
end
