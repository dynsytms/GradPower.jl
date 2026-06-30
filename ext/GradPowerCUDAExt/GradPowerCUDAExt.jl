module GradPowerCUDAExt

using CUDA
using CUDA.CUBLAS
using CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays
using KernelAbstractions

using GradPower
using KLU

# -----------------------------------------------------------------------
# CuDSSPreconditioner (D2)
# -----------------------------------------------------------------------

"""
    CuDSSPreconditioner

GPU-resident sparse LU preconditioner wrapping a CudssSolver from CUDSS.jl.
Satisfies `LinearAlgebra.ldiv!(y, P, x)` for Krylov.jl compatibility.

API surface used: CudssSolver, cudss("analysis"/"factorization"/"solve").
Fallback: if CUDSS proves unusable, replace with ILU(0) from
KrylovPreconditioners.jl.
"""
struct CuDSSPreconditioner
    solver::CudssSolver{Float64, Int32}
    Y_csr::CuSparseMatrixCSR{Float64, Int32}
end

function CuDSSPreconditioner(Y_csc::SparseMatrixCSC{Float64, Int})
    Y_csr = CuSparseMatrixCSR(Y_csc)
    solver = CudssSolver(Y_csr, "G", 'F')
    x_d = CUDA.zeros(Float64, size(Y_csc, 1))
    b_d = CUDA.zeros(Float64, size(Y_csc, 1))
    cudss("analysis", solver, x_d, b_d)
    cudss("factorization", solver, x_d, b_d)
    return CuDSSPreconditioner(solver, Y_csr)
end

function LinearAlgebra.ldiv!(y::CuVector{Float64}, P::CuDSSPreconditioner, x::CuVector{Float64})
    copyto!(y, x)
    cudss("solve", P.solver, y, x)
    return y
end

"""
    refresh_cudss_preconditioner!(P::CuDSSPreconditioner, Y_csc::SparseMatrixCSC)

Re-do numeric factorization after topology change (ybus_real modified).
Symbolic factorization survives because TripLineEvent only modifies
existing nonzero entries.
"""
function refresh_cudss_preconditioner!(P::CuDSSPreconditioner, Y_csc::SparseMatrixCSC{Float64, Int})
    Y_csr_new = CuSparseMatrixCSR(Y_csc)
    cudss_set(P.solver.matrix, Y_csr_new)
    x_d = CUDA.zeros(Float64, size(Y_csc, 1))
    b_d = CUDA.zeros(Float64, size(Y_csc, 1))
    cudss("factorization", P.solver, x_d, b_d)
    return nothing
end

# -----------------------------------------------------------------------
# GPU copies of SoA table arrays needed for KA kernel dispatch
# -----------------------------------------------------------------------

struct GpuGenrouArrays
    n::Int
    diff_ptr::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    ctrl_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    bus::CuVector{Int32}
    online::CuVector{Bool}
end

struct GpuZIPLoadArrays
    n::Int
    bus::CuVector{Int32}
    par_ptr::CuVector{Int32}
    online::CuVector{Bool}
end

struct GpuStaticGenArrays
    n::Int
    vr_idx::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    bus_type::CuVector{Int8}
    online::CuVector{Bool}
end

function GpuGenrouArrays(gt::GradPower.GenrouTable)
    gt.n == 0 && return GpuGenrouArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Bool}(undef,0))
    GpuGenrouArrays(gt.n, CuVector(gt.diff_ptr), CuVector(gt.alg_ptr),
                     CuVector(gt.ctrl_ptr), CuVector(gt.par_ptr),
                     CuVector(gt.bus), CuVector(gt.online))
end

function GpuZIPLoadArrays(zt::GradPower.ZIPLoadTable)
    zt.n == 0 && return GpuZIPLoadArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0))
    GpuZIPLoadArrays(zt.n, CuVector(zt.bus), CuVector(zt.par_ptr),
                      CuVector(zt.online))
end

function GpuStaticGenArrays(st::GradPower.StaticGenTable)
    st.n == 0 && return GpuStaticGenArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int8}(undef,0), CuVector{Bool}(undef,0))
    GpuStaticGenArrays(st.n, CuVector(st.vr_idx), CuVector(st.alg_ptr),
                        CuVector(st.par_ptr), CuVector(st.bus_type),
                        CuVector(st.online))
end

# -----------------------------------------------------------------------
# GPU BatchedLayout (D1)
# -----------------------------------------------------------------------

"""
    GpuBatchedLayout

GPU-resident version of BatchedLayout. All 2D arrays are CuMatrix,
sparsity pattern arrays are CuVector, SoA table pointers are CuVector.
"""
struct GpuBatchedLayout
    M::Int
    sys_dim::Int
    diff_dim::Int
    alg_dim::Int
    nbus::Int
    z::CuMatrix{Float64}
    p::CuMatrix{Float64}
    u::CuMatrix{Float64}
    f::CuMatrix{Float64}
    zold::CuMatrix{Float64}
    inj::CuMatrix{Float64}
    J_nzval::CuMatrix{Float64}
    # Shared sparsity (on GPU for kernel access)
    J_colptr::CuVector{Int}
    J_rowval::CuVector{Int}
    # Admittance matrix (sparse on GPU)
    ybus_csr::CuSparseMatrixCSR{Float64, Int32}
    # CPU-side ybus for host-side operations
    ybus_cpu::SparseMatrixCSC{Float64, Int}
    # Injection metadata
    inj_meta_bus_map::CuVector{Int32}
    inj_meta_n_genrou::Int
    inj_meta_n_zipload::Int
    inj_meta_n_static_gen::Int
    inj_meta_n_total::Int
    # uvec_idx on GPU
    uvec_idx::CuVector{Int64}
    # Diff indices on GPU (for backward Euler)
    diff_indices::Union{Nothing, CuVector{Int}}
    is_diff::Union{Nothing, CuVector{Bool}}
    # CPU-side bus_map for sequential bus injection reduce (avoids GPU race)
    inj_meta_bus_map_cpu::Vector{Int32}
    # GPU copies of SoA device table arrays for KA kernel dispatch
    gpu_genrou::GpuGenrouArrays
    gpu_zipload::GpuZIPLoadArrays
    gpu_static_gen::GpuStaticGenArrays
end

"""
    GpuBatchedLayout(dp, ps, M)

Construct a GPU-resident batched layout. Initialization runs on CPU,
then arrays are transferred to GPU.
"""
function GpuBatchedLayout(dp::GradPower.DynamicProblem, ps::GradPower.PowerSystem, M::Int)
    # Build CPU layout first (reuses existing code)
    bl_cpu = GradPower.BatchedLayout(dp, ps, M)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout

    # Transfer to GPU
    z    = CuMatrix(bl_cpu.z)
    p    = CuMatrix(bl_cpu.p)
    u    = CuMatrix(bl_cpu.u)
    f    = CUDA.zeros(Float64, M, bl_cpu.sys_dim)
    zold = CuMatrix(bl_cpu.zold)
    inj  = CUDA.zeros(Float64, M, 2 * bl_cpu.inj_meta.n_total)
    J_nzval = CuMatrix(bl_cpu.J_nzval)
    J_colptr = CuVector(bl_cpu.J_colptr)
    J_rowval = CuVector(bl_cpu.J_rowval)

    # Admittance matrix on GPU (CSR for cuSPARSE SpMV)
    ybus_csr = CuSparseMatrixCSR(bl_cpu.ybus)
    ybus_cpu = bl_cpu.ybus

    # Injection metadata
    bus_map_gpu = CuVector(bl_cpu.inj_meta.bus_map)
    uvec_idx_gpu = CuVector(bl_cpu.uvec_idx)

    # Diff indices
    di = dyn.diff_indices
    di_gpu = di !== nothing ? CuVector(di) : nothing
    isd = dyn.is_diff
    isd_gpu = isd !== nothing ? CuVector(Bool.(isd)) : nothing

    # GPU copies of SoA table arrays for KA kernels
    gpu_genrou = GpuGenrouArrays(L.genrou)
    gpu_zipload = GpuZIPLoadArrays(L.zipload)
    gpu_static_gen = GpuStaticGenArrays(L.static_gen)

    return GpuBatchedLayout(M, bl_cpu.sys_dim, bl_cpu.diff_dim, bl_cpu.alg_dim,
                             bl_cpu.nbus,
                             z, p, u, f, zold, inj, J_nzval,
                             J_colptr, J_rowval,
                             ybus_csr, ybus_cpu,
                             bus_map_gpu,
                             bl_cpu.inj_meta.n_genrou,
                             bl_cpu.inj_meta.n_zipload,
                             bl_cpu.inj_meta.n_static_gen,
                             bl_cpu.inj_meta.n_total,
                             uvec_idx_gpu,
                             di_gpu, isd_gpu,
                             copy(bl_cpu.inj_meta.bus_map),
                             gpu_genrou, gpu_zipload, gpu_static_gen)
end

# -----------------------------------------------------------------------
# GPU batched helper kernels (D1)
# -----------------------------------------------------------------------

@kernel function uvec_routing_gpu_kernel!(u, z, uvec_idx)
    j = @index(Global)
    @inbounds begin
        src = uvec_idx[j]
        if src != 0
            M = size(u, 1)
            for m in 1:M
                u[m, j] = z[m, src]
            end
        end
    end
end

@kernel function events_fun_gpu_kernel!(f, z, bus_arr, rfault_arr, status_arr, net_ptr)
    ei = @index(Global)
    @inbounds begin
        if status_arr[ei]
            bus = bus_arr[ei]
            yfault = 1.0 / rfault_arr[ei]
            vr_col = net_ptr + 2*(bus-1) + 1
            vi_col = vr_col + 1
            M = size(f, 1)
            for m in 1:M
                f[m, vr_col] -= yfault * z[m, vr_col]
                f[m, vi_col] -= yfault * z[m, vi_col]
            end
        end
    end
end

@kernel function beuler_diff_gpu_kernel!(f, z, zold, diff_indices, dt)
    idx = @index(Global)
    @inbounds begin
        i = diff_indices[idx]
        M = size(f, 1)
        for m in 1:M
            f[m, i] = z[m, i] - zold[m, i] - dt * f[m, i]
        end
    end
end

# 1D uvec routing kernel for per-scenario GPU dispatch
@kernel function uvec_routing_1d_kernel!(u, z, uvec_idx)
    j = @index(Global)
    @inbounds begin
        src = uvec_idx[j]
        if src != 0
            u[j] = z[src]
        end
    end
end

# -----------------------------------------------------------------------
# GPU residual evaluation using KA kernels (CRITICAL 1 fix)
#
# For each scenario m, extracts 1D CuVector views from the 2D CuMatrix
# and launches the existing KA @kernel functions with CUDABackend().
# Controllers (IEESGO, TGOV1, SEXS, ESDC1A, IEEEST) don't have KA
# kernels — their contributions are computed via CPU roundtrip on the
# controller-owned rows of f.
# -----------------------------------------------------------------------

# Evaluate residual for all M scenarios using KA kernels on GPU.
# Network injection devices (genrou, zipload, static_gen) run on GPU.
# Controllers run on CPU (no KA kernels; their computation is lightweight).
function _residual_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                      L::GradPower.SimulationLayout)
    M = gbl.M
    backend = CUDABackend()
    diff_dim = gbl.diff_dim
    alg_dim  = gbl.alg_dim
    net_ptr  = diff_dim + alg_dim
    twopi60  = 2.0 * π * 60.0
    nv = 2 * gbl.nbus
    sys_dim = gbl.sys_dim

    gt = gbl.gpu_genrou
    zt = gbl.gpu_zipload
    st = gbl.gpu_static_gen
    n_genrou = gbl.inj_meta_n_genrou
    n_zipload = gbl.inj_meta_n_zipload
    n_total = gbl.inj_meta_n_total

    # Preallocate contiguous CuVector buffers for cuSPARSE SpMV
    # (cuSPARSE mv! requires DenseCuVector, not SubArray views)
    v_buf = CUDA.zeros(Float64, nv)
    fv_buf = CUDA.zeros(Float64, nv)

    # Preallocate contiguous 1D CuVector buffers for KA kernels
    # (KA kernels need contiguous arrays, not row-views of 2D CuMatrix)
    f_1d  = CUDA.zeros(Float64, sys_dim)
    z_1d  = CUDA.zeros(Float64, sys_dim)
    u_1d  = CUDA.zeros(Float64, size(gbl.u, 2))
    p_1d  = CUDA.zeros(Float64, size(gbl.p, 2))
    inj_1d = CUDA.zeros(Float64, 2 * n_total)

    for m in 1:M
        # Copy scenario m data into contiguous 1D buffers
        copyto!(z_1d, view(gbl.z, m, :))
        copyto!(u_1d, view(gbl.u, m, :))
        copyto!(p_1d, view(gbl.p, m, :))
        fill!(f_1d, 0.0)

        # 1. Y*v → network portion of f (cuSPARSE SpMV)
        copyto!(v_buf, view(z_1d, net_ptr+1:net_ptr+nv))
        CUSPARSE.mv!('N', -1.0, gbl.ybus_csr, v_buf, 0.0, fv_buf, 'O')
        copyto!(view(f_1d, net_ptr+1:net_ptr+nv), fv_buf)

        # 2. uvec routing on GPU
        if length(gbl.uvec_idx) > 0
            kernel = uvec_routing_1d_kernel!(backend)
            kernel(u_1d, z_1d, gbl.uvec_idx; ndrange=length(gbl.uvec_idx))
        end

        # 3. Zero injection buffer
        fill!(inj_1d, 0.0)

        # 4. Genrou KA kernel
        if gt.n > 0
            kernel = GradPower.genrou_residual_ka!(backend)
            kernel(f_1d, z_1d, u_1d, p_1d, inj_1d, gt.online,
                   gt.diff_ptr, gt.alg_ptr, gt.ctrl_ptr, gt.par_ptr, gt.bus,
                   diff_dim, net_ptr, twopi60; ndrange=gt.n)
        end

        # 5. ZIPLoad KA kernel
        if zt.n > 0
            kernel = GradPower.zipload_residual_ka!(backend)
            kernel(f_1d, z_1d, p_1d, inj_1d, zt.online,
                   zt.bus, zt.par_ptr,
                   net_ptr, n_genrou; ndrange=zt.n)
        end

        # 6. StaticGen KA kernel
        if st.n > 0
            kernel = GradPower.static_gen_residual_ka!(backend)
            kernel(f_1d, z_1d, p_1d, inj_1d, st.online,
                   st.vr_idx, st.alg_ptr, st.par_ptr, st.bus_type,
                   n_genrou + n_zipload; ndrange=st.n)
        end

        KernelAbstractions.synchronize(backend)

        # 7. Bus injection reduce — sequential on CPU to avoid race
        # conditions. Multiple devices (genrou, zipload) on the same bus
        # write f[vr_idx] +=, which races on GPU. GPU atomics deferred.
        if n_total > 0
            f_host = Array(f_1d)
            inj_host = Array(inj_1d)
            GradPower.bus_injection_reduce!(f_host, inj_host,
                gbl.inj_meta_bus_map_cpu, n_total, net_ptr)
            copyto!(f_1d, CuVector(f_host))
        end

        # Copy f_1d back to the 2D f matrix
        copyto!(view(gbl.f, m, :), f_1d)
    end

    # 8. Controllers: compute on CPU. Download z, u, p for all scenarios,
    # compute controller contributions to f, upload back.
    # Controllers write to their own exclusive diff/alg rows (no network rows).
    z_cpu = Array(gbl.z)
    p_cpu = Array(gbl.p)
    f_cpu = Array(gbl.f)

    for m in 1:M
        f_row = view(f_cpu, m, :)
        z_row = view(z_cpu, m, :)
        p_row = view(p_cpu, m, :)

        GradPower.ieesgo_residual_batch!(f_row, z_row, p_row, L.ieesgo, diff_dim)
        GradPower.tgov1_residual_batch!(f_row, z_row, p_row, L.tgov1, diff_dim)
        GradPower.ieeest_residual_batch!(f_row, z_row, p_row, L.ieeest, diff_dim)
        GradPower.sexs_residual_batch!(f_row, z_row, p_row, L.sexs)
        GradPower.esdc1a_residual_batch!(f_row, z_row, p_row, L.esdc1a)
    end

    # Upload f with controller contributions
    copyto!(gbl.f, CuMatrix(f_cpu))

    # 9. Events on GPU
    events = dyn.events
    if !isempty(events)
        bus_arr    = CuVector(Int64[ev.bus for ev in events])
        rfault_arr = CuVector(Float64[ev.rfault for ev in events])
        status_arr = CuVector(Bool[ev.status for ev in events])
        kernel = events_fun_gpu_kernel!(backend)
        kernel(gbl.f, gbl.z, bus_arr, rfault_arr, status_arr, net_ptr; ndrange=length(events))
        KernelAbstractions.synchronize(backend)
    end

    return nothing
end

# -----------------------------------------------------------------------
# GPU Jacobian evaluation (CPU roundtrip — Jacobian is used for KLU
# factorization which is CPU-only. Phase 14b requirement is that the
# RESIDUAL runs on GPU; Jacobian on CPU is acceptable.)
# -----------------------------------------------------------------------

function _jacobian_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                      L::GradPower.SimulationLayout)
    M = gbl.M
    z_cpu = Array(gbl.z)
    u_cpu = Array(gbl.u)
    p_cpu = Array(gbl.p)
    nnz_J = size(gbl.J_nzval, 2)
    J_nzval_cpu = zeros(Float64, M, nnz_J)
    J_colptr_cpu = Array(gbl.J_colptr)
    J_rowval_cpu = Array(gbl.J_rowval)

    J_buf = SparseMatrixCSC(gbl.sys_dim, gbl.sys_dim,
                             J_colptr_cpu, J_rowval_cpu,
                             zeros(Float64, nnz_J))
    for m in 1:M
        fill!(J_buf.nzval, 0.0)
        z_row = view(z_cpu, m, :)
        u_row = view(u_cpu, m, :)
        p_row = view(p_cpu, m, :)
        GradPower._rhs_jac_batched!(J_buf, z_row, u_row, p_row, dyn, gbl.ybus_cpu, L)
        J_nzval_cpu[m, :] .= J_buf.nzval
    end
    copyto!(gbl.J_nzval, CuMatrix(J_nzval_cpu))
    return nothing
end

# -----------------------------------------------------------------------
# GPU backward Euler
# -----------------------------------------------------------------------

function _beuler_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                     L::GradPower.SimulationLayout, dt::Float64)
    _residual_all_scenarios_gpu!(gbl, dyn, L)

    if gbl.diff_indices !== nothing
        backend = CUDABackend()
        kernel = beuler_diff_gpu_kernel!(backend)
        kernel(gbl.f, gbl.z, gbl.zold, gbl.diff_indices, dt; ndrange=length(gbl.diff_indices))
        KernelAbstractions.synchronize(backend)
    else
        f_cpu = Array(gbl.f)
        z_cpu = Array(gbl.z)
        zold_cpu = Array(gbl.zold)
        GradPower.beuler_batched_2d!(f_cpu, z_cpu, zold_cpu,
                                      dyn.is_diff, dyn.diff_indices,
                                      gbl.diff_dim, dt, gbl.M)
        copyto!(gbl.f, CuMatrix(f_cpu))
    end
    return nothing
end

function _beuler_jac_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                         L::GradPower.SimulationLayout, dt::Float64)
    _jacobian_all_scenarios_gpu!(gbl, dyn, L)

    # Apply backward Euler scaling on CPU (Jacobian nzval is small enough)
    J_nzval_cpu = Array(gbl.J_nzval)
    J_colptr_cpu = Array(gbl.J_colptr)
    J_rowval_cpu = Array(gbl.J_rowval)
    is_diff = dyn.is_diff
    n_da = is_diff !== nothing ? length(is_diff) : gbl.diff_dim

    @inbounds for m in 1:gbl.M
        nzv = view(J_nzval_cpu, m, :)
        for col in 1:gbl.sys_dim
            for nz_idx in J_colptr_cpu[col]:(J_colptr_cpu[col+1]-1)
                row = J_rowval_cpu[nz_idx]
                if is_diff !== nothing
                    if row <= n_da && is_diff[row]
                        nzv[nz_idx] *= -dt
                        if row == col
                            nzv[nz_idx] += 1.0
                        end
                    end
                else
                    if row <= gbl.diff_dim
                        nzv[nz_idx] *= -dt
                        if row == col
                            nzv[nz_idx] += 1.0
                        end
                    end
                end
            end
        end
    end
    copyto!(gbl.J_nzval, CuMatrix(J_nzval_cpu))
    return nothing
end

# -----------------------------------------------------------------------
# GPU Newton step (per-scenario KLU on CPU, state on GPU)
# -----------------------------------------------------------------------

function _newton_step_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                            L::GradPower.SimulationLayout, dt::Float64;
                            itermax::Int=30, tol::Float64=1e-10)
    M = gbl.M
    sys_dim = gbl.sys_dim
    nnz_J = size(gbl.J_nzval, 2)
    J_colptr_cpu = Array(gbl.J_colptr)
    J_rowval_cpu = Array(gbl.J_rowval)

    J_bufs = [SparseMatrixCSC(sys_dim, sys_dim, copy(J_colptr_cpu), copy(J_rowval_cpu),
                               zeros(Float64, nnz_J)) for _ in 1:M]
    facts = Vector{Any}(undef, M)
    dx_buf = zeros(Float64, sys_dim)
    f_buf  = zeros(Float64, sys_dim)

    for iter in 1:itermax
        _beuler_all_scenarios_gpu!(gbl, dyn, L, dt)

        f_cpu = Array(gbl.f)
        max_norm = 0.0
        @inbounds for m in 1:M
            for i in 1:sys_dim
                a = abs(f_cpu[m, i])
                if a > max_norm; max_norm = a; end
            end
        end
        max_norm < tol && return true

        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)

        J_nzval_cpu = Array(gbl.J_nzval)
        z_cpu = Array(gbl.z)

        for m in 1:M
            copyto!(J_bufs[m].nzval, view(J_nzval_cpu, m, :))
            copyto!(f_buf, view(f_cpu, m, :))
            if iter == 1
                facts[m] = klu(J_bufs[m])
            else
                klu!(facts[m], J_bufs[m])
            end
            ldiv!(dx_buf, facts[m], f_buf)
            @inbounds for k in 1:sys_dim
                z_cpu[m, k] -= dx_buf[k]
            end
        end
        copyto!(gbl.z, CuMatrix(z_cpu))
    end
    return false
end

# -----------------------------------------------------------------------
# integrate_gpu! (D1 + D4 combined)
# -----------------------------------------------------------------------

"""
    integrate_gpu!(gbl, ps, tf; dt=1/120, newton_tol=1e-10)

GPU-resident batched integration. Returns (tvec, trajs) where
trajs[m] is the trajectory matrix for scenario m.
"""
function integrate_gpu!(gbl::GpuBatchedLayout, ps::GradPower.PowerSystem, tf::Float64;
                         dt::Float64=1.0/120.0, newton_tol::Float64=1e-10)
    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout
    M = gbl.M
    sys_dim = gbl.sys_dim

    nsteps = Int(round(tf / dt))
    tvec = collect(0:dt:tf)

    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule, by = x -> x[1])

    # Store initial z on CPU
    z0_cpu = Array(gbl.z)
    trajs = [zeros(Float64, sys_dim, nsteps + 1) for _ in 1:M]
    for m in 1:M
        trajs[m][:, 1] .= z0_cpu[m, :]
    end

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_gpu!(gbl, dyn, L, dt; tol=newton_tol)

        z_cpu = Array(gbl.z)
        for m in 1:M
            trajs[m][:, k + 1] .= z_cpu[m, :]
        end

        any_event = false
        while sched_idx <= length(event_schedule) && event_schedule[sched_idx][1] == k
            _, idx, action = event_schedule[sched_idx]
            if action === :on
                GradPower.activate!(events[idx])
            elseif action === :off
                GradPower.deactivate!(events[idx])
            end
            any_event = true
            sched_idx += 1
        end

        if any_event
            copyto!(gbl.zold, gbl.z)
            _newton_step_gpu!(gbl, dyn, L, 0.0; tol=newton_tol)
        end
    end

    for event in events
        GradPower.deactivate!(event)
    end

    return tvec, trajs
end

# -----------------------------------------------------------------------
# Batched dense LU (D3) — cuBLAS getrf_strided_batched! / getrs_strided_batched!
# -----------------------------------------------------------------------

"""
    GpuBatchedLU

Holds packed dense A_k blocks for one w_k-group across all clusters
and scenarios, plus pivot storage. Uses cuBLAS strided batched LU.

Layout: A_packed[w_k, w_k, n_clusters_in_group * M]
        B_packed[w_k, 2, n_clusters_in_group * M]
        ipiv[w_k, n_clusters_in_group * M]
"""
struct GpuBatchedLU
    w_k::Int
    n_clusters::Int
    M::Int
    A_packed::CuArray{Float64, 3}
    B_packed::CuArray{Float64, 3}    # w_k x 2 x (n_clusters * M)
    C_packed::CuArray{Float64, 3}    # 2 x w_k x (n_clusters * M)
    ipiv::CuMatrix{Int32}            # w_k x (n_clusters * M)
    info::CuVector{Int32}            # n_clusters * M
end

function GpuBatchedLU(w_k::Int, n_clusters::Int, M::Int)
    batch = n_clusters * M
    A = CUDA.zeros(Float64, w_k, w_k, batch)
    B = CUDA.zeros(Float64, w_k, 2, batch)
    C = CUDA.zeros(Float64, 2, w_k, batch)
    ipiv = CUDA.zeros(Int32, w_k, batch)
    info = CUDA.zeros(Int32, batch)
    return GpuBatchedLU(w_k, n_clusters, M, A, B, C, ipiv, info)
end

"""
    gpu_batched_lu_factor!(glu::GpuBatchedLU)

Factor all A_k blocks using cuBLAS getrf_strided_batched!.
"""
function gpu_batched_lu_factor!(glu::GpuBatchedLU)
    CUDA.CUBLAS.getrf_strided_batched!(glu.A_packed, glu.ipiv, glu.info)
    return nothing
end

"""
    gpu_batched_lu_solve!(glu::GpuBatchedLU)

Solve A_k * x = B_packed using cuBLAS getrs_strided_batched!.
Solution overwrites B_packed.
"""
function gpu_batched_lu_solve!(glu::GpuBatchedLU)
    CUDA.CUBLAS.getrs_strided_batched!('N', glu.A_packed, glu.B_packed, glu.ipiv)
    return nothing
end

# -----------------------------------------------------------------------
# GPU Schur complement operator (D4)
#
# TODO: Matrix-free GPU Schur operator (S·x = Y·x − Σ_k C_k A_k⁻¹ B_k·x)
# is deferred. The current implementation uses the assembled S matrix on
# CPU for correctness. The GpuBatchedLU factor/solve is tested independently
# in G3. Full matrix-free GPU path is planned for phase 14c.
# -----------------------------------------------------------------------

"""
    GpuSchurOperator

Operator computing S·x via the assembled S matrix on CPU.
Implements mul!, size, eltype for Krylov.jl compatibility.

Known limitation: this uses the CPU-assembled S matrix, not the
matrix-free GPU formula S·x = Y·x − Σ_k C_k A_k⁻¹ B_k·x.
The GpuBatchedLU factor/solve path is validated independently (G3).
The full matrix-free GPU Schur operator is planned for phase 14c.
"""
struct GpuSchurOperator
    n_red::Int
    # CPU-side Schur assembly (reuses existing SchurWorkspace)
    sw::GradPower.SchurWorkspace
end

Base.size(op::GpuSchurOperator) = (op.n_red, op.n_red)
Base.eltype(::GpuSchurOperator) = Float64

function LinearAlgebra.mul!(y::AbstractVector, op::GpuSchurOperator, x::AbstractVector)
    # S * x via the assembled S matrix (CPU path — known limitation, see docstring)
    mul!(y, op.sw.S, x)
    return y
end

end # module GradPowerCUDAExt
