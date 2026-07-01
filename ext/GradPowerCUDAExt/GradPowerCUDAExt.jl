module GradPowerCUDAExt

using CUDA
using CUDA.CUBLAS
using CUDA.CUSPARSE
using CUDSS
using LinearAlgebra
using SparseArrays
using KernelAbstractions
using Krylov

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
    jac_pos::CuMatrix{Int32}
    has_gov::CuVector{Bool}
    has_exc::CuVector{Bool}
end

struct GpuZIPLoadArrays
    n::Int
    bus::CuVector{Int32}
    par_ptr::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

struct GpuStaticGenArrays
    n::Int
    vr_idx::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    bus_type::CuVector{Int8}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuGenrouArrays(gt::GradPower.GenrouTable)
    gt.n == 0 && return GpuGenrouArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Bool}(undef,0), CuMatrix{Int32}(undef,0,0),
        CuVector{Bool}(undef,0), CuVector{Bool}(undef,0))
    GpuGenrouArrays(gt.n, CuVector(gt.diff_ptr), CuVector(gt.alg_ptr),
                     CuVector(gt.ctrl_ptr), CuVector(gt.par_ptr),
                     CuVector(gt.bus), CuVector(gt.online),
                     CuMatrix(gt.jac_pos), CuVector(gt.has_gov),
                     CuVector(gt.has_exc))
end

function GpuZIPLoadArrays(zt::GradPower.ZIPLoadTable)
    zt.n == 0 && return GpuZIPLoadArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuZIPLoadArrays(zt.n, CuVector(zt.bus), CuVector(zt.par_ptr),
                      CuVector(zt.online), CuMatrix(zt.jac_pos))
end

function GpuStaticGenArrays(st::GradPower.StaticGenTable)
    st.n == 0 && return GpuStaticGenArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int8}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuStaticGenArrays(st.n, CuVector(st.vr_idx), CuVector(st.alg_ptr),
                        CuVector(st.par_ptr), CuVector(st.bus_type),
                        CuVector(st.online), CuMatrix(st.jac_pos))
end

# -----------------------------------------------------------------------
# GPU copies of SoA controller table arrays (phase 14c D1)
# -----------------------------------------------------------------------

struct GpuIEESGOArrays
    n::Int
    diff_ptr::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    w_idx::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuIEESGOArrays(t::GradPower.IEESGOTable)
    t.n == 0 && return GpuIEESGOArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuIEESGOArrays(t.n, CuVector(t.diff_ptr), CuVector(t.alg_ptr),
                     CuVector(t.par_ptr), CuVector(t.w_idx), CuVector(t.online),
                     CuMatrix(t.jac_pos))
end

struct GpuTGOV1Arrays
    n::Int
    diff_ptr::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    w_idx::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuTGOV1Arrays(t::GradPower.TGOV1Table)
    t.n == 0 && return GpuTGOV1Arrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuTGOV1Arrays(t.n, CuVector(t.diff_ptr), CuVector(t.alg_ptr),
                    CuVector(t.par_ptr), CuVector(t.w_idx), CuVector(t.online),
                    CuMatrix(t.jac_pos))
end

struct GpuSEXSArrays
    n::Int
    diff_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    vr_idx::CuVector{Int32}
    vs_idx::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuSEXSArrays(t::GradPower.SEXSTable)
    t.n == 0 && return GpuSEXSArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuSEXSArrays(t.n, CuVector(t.diff_ptr), CuVector(t.par_ptr),
                   CuVector(t.vr_idx), CuVector(t.vs_idx), CuVector(t.online),
                   CuMatrix(t.jac_pos))
end

struct GpuESDC1AArrays
    n::Int
    diff_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    vr_idx::CuVector{Int32}
    vs_idx::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuESDC1AArrays(t::GradPower.ESDC1ATable)
    t.n == 0 && return GpuESDC1AArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuESDC1AArrays(t.n, CuVector(t.diff_ptr), CuVector(t.par_ptr),
                     CuVector(t.vr_idx), CuVector(t.vs_idx), CuVector(t.online),
                     CuMatrix(t.jac_pos))
end

struct GpuIEEESTArrays
    n::Int
    diff_ptr::CuVector{Int32}
    alg_ptr::CuVector{Int32}
    par_ptr::CuVector{Int32}
    w_idx::CuVector{Int32}
    online::CuVector{Bool}
    jac_pos::CuMatrix{Int32}
end

function GpuIEEESTArrays(t::GradPower.IEEESTTable)
    t.n == 0 && return GpuIEEESTArrays(0, CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Int32}(undef,0),
        CuVector{Int32}(undef,0), CuVector{Bool}(undef,0),
        CuMatrix{Int32}(undef,0,0))
    GpuIEEESTArrays(t.n, CuVector(t.diff_ptr), CuVector(t.alg_ptr),
                     CuVector(t.par_ptr), CuVector(t.w_idx), CuVector(t.online),
                     CuMatrix(t.jac_pos))
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
    # Per-bus injection index (CSR-style on GPU) for deterministic reduce
    bus_inj_start::CuVector{Int32}
    bus_inj_list::CuVector{Int32}
    # GPU copies of SoA device table arrays for KA kernel dispatch
    gpu_genrou::GpuGenrouArrays
    gpu_zipload::GpuZIPLoadArrays
    gpu_static_gen::GpuStaticGenArrays
    # Controller GPU arrays (phase 14c D1)
    gpu_ieesgo::GpuIEESGOArrays
    gpu_tgov1::GpuTGOV1Arrays
    gpu_sexs::GpuSEXSArrays
    gpu_esdc1a::GpuESDC1AArrays
    gpu_ieeest::GpuIEEESTArrays
    # GPU event arrays (phase 14c — avoid per-step allocation)
    event_bus::CuVector{Int64}
    event_rfault::CuVector{Float64}
    event_status::CuVector{Bool}
    # Ybus→J_nzval index map for GPU Jacobian (phase 14c D2)
    ybus_to_jnz::CuVector{Int32}
    ybus_nzval_cpu::Vector{Float64}
    # Event Jacobian: nzval positions for bus diagonal (vr,vr) and (vi,vi)
    event_jac_diag_pos::CuVector{Int32}   # length 2*n_events: [vr_pos_1, vi_pos_1, ...]
    # Ybus CSC arrays on GPU for batched SpMV kernel (phase 14c D6)
    ybus_colptr_gpu::CuVector{Int}
    ybus_rowval_gpu::CuVector{Int}
    ybus_nzval_gpu::CuVector{Float64}
    ybus_ncols::Int
    # Schur workspace data on GPU (phase 14c D3)
    # Precomputed nzval→S index maps
    reduced_idx_gpu::Union{Nothing, CuVector{Int}}
    g2r_gpu::Union{Nothing, CuVector{Int}}
    J_to_S_row::Union{Nothing, CuVector{Int32}}
    J_to_S_col::Union{Nothing, CuVector{Int32}}
    J_to_S_nzpos::Union{Nothing, CuVector{Int32}}
    J_nz_count_for_S::Int
    # GPU Schur complement S storage
    S_nzval_gpu::Union{Nothing, CuVector{Float64}}
    S_colptr_gpu::Union{Nothing, CuVector{Int32}}
    S_rowval_gpu::Union{Nothing, CuVector{Int32}}
    S_n::Int  # n_red
    S_nnz::Int
    # Per-cluster GPU index maps for A_k/B_k/C_k extraction
    cluster_A_jnz::Union{Nothing, CuVector{Int32}}  # flat: all A_k nzval indices
    cluster_A_local::Union{Nothing, CuVector{Int32}} # flat: local positions in A_packed
    cluster_B_jnz::Union{Nothing, CuVector{Int32}}
    cluster_B_local::Union{Nothing, CuVector{Int32}}
    cluster_C_jnz::Union{Nothing, CuVector{Int32}}
    cluster_C_local::Union{Nothing, CuVector{Int32}}
    cluster_D_S_nzpos::Union{Nothing, CuVector{Int32}} # where D_k subtracts in S
    n_A_entries::Int
    n_B_entries::Int
    n_C_entries::Int
    n_D_entries::Int
    # cuDSS monolithic direct solver (phase 14c)
    cudss_solver::Union{Nothing, CudssSolver{Float64, Int32}}
    cudss_csr::Union{Nothing, CuSparseMatrixCSR{Float64, Int32}}  # persistent CSR shell
    csc_to_csr_perm::Union{Nothing, CuVector{Int32}}              # nzval reorder map
    cudss_rhs::Union{Nothing, CuVector{Float64}}                  # reusable RHS buffer
    cudss_sol::Union{Nothing, CuVector{Float64}}                  # reusable solution buffer
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

    # Controller GPU arrays (phase 14c D1)
    gpu_ieesgo = GpuIEESGOArrays(L.ieesgo)
    gpu_tgov1  = GpuTGOV1Arrays(L.tgov1)
    gpu_sexs   = GpuSEXSArrays(L.sexs)
    gpu_esdc1a = GpuESDC1AArrays(L.esdc1a)
    gpu_ieeest = GpuIEEESTArrays(L.ieeest)

    # Per-bus injection index for deterministic GPU reduce
    bis_cpu, bil_cpu = build_bus_injection_index(bl_cpu.inj_meta.bus_map, bl_cpu.nbus)
    bus_inj_start_gpu = CuVector(bis_cpu)
    bus_inj_list_gpu  = CuVector(bil_cpu)

    # Pre-allocated event arrays on GPU (avoid per-step allocation)
    events = dyn.events
    event_bus = CuVector(Int64[ev.bus for ev in events])
    event_rfault = CuVector(Float64[ev.rfault for ev in events])
    event_status = CuVector(Bool[ev.status for ev in events])

    # Ybus→J_nzval index map (phase 14c D2)
    ybus_cpu_mat = bl_cpu.ybus
    net_ptr_cpu = bl_cpu.diff_dim + dyn.alg_dim
    J_colptr_cpu = bl_cpu.J_colptr
    J_rowval_cpu = bl_cpu.J_rowval
    ybus_rows = rowvals(ybus_cpu_mat)
    ybus_nnz = length(nonzeros(ybus_cpu_mat))
    ybus_to_jnz_cpu = zeros(Int32, ybus_nnz)
    for col_y in 1:size(ybus_cpu_mat, 2)
        for nz_y in nzrange(ybus_cpu_mat, col_y)
            row_y = ybus_rows[nz_y]
            new_row = row_y + net_ptr_cpu
            new_col = col_y + net_ptr_cpu
            for j in J_colptr_cpu[new_col]:(J_colptr_cpu[new_col+1]-1)
                if J_rowval_cpu[j] == new_row
                    ybus_to_jnz_cpu[nz_y] = Int32(j)
                    break
                end
            end
        end
    end
    ybus_to_jnz_gpu = CuVector(ybus_to_jnz_cpu)
    ybus_nzval_cpu_arr = copy(nonzeros(ybus_cpu_mat))

    # Event Jacobian diagonal positions
    n_events = length(events)
    event_jac_diag_pos_cpu = zeros(Int32, 2 * n_events)
    for (ei, ev) in enumerate(events)
        bus = ev.bus
        vr_col = net_ptr_cpu + 2*(bus-1) + 1
        vi_col = vr_col + 1
        # Find diagonal nzval position for (vr_col, vr_col)
        for j in J_colptr_cpu[vr_col]:(J_colptr_cpu[vr_col+1]-1)
            if J_rowval_cpu[j] == vr_col
                event_jac_diag_pos_cpu[2*ei-1] = Int32(j)
                break
            end
        end
        # Find diagonal nzval position for (vi_col, vi_col)
        for j in J_colptr_cpu[vi_col]:(J_colptr_cpu[vi_col+1]-1)
            if J_rowval_cpu[j] == vi_col
                event_jac_diag_pos_cpu[2*ei] = Int32(j)
                break
            end
        end
    end
    event_jac_diag_pos_gpu = CuVector(event_jac_diag_pos_cpu)

    # Ybus CSC arrays on GPU for batched SpMV kernel (phase 14c D6)
    ybus_colptr_g = CuVector(ybus_cpu_mat.colptr)
    ybus_rowval_g = CuVector(ybus_cpu_mat.rowval)
    ybus_nzval_g  = CuVector(nonzeros(ybus_cpu_mat))
    ybus_ncols_v  = size(ybus_cpu_mat, 2)

    # Schur index maps — built lazily via setup_gpu_schur!
    reduced_idx_g = nothing
    g2r_g = nothing
    J_to_S_row_g = nothing
    J_to_S_col_g = nothing
    J_to_S_nzpos_g = nothing
    J_nz_count_for_S_v = 0
    S_nzval_g = nothing
    S_colptr_g = nothing
    S_rowval_g = nothing
    S_n_v = 0
    S_nnz_v = 0
    cluster_A_jnz_g = nothing
    cluster_A_local_g = nothing
    cluster_B_jnz_g = nothing
    cluster_B_local_g = nothing
    cluster_C_jnz_g = nothing
    cluster_C_local_g = nothing
    cluster_D_S_nzpos_g = nothing
    n_A_entries_v = 0
    n_B_entries_v = 0
    n_C_entries_v = 0
    n_D_entries_v = 0

    # --- cuDSS monolithic direct solver setup ---
    # Build CSC→CSR nzval permutation (computed once, fixed sparsity)
    nnz_J = length(bl_cpu.J_rowval)
    J_csc_idx = SparseMatrixCSC(bl_cpu.sys_dim, bl_cpu.sys_dim,
                                 copy(bl_cpu.J_colptr), copy(bl_cpu.J_rowval),
                                 collect(Float64, 1:nnz_J))
    J_csr_idx = copy(J_csc_idx')  # CSC of transpose = CSR layout
    csc_to_csr_perm = CuVector(Int32.(J_csr_idx.nzval))

    # Build persistent CuSparseMatrixCSR with dummy values for symbolic analysis
    J_csc_ones = SparseMatrixCSC(bl_cpu.sys_dim, bl_cpu.sys_dim,
                                  copy(bl_cpu.J_colptr), copy(bl_cpu.J_rowval),
                                  ones(Float64, nnz_J))
    cudss_csr = CuSparseMatrixCSR(J_csc_ones)

    # Create solver and perform symbolic analysis (once)
    cudss_solver = CudssSolver(cudss_csr, "G", 'F')
    _tmp_x = CUDA.zeros(Float64, bl_cpu.sys_dim)
    _tmp_b = CUDA.zeros(Float64, bl_cpu.sys_dim)
    cudss("analysis", cudss_solver, _tmp_x, _tmp_b)

    cudss_rhs = CUDA.zeros(Float64, bl_cpu.sys_dim)
    cudss_sol = CUDA.zeros(Float64, bl_cpu.sys_dim)

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
                             bus_inj_start_gpu, bus_inj_list_gpu,
                             gpu_genrou, gpu_zipload, gpu_static_gen,
                             gpu_ieesgo, gpu_tgov1, gpu_sexs, gpu_esdc1a, gpu_ieeest,
                             event_bus, event_rfault, event_status,
                             ybus_to_jnz_gpu, ybus_nzval_cpu_arr,
                             event_jac_diag_pos_gpu,
                             ybus_colptr_g, ybus_rowval_g, ybus_nzval_g, ybus_ncols_v,
                             reduced_idx_g, g2r_g,
                             J_to_S_row_g, J_to_S_col_g, J_to_S_nzpos_g, J_nz_count_for_S_v,
                             S_nzval_g, S_colptr_g, S_rowval_g, S_n_v, S_nnz_v,
                             cluster_A_jnz_g, cluster_A_local_g,
                             cluster_B_jnz_g, cluster_B_local_g,
                             cluster_C_jnz_g, cluster_C_local_g,
                             cluster_D_S_nzpos_g,
                             n_A_entries_v, n_B_entries_v, n_C_entries_v, n_D_entries_v,
                             cudss_solver, cudss_csr, csc_to_csr_perm,
                             cudss_rhs, cudss_sol)
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

# Event Jacobian kernel: adds -1/rfault to diagonal entries for active faults
@kernel function events_jac_gpu_kernel!(nzval, rfault_arr, status_arr, diag_pos, @Const(M))
    ei = @index(Global)
    @inbounds begin
        if status_arr[ei]
            yfault = 1.0 / rfault_arr[ei]
            vr_pos = diag_pos[2*ei - 1]
            vi_pos = diag_pos[2*ei]
            for m in 1:M
                nzval[m, vr_pos] += -yfault
                nzval[m, vi_pos] += -yfault
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

# -----------------------------------------------------------------------
# Per-bus injection reduce: deterministic accumulation, no atomics
# -----------------------------------------------------------------------

"""
    build_bus_injection_index(bus_map, nbus) -> (bus_inj_start, bus_inj_list)

CSR-style index mapping each bus to its injector slots.
`bus_inj_start` has length `nbus+1`; injectors for bus `b` are
`bus_inj_list[bus_inj_start[b]:bus_inj_start[b+1]-1]`.
"""
function build_bus_injection_index(bus_map::AbstractVector{Int32}, nbus::Int)
    counts = zeros(Int32, nbus)
    for ik in eachindex(bus_map)
        counts[Int(bus_map[ik])] += one(Int32)
    end
    bus_inj_start = Vector{Int32}(undef, nbus + 1)
    bus_inj_start[1] = one(Int32)
    for b in 1:nbus
        bus_inj_start[b + 1] = bus_inj_start[b] + counts[b]
    end
    bus_inj_list = Vector{Int32}(undef, length(bus_map))
    fill!(counts, zero(Int32))
    for ik in eachindex(bus_map)
        b = Int(bus_map[ik])
        pos = bus_inj_start[b] + counts[b]
        bus_inj_list[pos] = Int32(ik)
        counts[b] += one(Int32)
    end
    return bus_inj_start, bus_inj_list
end

@kernel function bus_injection_reduce_perbus_ka!(f, inj, bus_inj_start, bus_inj_list, @Const(nbus), @Const(net_ptr))
    b = @index(Global)
    @inbounds if b <= nbus
        vr_idx = net_ptr + 2 * (b - 1) + 1
        vi_idx = vr_idx + 1
        s_re = zero(eltype(f))
        s_im = zero(eltype(f))
        for pos in bus_inj_start[b]:(bus_inj_start[b + 1] - Int32(1))
            ik = Int(bus_inj_list[pos])
            s_re += inj[2 * ik - 1]
            s_im += inj[2 * ik]
        end
        f[vr_idx] += s_re
        f[vi_idx] += s_im
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
# Batched 2D KA kernels (phase 14c D6)
#
# Each kernel launches with ndrange = n_devices and loops over M
# scenarios internally. Calls existing _*_one! leaf functions via
# @view(array[m, :]) for 1D access into 2D arrays.
# -----------------------------------------------------------------------

@kernel function genrou_residual_batched_ka!(f, z, u, p, inj, online,
        diff_ptr, alg_ptr, ctrl_ptr, par_ptr, bus_arr,
        @Const(diff_dim), @Const(net_ptr), @Const(twopi60), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._genrou_residual_one!(
                @view(f[m,:]), @view(z[m,:]), @view(u[m,:]), @view(p[m,:]),
                diff_ptr, alg_ptr, ctrl_ptr, par_ptr, bus_arr,
                k, diff_dim, net_ptr, twopi60, @view(inj[m,:]), k)
        end
    end
end

@kernel function ieesgo_residual_batched_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._ieesgo_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, alg_ptr, par_ptr, w_idx_arr, k, diff_dim)
        end
    end
end

@kernel function tgov1_residual_batched_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._tgov1_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, alg_ptr, par_ptr, w_idx_arr, k, diff_dim)
        end
    end
end

@kernel function sexs_residual_batched_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._sexs_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
        end
    end
end

@kernel function esdc1a_residual_batched_ka!(f, z, p, online,
        diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._esdc1a_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, par_ptr, vr_idx_arr, vs_idx_arr, k)
        end
    end
end

@kernel function ieeest_residual_batched_ka!(f, z, p, online,
        diff_ptr, alg_ptr, par_ptr, w_idx_arr,
        @Const(diff_dim), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._ieeest_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, alg_ptr, par_ptr, w_idx_arr, diff_dim, k)
        end
    end
end

@kernel function zipload_residual_batched_ka!(f, z, p, inj, online,
        bus_arr, par_ptr,
        @Const(net_ptr), @Const(inj_offset), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._zipload_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                bus_arr, par_ptr, k, net_ptr, @view(inj[m,:]), inj_offset + k)
        end
    end
end

@kernel function static_gen_residual_batched_ka!(f, z, p, inj, online,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr,
        @Const(inj_offset), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._static_gen_residual_one!(@view(f[m,:]), @view(z[m,:]), @view(p[m,:]),
                vr_idx_arr, alg_ptr, par_ptr, bus_type_arr,
                k, @view(inj[m,:]), inj_offset + k)
        end
    end
end

# Per-bus injection reduce, batched over M scenarios
@kernel function bus_injection_reduce_batched_ka!(f, inj, bus_inj_start, bus_inj_list,
        @Const(nbus), @Const(net_ptr), @Const(M))
    b = @index(Global)
    @inbounds if b <= nbus
        vr_idx = net_ptr + 2 * (b - 1) + 1
        vi_idx = vr_idx + 1
        for m in 1:M
            s_re = zero(eltype(f))
            s_im = zero(eltype(f))
            for pos in bus_inj_start[b]:(bus_inj_start[b + 1] - Int32(1))
                ik = Int(bus_inj_list[pos])
                s_re += inj[m, 2*ik - 1]
                s_im += inj[m, 2*ik]
            end
            f[m, vr_idx] += s_re
            f[m, vi_idx] += s_im
        end
    end
end

# Row extraction/placement kernels for cuSPARSE SpMV on 2D arrays
@kernel function _extract_row_ka!(dst, src, @Const(m), @Const(col_start), @Const(n))
    i = @index(Global)
    @inbounds dst[i] = src[m, col_start + i - 1]
end

@kernel function _place_row_ka!(dst, src, @Const(m), @Const(col_start), @Const(n))
    i = @index(Global)
    @inbounds dst[m, col_start + i - 1] = src[i]
end

function _extract_row_range!(dst::CuVector, src::CuMatrix, m::Int, col_start::Int, n::Int)
    backend = CUDABackend()
    kernel = _extract_row_ka!(backend)
    kernel(dst, src, m, col_start, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
end

function _place_row_range!(dst::CuMatrix, src::CuVector, m::Int, col_start::Int, n::Int)
    backend = CUDABackend()
    kernel = _place_row_ka!(backend)
    kernel(dst, src, m, col_start, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
end

# Batched Jacobian kernels (D6) — same pattern, loop M internally

@kernel function genrou_jacobian_batched_ka!(nzval, z, p, online,
        diff_ptr, alg_ptr, par_ptr, bus_arr, jac_pos, has_gov, has_exc,
        @Const(diff_dim), @Const(net_ptr), @Const(twopi60), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._genrou_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                diff_ptr, alg_ptr, par_ptr, bus_arr, jac_pos, has_gov, has_exc,
                k, diff_dim, net_ptr, twopi60)
        end
    end
end

@kernel function ieesgo_jacobian_batched_ka!(nzval, p, online,
        par_ptr, jac_pos, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._ieesgo_jacobian_one!(@view(nzval[m,:]), @view(p[m,:]),
                par_ptr, jac_pos, k)
        end
    end
end

@kernel function tgov1_jacobian_batched_ka!(nzval, p, online,
        par_ptr, jac_pos, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._tgov1_jacobian_one!(@view(nzval[m,:]), @view(p[m,:]),
                par_ptr, jac_pos, k)
        end
    end
end

@kernel function sexs_jacobian_batched_ka!(nzval, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, jac_pos, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._sexs_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                par_ptr, vr_idx_arr, vs_idx_arr, jac_pos, k)
        end
    end
end

@kernel function esdc1a_jacobian_batched_ka!(nzval, z, p, online,
        par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._esdc1a_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                par_ptr, vr_idx_arr, vs_idx_arr, diff_ptr, jac_pos, k)
        end
    end
end

@kernel function ieeest_jacobian_batched_ka!(nzval, z, p, online,
        par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos,
        @Const(diff_dim), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._ieeest_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                par_ptr, diff_ptr, alg_ptr, w_idx_arr, jac_pos, diff_dim, k)
        end
    end
end

@kernel function zipload_jacobian_batched_ka!(nzval, z, p, online,
        bus_arr, par_ptr, jac_pos,
        @Const(net_ptr), @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._zipload_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                bus_arr, par_ptr, jac_pos, k, net_ptr)
        end
    end
end

@kernel function static_gen_jacobian_batched_ka!(nzval, z, p, online,
        vr_idx_arr, alg_ptr, par_ptr, bus_type_arr, jac_pos, @Const(M))
    k = @index(Global)
    @inbounds if online[k]
        for m in 1:M
            GradPower._static_gen_jacobian_one!(@view(nzval[m,:]), @view(z[m,:]), @view(p[m,:]),
                vr_idx_arr, alg_ptr, par_ptr, bus_type_arr, jac_pos, k)
        end
    end
end

# Batched ybus → J_nzval copy (all M scenarios)
@kernel function ybus_to_jnzval_batched_ka!(nzval, ybus_nzval, ybus_to_jnz, @Const(M))
    i = @index(Global)
    @inbounds begin
        jpos = ybus_to_jnz[i]
        if jpos > 0
            val = -ybus_nzval[i]
            for m in 1:M
                nzval[m, jpos] = val
            end
        end
    end
end

# Batched backward Euler Jacobian scaling
@kernel function beuler_jac_scale_batched_ka!(nzval, J_colptr, J_rowval, is_diff,
        @Const(n_da), @Const(sys_dim), @Const(dt), @Const(M))
    col = @index(Global)
    @inbounds for nz_idx in J_colptr[col]:(J_colptr[col+1]-1)
        row = J_rowval[nz_idx]
        if row <= n_da && is_diff[row]
            for m in 1:M
                nzval[m, nz_idx] *= -dt
                if row == col
                    nzval[m, nz_idx] += 1.0
                end
            end
        end
    end
end

@kernel function beuler_jac_scale_simple_batched_ka!(nzval, J_colptr, J_rowval,
        @Const(diff_dim), @Const(sys_dim), @Const(dt), @Const(M))
    col = @index(Global)
    @inbounds for nz_idx in J_colptr[col]:(J_colptr[col+1]-1)
        row = J_rowval[nz_idx]
        if row <= diff_dim
            for m in 1:M
                nzval[m, nz_idx] *= -dt
                if row == col
                    nzval[m, nz_idx] += 1.0
                end
            end
        end
    end
end

# Batched backward Euler diff residual scaling
@kernel function beuler_diff_batched_ka!(f, z, zold, diff_indices, @Const(dt), @Const(M))
    idx = @index(Global)
    @inbounds begin
        i = diff_indices[idx]
        for m in 1:M
            f[m, i] = z[m, i] - zold[m, i] - dt * f[m, i]
        end
    end
end

# -----------------------------------------------------------------------
# GPU residual evaluation — batched (phase 14c D6)
#
# One kernel launch per device type covering all M scenarios.
# Each kernel loops M internally using @view(array[m,:]).
# Zero per-scenario allocation, zero CPU→GPU transfers.
# -----------------------------------------------------------------------

function _residual_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                      L::GradPower.SimulationLayout)
    M = gbl.M
    backend = CUDABackend()
    diff_dim = gbl.diff_dim
    alg_dim  = gbl.alg_dim
    net_ptr  = diff_dim + alg_dim
    twopi60  = 2.0 * π * 60.0

    gt = gbl.gpu_genrou
    zt = gbl.gpu_zipload
    st = gbl.gpu_static_gen
    ig = gbl.gpu_ieesgo
    tg = gbl.gpu_tgov1
    sx = gbl.gpu_sexs
    ex = gbl.gpu_esdc1a
    pss = gbl.gpu_ieeest
    n_genrou = gbl.inj_meta_n_genrou
    n_zipload = gbl.inj_meta_n_zipload
    n_total = gbl.inj_meta_n_total

    # 1. Zero f and inj
    fill!(gbl.f, 0.0)
    fill!(gbl.inj, 0.0)

    # 2. Batched ybus SpMV via per-scenario cuSPARSE mv!
    # Uses pre-allocated v_buf/fv_buf stored alongside gbl
    nv = 2 * gbl.nbus
    if nv > 0
        v_buf = CuVector{Float64}(undef, nv)
        fv_buf = CuVector{Float64}(undef, nv)
        for m in 1:M
            # Extract voltage subvector: v_buf = z[m, net_ptr+1:end]
            _extract_row_range!(v_buf, gbl.z, m, net_ptr+1, nv)
            CUSPARSE.mv!('N', -1.0, gbl.ybus_csr, v_buf, 0.0, fv_buf, 'O')
            # Place result: f[m, net_ptr+1:end] = fv_buf
            _place_row_range!(gbl.f, fv_buf, m, net_ptr+1, nv)
        end
    end

    # 3. uvec routing (one kernel, M-loop inside)
    if length(gbl.uvec_idx) > 0
        kernel = uvec_routing_gpu_kernel!(backend)
        kernel(gbl.u, gbl.z, gbl.uvec_idx; ndrange=length(gbl.uvec_idx))
    end

    # 4. Device residual kernels — one launch per device type
    if gt.n > 0
        kernel = genrou_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.u, gbl.p, gbl.inj, gt.online,
               gt.diff_ptr, gt.alg_ptr, gt.ctrl_ptr, gt.par_ptr, gt.bus,
               diff_dim, net_ptr, twopi60, M; ndrange=gt.n)
    end
    if ig.n > 0
        kernel = ieesgo_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, ig.online,
               ig.diff_ptr, ig.alg_ptr, ig.par_ptr, ig.w_idx,
               diff_dim, M; ndrange=ig.n)
    end
    if tg.n > 0
        kernel = tgov1_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, tg.online,
               tg.diff_ptr, tg.alg_ptr, tg.par_ptr, tg.w_idx,
               diff_dim, M; ndrange=tg.n)
    end
    if pss.n > 0
        kernel = ieeest_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, pss.online,
               pss.diff_ptr, pss.alg_ptr, pss.par_ptr, pss.w_idx,
               diff_dim, M; ndrange=pss.n)
    end
    if sx.n > 0
        kernel = sexs_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, sx.online,
               sx.diff_ptr, sx.par_ptr, sx.vr_idx, sx.vs_idx, M; ndrange=sx.n)
    end
    if ex.n > 0
        kernel = esdc1a_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, ex.online,
               ex.diff_ptr, ex.par_ptr, ex.vr_idx, ex.vs_idx, M; ndrange=ex.n)
    end
    if zt.n > 0
        kernel = zipload_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, gbl.inj, zt.online,
               zt.bus, zt.par_ptr, net_ptr, n_genrou, M; ndrange=zt.n)
    end
    if st.n > 0
        kernel = static_gen_residual_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.p, gbl.inj, st.online,
               st.vr_idx, st.alg_ptr, st.par_ptr, st.bus_type,
               n_genrou + n_zipload, M; ndrange=st.n)
    end

    # 5. Bus injection reduce — one launch, M-loop inside
    if n_total > 0
        kernel = bus_injection_reduce_batched_ka!(backend)
        kernel(gbl.f, gbl.inj, gbl.bus_inj_start, gbl.bus_inj_list,
               gbl.nbus, net_ptr, M; ndrange=gbl.nbus)
    end

    # 6. Events — update status from CPU (O(n_events), small)
    events = dyn.events
    if !isempty(events)
        status_cpu = Bool[ev.status for ev in events]
        copyto!(gbl.event_status, CuVector(status_cpu))
        kernel = events_fun_gpu_kernel!(backend)
        kernel(gbl.f, gbl.z, gbl.event_bus, gbl.event_rfault, gbl.event_status, net_ptr;
               ndrange=length(events))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end

# -----------------------------------------------------------------------
# GPU Jacobian — batched (phase 14c D6)
#
# One kernel launch per device type covering all M scenarios.
# -----------------------------------------------------------------------

function _jacobian_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                      L::GradPower.SimulationLayout)
    M = gbl.M
    backend = CUDABackend()
    diff_dim = gbl.diff_dim
    alg_dim  = gbl.alg_dim
    net_ptr  = diff_dim + alg_dim
    twopi60  = 2.0 * π * 60.0

    gt = gbl.gpu_genrou
    zt = gbl.gpu_zipload
    st = gbl.gpu_static_gen
    ig = gbl.gpu_ieesgo
    tg = gbl.gpu_tgov1
    sx = gbl.gpu_sexs
    ex = gbl.gpu_esdc1a
    pss = gbl.gpu_ieeest
    ybus_nnz = length(gbl.ybus_nzval_cpu)

    # 1. Zero J_nzval
    fill!(gbl.J_nzval, 0.0)

    # 2. Copy −ybus into J_nzval (all M scenarios)
    if ybus_nnz > 0
        kernel = ybus_to_jnzval_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.ybus_nzval_gpu, gbl.ybus_to_jnz, M; ndrange=ybus_nnz)
    end

    # 3. Device Jacobians — one launch per type
    if gt.n > 0
        kernel = genrou_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, gt.online,
               gt.diff_ptr, gt.alg_ptr, gt.par_ptr, gt.bus, gt.jac_pos,
               gt.has_gov, gt.has_exc,
               diff_dim, net_ptr, twopi60, M; ndrange=gt.n)
    end
    if ig.n > 0
        kernel = ieesgo_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.p, ig.online,
               ig.par_ptr, ig.jac_pos, M; ndrange=ig.n)
    end
    if tg.n > 0
        kernel = tgov1_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.p, tg.online,
               tg.par_ptr, tg.jac_pos, M; ndrange=tg.n)
    end
    if sx.n > 0
        kernel = sexs_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, sx.online,
               sx.par_ptr, sx.vr_idx, sx.vs_idx, sx.jac_pos, M; ndrange=sx.n)
    end
    if ex.n > 0
        kernel = esdc1a_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, ex.online,
               ex.par_ptr, ex.vr_idx, ex.vs_idx, ex.diff_ptr, ex.jac_pos, M; ndrange=ex.n)
    end
    if pss.n > 0
        kernel = ieeest_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, pss.online,
               pss.par_ptr, pss.diff_ptr, pss.alg_ptr, pss.w_idx, pss.jac_pos,
               diff_dim, M; ndrange=pss.n)
    end
    if zt.n > 0
        kernel = zipload_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, zt.online,
               zt.bus, zt.par_ptr, zt.jac_pos, net_ptr, M; ndrange=zt.n)
    end
    if st.n > 0
        kernel = static_gen_jacobian_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.z, gbl.p, st.online,
               st.vr_idx, st.alg_ptr, st.par_ptr, st.bus_type, st.jac_pos, M; ndrange=st.n)
    end

    # 4. Event Jacobian: add -1/rfault to bus diagonals for active faults
    events = dyn.events
    if !isempty(events)
        status_cpu = Bool[ev.status for ev in events]
        copyto!(gbl.event_status, CuVector(status_cpu))
        kernel = events_jac_gpu_kernel!(backend)
        kernel(gbl.J_nzval, gbl.event_rfault, gbl.event_status,
               gbl.event_jac_diag_pos, M; ndrange=length(events))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end

# -----------------------------------------------------------------------
# GPU backward Euler — batched (phase 14c D6)
# -----------------------------------------------------------------------

function _beuler_all_scenarios_gpu!(gbl::GpuBatchedLayout, dyn::GradPower.PowerSystemDynamics,
                                     L::GradPower.SimulationLayout, dt::Float64)
    _residual_all_scenarios_gpu!(gbl, dyn, L)

    if gbl.diff_indices !== nothing
        backend = CUDABackend()
        kernel = beuler_diff_batched_ka!(backend)
        kernel(gbl.f, gbl.z, gbl.zold, gbl.diff_indices, dt, gbl.M;
               ndrange=length(gbl.diff_indices))
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

    # Apply backward Euler scaling — batched, one launch
    backend = CUDABackend()
    is_diff = dyn.is_diff
    sys_dim = gbl.sys_dim

    if is_diff !== nothing
        n_da = length(is_diff)
        kernel = beuler_jac_scale_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.J_colptr, gbl.J_rowval, gbl.is_diff,
               n_da, sys_dim, dt, gbl.M; ndrange=sys_dim)
    else
        kernel = beuler_jac_scale_simple_batched_ka!(backend)
        kernel(gbl.J_nzval, gbl.J_colptr, gbl.J_rowval,
               gbl.diff_dim, sys_dim, dt, gbl.M; ndrange=sys_dim)
    end
    KernelAbstractions.synchronize(backend)
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

        # D7: GPU convergence check — no full download
        max_norm = CUDA.mapreduce(abs, max, gbl.f)
        max_norm < tol && return true

        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)

        J_nzval_cpu = Array(gbl.J_nzval)
        z_cpu = Array(gbl.z)
        f_cpu = Array(gbl.f)

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
# GPU Schur complement operator (D4) — matrix-free
#
# Computes S·x = D·x − Σ_k C_k A_k⁻¹ B_k·x without assembling S.
# D is the reduced subblock of J (voltage + trivial rows/cols).
# Cluster elimination uses cuBLAS batched LU (GpuBatchedLU).
# -----------------------------------------------------------------------

"""
    GpuSchurOperator

GPU-resident matrix-free Schur complement operator.
Implements size, eltype, mul! for Krylov.jl compatibility.

D·x uses cuSPARSE SpMV. A_k⁻¹ uses pre-factored GpuBatchedLU.
B_k and C_k are stored on CPU (small, 2×w_k) and applied in a
host-side loop per cluster group.
"""
struct GpuSchurOperator
    n_red::Int
    D_gpu::CuSparseMatrixCSR{Float64, Int32}
    batched_lus::Vector{GpuBatchedLU}
    B_cpu::Vector{Vector{Matrix{Float64}}}
    C_cpu::Vector{Vector{Matrix{Float64}}}
    group_bus_reduced::Vector{Vector{Tuple{Int,Int}}}
end

Base.size(op::GpuSchurOperator) = (op.n_red, op.n_red)
Base.eltype(::GpuSchurOperator) = Float64

"""
    GpuSchurOperator(sw, J, ct, net_ptr)

Construct from an assembled SchurWorkspace and the Jacobian J.
Re-extracts A_k/B_k/C_k from J (assemble_schur! overwrites A_k_bufs
with LU factors). Recovers D = S + Σ D_k and uploads to GPU.
"""
function GpuSchurOperator(sw::GradPower.SchurWorkspace, J::SparseMatrixCSC,
                           ct::GradPower.ClusterTable, net_ptr::Int)
    n_red = length(sw.reduced_idx)
    g2r = sw.global_to_reduced

    # D = S + Σ D_k  (undo the Schur elimination to recover D)
    D_cpu = copy(sw.S)
    for (g, group) in enumerate(sw.nt_groups), (ki, ci) in enumerate(group)
        cl = ct.clusters[ci]
        vr_l = g2r[net_ptr + 2*(cl.bus - 1) + 1]
        vi_l = g2r[net_ptr + 2*(cl.bus - 1) + 2]
        GradPower._subtract_2x2_from_sparse!(D_cpu, vr_l, vi_l, -sw.D_k_bufs[g][ki])
    end

    ng = length(sw.nt_groups)
    batched_lus = Vector{GpuBatchedLU}(undef, ng)
    B_all = Vector{Vector{Matrix{Float64}}}(undef, ng)
    C_all = Vector{Vector{Matrix{Float64}}}(undef, ng)
    bus_all = Vector{Vector{Tuple{Int,Int}}}(undef, ng)

    for g in 1:ng
        group = sw.nt_groups[g]
        nc = length(group)
        wk = ct.clusters[group[1]].w_size
        glu = GpuBatchedLU(wk, nc, 1)
        A_pk = zeros(Float64, wk, wk, nc)
        Bv = Vector{Matrix{Float64}}(undef, nc)
        Cv = Vector{Matrix{Float64}}(undef, nc)
        bv = Vector{Tuple{Int,Int}}(undef, nc)
        for (ki, ci) in enumerate(group)
            cl = ct.clusters[ci]
            A_buf = zeros(wk, wk); B_buf = zeros(wk, 2); C_buf = zeros(2, wk)
            GradPower.extract_Ak!(A_buf, J, cl)
            GradPower.extract_Bk_Ck!(B_buf, C_buf, J, cl, net_ptr)
            A_pk[:,:,ki] .= A_buf; Bv[ki] = B_buf; Cv[ki] = C_buf
            bv[ki] = (g2r[net_ptr + 2*(cl.bus-1) + 1], g2r[net_ptr + 2*(cl.bus-1) + 2])
        end
        copyto!(glu.A_packed, CuArray(A_pk))
        gpu_batched_lu_factor!(glu)
        batched_lus[g] = glu; B_all[g] = Bv; C_all[g] = Cv; bus_all[g] = bv
    end

    return GpuSchurOperator(n_red, CuSparseMatrixCSR(D_cpu),
                             batched_lus, B_all, C_all, bus_all)
end

function LinearAlgebra.mul!(y::CuVector{Float64}, op::GpuSchurOperator,
                             x::CuVector{Float64})
    # 1. y = D * x  (cuSPARSE SpMV)
    CUSPARSE.mv!('N', 1.0, op.D_gpu, x, 0.0, y, 'O')

    # 2. y_bus -= C_k A_k⁻¹ B_k x_bus  for each non-trivial cluster group
    x_h = Array(x); y_h = Array(y)
    for g in eachindex(op.batched_lus)
        glu = op.batched_lus[g]
        nc = glu.n_clusters; wk = glu.w_k
        rhs = zeros(Float64, wk, 2, nc)
        for ki in 1:nc
            vr_l, vi_l = op.group_bus_reduced[g][ki]
            B = op.B_cpu[g][ki]; xv = x_h[vr_l]; xi = x_h[vi_l]
            @inbounds for i in 1:wk
                rhs[i, 1, ki] = B[i,1]*xv + B[i,2]*xi
            end
        end
        copyto!(glu.B_packed, CuArray(rhs))
        gpu_batched_lu_solve!(glu)
        sol = Array(glu.B_packed)
        for ki in 1:nc
            vr_l, vi_l = op.group_bus_reduced[g][ki]
            C = op.C_cpu[g][ki]
            @inbounds for i in 1:wk
                y_h[vr_l] -= C[1,i] * sol[i,1,ki]
                y_h[vi_l] -= C[2,i] * sol[i,1,ki]
            end
        end
    end
    copyto!(y, CuVector(y_h))
    return y
end

# AbstractVector fallback for Krylov.jl compatibility
function LinearAlgebra.mul!(y::AbstractVector, op::GpuSchurOperator,
                             x::AbstractVector)
    x_d = x isa CuVector ? x : CuVector(Float64.(x))
    y_d = y isa CuVector ? y : CUDA.zeros(Float64, op.n_red)
    mul!(y_d, op, x_d)
    y isa CuVector || copyto!(y, Array(y_d))
    return y
end

# -----------------------------------------------------------------------
# Schur-complement Newton step for GPU batched layout
# -----------------------------------------------------------------------

"""
    _newton_step_schur_gpu!(gbl, dyn, L, sw, dt; itermax=30, tol=1e-10)

Backward-Euler Newton loop using Schur-complement reduction.
Residual on GPU, Jacobian + Schur assembly + KLU solve on CPU per scenario.
"""
function _newton_step_schur_gpu!(
    gbl::GpuBatchedLayout,
    dyn::GradPower.PowerSystemDynamics,
    L::GradPower.SimulationLayout,
    sw::GradPower.SchurWorkspace,
    dt::Float64;
    itermax::Int = 30,
    tol::Float64 = 1e-10,
)
    M       = gbl.M
    sys_dim = gbl.sys_dim
    ct      = dyn.clusters::GradPower.ClusterTable
    net_ptr = dyn.diff_dim + dyn.alg_dim
    n_red   = length(sw.reduced_idx)

    J_colptr_cpu = Array(gbl.J_colptr)
    J_rowval_cpu = Array(gbl.J_rowval)
    nnz_J = size(gbl.J_nzval, 2)

    J_m = SparseMatrixCSC(sys_dim, sys_dim,
                          copy(J_colptr_cpu), copy(J_rowval_cpu),
                          zeros(Float64, nnz_J))

    first_klu = true

    for iter in 1:itermax
        _beuler_all_scenarios_gpu!(gbl, dyn, L, dt)

        # D7: GPU convergence check
        max_norm = CUDA.mapreduce(abs, max, gbl.f)
        max_norm < tol && return true

        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)
        J_nzval_cpu = Array(gbl.J_nzval)
        z_cpu = Array(gbl.z)
        f_cpu = Array(gbl.f)

        for m in 1:M
            copyto!(J_m.nzval, view(J_nzval_cpu, m, :))

            GradPower.assemble_schur!(sw, J_m, ct, net_ptr)

            @inbounds for i in 1:n_red
                sw.rhs_red[i] = f_cpu[m, sw.reduced_idx[i]]
            end

            for (g, group) in enumerate(sw.nt_groups)
                for (ki, ci) in enumerate(group)
                    cl = ct.clusters[ci]
                    wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                    vr_g = net_ptr + 2*(bus - 1) + 1
                    vr_l = sw.global_to_reduced[vr_g]
                    vi_l = sw.global_to_reduced[vr_g + 1]

                    A    = sw.A_k_bufs[g][ki]
                    C    = sw.C_k_bufs[g][ki]
                    ipiv = sw.lu_pivots[g][ki]
                    tmp  = sw.tmp_w[g][ki]

                    @inbounds for i in 1:wk; tmp[i] = f_cpu[m, ws + i - 1]; end
                    GradPower._lu_solve!(A, ipiv, tmp, wk)

                    @inbounds for i in 1:wk
                        sw.rhs_red[vr_l] -= C[1, i] * tmp[i]
                        sw.rhs_red[vi_l] -= C[2, i] * tmp[i]
                    end
                end
            end

            @inbounds for i in 1:n_red; sw.rhs_red[i] = -sw.rhs_red[i]; end

            if first_klu
                sw.S_fact[] = klu(sw.S)
            else
                klu!(sw.S_fact[], sw.S)
            end
            ldiv!(sw.S_fact[], sw.rhs_red)

            @inbounds for i in 1:n_red
                sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
            end

            for (g, group) in enumerate(sw.nt_groups)
                for (ki, ci) in enumerate(group)
                    cl = ct.clusters[ci]
                    wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                    vr_g = net_ptr + 2*(bus - 1) + 1

                    A    = sw.A_k_bufs[g][ki]
                    B    = sw.B_k_bufs[g][ki]
                    ipiv = sw.lu_pivots[g][ki]
                    tmp  = sw.tmp_w[g][ki]

                    dv1 = sw.dz[vr_g]
                    dv2 = sw.dz[vr_g + 1]

                    @inbounds for i in 1:wk
                        tmp[i] = f_cpu[m, ws + i - 1] + B[i, 1] * dv1 + B[i, 2] * dv2
                    end
                    GradPower._lu_solve!(A, ipiv, tmp, wk)

                    @inbounds for i in 1:wk
                        sw.dz[ws + i - 1] = -tmp[i]
                    end
                end
            end

            @inbounds for k in 1:sys_dim
                z_cpu[m, k] += sw.dz[k]
            end
        end

        first_klu = false
        copyto!(gbl.z, CuMatrix(z_cpu))
    end

    return false
end

# -----------------------------------------------------------------------
# integrate_gpu_schur! — uses Schur-complement Newton
# -----------------------------------------------------------------------

"""
    integrate_gpu_schur!(gbl, ps, sw, tf; dt=1/120, newton_tol=1e-10)

GPU-resident batched integration using Schur-complement Newton steps.
Residual on GPU, Jacobian + Schur reduction + KLU solve on CPU.
"""
function integrate_gpu_schur!(
    gbl::GpuBatchedLayout,
    ps::GradPower.PowerSystem,
    sw::GradPower.SchurWorkspace,
    tf::Float64;
    dt::Float64 = 1.0 / 120.0,
    newton_tol::Float64 = 1e-10,
)
    dyn     = ps.dynamic::GradPower.PowerSystemDynamics
    L       = dyn.layout::GradPower.SimulationLayout
    M       = gbl.M
    sys_dim = gbl.sys_dim

    nsteps = Int(round(tf / dt))
    tvec   = collect(0:dt:tf)

    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule, by = x -> x[1])

    z0_cpu = Array(gbl.z)
    trajs  = [zeros(Float64, sys_dim, nsteps + 1) for _ in 1:M]
    for m in 1:M
        trajs[m][:, 1] .= z0_cpu[m, :]
    end

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_schur_gpu!(gbl, dyn, L, sw, dt; tol = newton_tol)

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
            _newton_step_schur_gpu!(gbl, dyn, L, sw, 0.0; tol = newton_tol)
        end
    end

    for event in events
        GradPower.deactivate!(event)
    end

    return tvec, trajs
end

# -----------------------------------------------------------------------
# CSC→CSR nzval gather kernel (for cuDSS)
# -----------------------------------------------------------------------

@kernel function csc_to_csr_gather_ka!(csr_nzval, csc_nzval, perm)
    k = @index(Global)
    @inbounds csr_nzval[k] = csc_nzval[perm[k]]
end

# -----------------------------------------------------------------------
# GPU monolithic Newton step via cuDSS direct solve
#
# Everything on GPU. Zero host↔device transfers per Newton iteration.
# Per scenario: gather J_nzval[m,:] into CSR order, cuDSS refactorize,
# cuDSS solve, update z[m,:].
# -----------------------------------------------------------------------

function _newton_step_cudss_gpu!(
    gbl::GpuBatchedLayout,
    dyn::GradPower.PowerSystemDynamics,
    L::GradPower.SimulationLayout,
    dt::Float64;
    itermax::Int = 30,
    tol::Float64 = 1e-10,
)
    M       = gbl.M
    sys_dim = gbl.sys_dim
    nnz_J   = size(gbl.J_nzval, 2)
    backend = CUDABackend()

    # Scratch buffer for one scenario's CSC nzvals (1D view from 2D J_nzval)
    csc_buf = CUDA.zeros(Float64, nnz_J)
    f_flat = reshape(gbl.f, :)
    n_flat = length(f_flat)
    tol_l2 = tol * sqrt(Float64(sys_dim * M))

    for iter in 1:itermax
        # 1. Residual + backward Euler on GPU
        _beuler_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 2. Convergence check — CUBLAS nrm2 (no full download)
        norm_f = CUDA.CUBLAS.nrm2(f_flat)
        norm_f < tol_l2 && return true

        # 3. Jacobian + backward Euler scaling on GPU
        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 4. Per-scenario: cuDSS factorize + solve
        for m in 1:M
            # 4a. Extract scenario m's nzvals (CSC order) into contiguous buffer
            copyto!(csc_buf, view(gbl.J_nzval, m, :))

            # 4b. Gather into CSR order in-place on the solver's CSR
            kernel = csc_to_csr_gather_ka!(backend)
            kernel(gbl.cudss_csr.nzVal, csc_buf, gbl.csc_to_csr_perm; ndrange=nnz_J)
            KernelAbstractions.synchronize(backend)

            # 4c. Numeric refactorization (symbolic already done at setup)
            cudss("factorization", gbl.cudss_solver, gbl.cudss_sol, gbl.cudss_rhs)

            # 4d. Build RHS = -f[m, :]
            copyto!(gbl.cudss_rhs, view(gbl.f, m, :))
            CUDA.CUBLAS.scal!(sys_dim, -1.0, gbl.cudss_rhs)

            # 4e. Solve J * dz = -f
            cudss("solve", gbl.cudss_solver, gbl.cudss_sol, gbl.cudss_rhs)

            # 4f. Update z[m, :] += dz
            # Use a kernel to add sol into z[m, :]
            _add_to_row!(gbl.z, gbl.cudss_sol, m, sys_dim, backend)
        end
    end

    return false
end

@kernel function _add_to_row_ka!(z, dz, @Const(m), @Const(n))
    j = @index(Global)
    @inbounds if j <= n
        z[m, j] += dz[j]
    end
end

function _add_to_row!(z::CuMatrix, dz::CuVector, m::Int, n::Int, backend)
    kernel = _add_to_row_ka!(backend)
    kernel(z, dz, m, n; ndrange=n)
    KernelAbstractions.synchronize(backend)
end

# -----------------------------------------------------------------------
# integrate_gpu_cudss! — fully GPU-resident with cuDSS direct solve
# -----------------------------------------------------------------------

"""
    integrate_gpu_cudss!(gbl, ps, tf; dt=1/120, newton_tol=1e-10)

GPU-resident batched integration with cuDSS monolithic direct solve.
Zero CPU↔GPU transfers in the Newton loop. Trajectory snapshots
downloaded once per time step.
"""
function integrate_gpu_cudss!(
    gbl::GpuBatchedLayout,
    ps::GradPower.PowerSystem,
    tf::Float64;
    dt::Float64 = 1.0 / 120.0,
    newton_tol::Float64 = 1e-10,
)
    dyn     = ps.dynamic::GradPower.PowerSystemDynamics
    L       = dyn.layout::GradPower.SimulationLayout
    M       = gbl.M
    sys_dim = gbl.sys_dim

    nsteps = Int(round(tf / dt))
    tvec   = collect(0:dt:tf)

    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule, by = x -> x[1])

    z0_cpu = Array(gbl.z)
    trajs  = [zeros(Float64, sys_dim, nsteps + 1) for _ in 1:M]
    for m in 1:M
        trajs[m][:, 1] .= z0_cpu[m, :]
    end

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_cudss_gpu!(gbl, dyn, L, dt; tol = newton_tol)

        # Trajectory snapshot (one download per step)
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
            # Update event arrays on GPU
            for (ei, ev) in enumerate(events)
                CUDA.@allowscalar gbl.event_status[ei] = ev.status
            end
            copyto!(gbl.zold, gbl.z)
            _newton_step_cudss_gpu!(gbl, dyn, L, 0.0; tol = newton_tol)
        end
    end

    for event in events
        GradPower.deactivate!(event)
    end

    return tvec, trajs
end

end # module GradPowerCUDAExt
