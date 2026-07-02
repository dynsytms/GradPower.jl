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
# Batched dense LU — cuBLAS getrf_strided_batched! / getrs_strided_batched!
# (moved before GpuBatchedLayout so the type is available for fields)
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
    # Schur batched LU per w_k-group and metadata for GPU Schur solver
    schur_batched_lus::Vector{GpuBatchedLU}
    schur_nt_groups::Vector{Vector{Int}}      # nt_groups from SchurWorkspace
    schur_group_A_offsets::Vector{Int}         # start/end in flat A_jnz for each group
    schur_group_B_offsets::Vector{Int}
    schur_group_C_offsets::Vector{Int}
    schur_group_D_offsets::Vector{Int}         # 4 entries per cluster, sequential
    schur_cluster_bus_reduced::Vector{Vector{Tuple{Int,Int}}} # (vr_l, vi_l) per cluster per group
    schur_cluster_w_start::Vector{Vector{Int}} # w_start per cluster per group
    # cuDSS solver for the reduced Schur system S
    cudss_S_solver::Union{Nothing, CudssSolver{Float64, Int32}}
    cudss_S_csr::Union{Nothing, CuSparseMatrixCSR{Float64, Int32}}
    csc_to_csr_perm_S::Union{Nothing, CuVector{Int32}}
    cudss_S_rhs::Union{Nothing, CuVector{Float64}}
    cudss_S_sol::Union{Nothing, CuVector{Float64}}
    # Scratch buffers for GPU Schur Newton
    schur_fwk_packed::Vector{CuArray{Float64, 3}}     # wk × 1 × (nc*M) per group, for batched A⁻¹ f_wk solve
    schur_dz_gpu::Union{Nothing, CuVector{Float64}}   # sys_dim buffer for dz
    schur_S_csc_buf::Union{Nothing, CuVector{Float64}} # S_nnz buffer for CSC→CSR
    # GPU copies of per-cluster metadata for fused kernels
    schur_w_start_gpu::Vector{CuVector{Int}}           # per group: w_start for each cluster
    schur_vr_l_gpu::Vector{CuVector{Int}}              # per group: reduced vr index
    schur_vi_l_gpu::Vector{CuVector{Int}}              # per group: reduced vi index
    # cuDSS monolithic direct solver (phase 14c) — single scenario (legacy)
    cudss_solver::Union{Nothing, CudssSolver{Float64, Int32}}
    cudss_csr::Union{Nothing, CuSparseMatrixCSR{Float64, Int32}}  # persistent CSR shell
    csc_to_csr_perm::Union{Nothing, CuVector{Int32}}              # nzval reorder map
    cudss_rhs::Union{Nothing, CuVector{Float64}}                  # reusable RHS buffer
    cudss_sol::Union{Nothing, CuVector{Float64}}                  # reusable solution buffer
    # cuDSS batched monolithic solver — all M scenarios in one call
    cudss_batch_solver::Union{Nothing, CudssSolver{Float64, Int32}}
    cudss_batch_nzval::Union{Nothing, CuMatrix{Float64}}          # nnz × M, CSR order
    cudss_batch_rhs::Union{Nothing, CuMatrix{Float64}}            # sys_dim × M
    cudss_batch_sol::Union{Nothing, CuMatrix{Float64}}            # sys_dim × M
    cudss_batch_rowPtr::Union{Nothing, CuVector{Int32}}           # shared CSR sparsity
    cudss_batch_colVal::Union{Nothing, CuVector{Int32}}           # shared CSR sparsity
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

    # --- Schur index maps (precomputed on CPU, uploaded to GPU) ---
    ct = dyn.clusters::GradPower.ClusterTable
    net_ptr_v = dyn.diff_dim + dyn.alg_dim
    sys_dim_v = bl_cpu.sys_dim
    sw_tmp = GradPower.SchurWorkspace(ps)

    reduced_idx_g = CuVector(sw_tmp.reduced_idx)
    g2r_g = CuVector(sw_tmp.global_to_reduced)
    n_red = length(sw_tmp.reduced_idx)

    # Build J_nzval→S_nzval index map for the reduced block copy.
    # For each nz in J whose (row, col) are both in reduced_idx,
    # record (J_nzval_position, S_nzval_position).
    S_rows_cpu = rowvals(sw_tmp.S)
    J_to_S_jnz = Int32[]   # J nzval positions
    J_to_S_snz = Int32[]   # S nzval positions
    for (lj, gj) in enumerate(sw_tmp.reduced_idx)
        for j_nz in J_colptr_cpu[gj]:(J_colptr_cpu[gj+1]-1)
            gi = J_rowval_cpu[j_nz]
            li = sw_tmp.global_to_reduced[gi]
            li == 0 && continue
            # Find li in S's column lj
            for s_nz in nzrange(sw_tmp.S, lj)
                if S_rows_cpu[s_nz] == li
                    push!(J_to_S_jnz, Int32(j_nz))
                    push!(J_to_S_snz, Int32(s_nz))
                    break
                end
            end
        end
    end
    J_to_S_nzpos_g = CuVector(J_to_S_snz)
    J_to_S_jnz_g   = CuVector(J_to_S_jnz)
    # Reuse J_to_S_row/col fields for jnz/snz pair arrays
    J_to_S_row_g = J_to_S_jnz_g    # J nzval indices
    J_to_S_col_g = J_to_S_nzpos_g  # S nzval indices
    J_nz_count_for_S_v = length(J_to_S_jnz)

    S_nzval_g = CuVector(zeros(Float64, nnz(sw_tmp.S)))
    S_colptr_g = CuVector(Int32.(sw_tmp.S.colptr))
    S_rowval_g = CuVector(Int32.(rowvals(sw_tmp.S)))
    S_n_v = n_red
    S_nnz_v = nnz(sw_tmp.S)

    # Build per-cluster A_k/B_k/C_k nzval index maps.
    # For each nonzero in J that falls inside A_k/B_k/C_k of cluster ci,
    # record (J_nzval_position, local_position_in_packed_buffer).
    # Local position in A_packed: linear index into w_k × w_k matrix
    # for the cluster's slice (column-major).
    A_jnz_list = Int32[]
    A_local_list = Int32[]
    B_jnz_list = Int32[]
    B_local_list = Int32[]
    C_jnz_list = Int32[]
    C_local_list = Int32[]

    # For each group, we need cluster_within_group index to know the
    # batch offset. Batch index = (ki-1)*M + m, but for index maps
    # we store the ki (1-based within group) and apply M offset in the kernel.
    # Actually, store flat: per-cluster maps with a group_offset field.
    # Simpler: store (jnz_pos, local_row, local_col, ki_in_group, group_idx).
    # But that's too many arrays. Instead, pre-flatten per group.

    # We'll build per-group maps, then concatenate with group metadata.
    nt_groups = sw_tmp.nt_groups
    ng = length(nt_groups)

    # Per-group metadata for A/B/C extraction kernel dispatch
    # group_A_offset[g] = start index into flat A_jnz_list for group g
    group_A_offsets = zeros(Int, ng + 1)
    group_B_offsets = zeros(Int, ng + 1)
    group_C_offsets = zeros(Int, ng + 1)

    # Temporary per-group lists
    for (g, group) in enumerate(nt_groups)
        wk = ct.clusters[group[1]].w_size
        for (ki, ci) in enumerate(group)
            cl = ct.clusters[ci]
            ws_cl = cl.w_start; we_cl = cl.w_end
            vr_col = net_ptr_v + 2*(cl.bus - 1) + 1
            vi_col = vr_col + 1

            # A_k: rows/cols in [ws_cl, we_cl]
            for col_global in ws_cl:we_cl
                local_col = col_global - ws_cl + 1
                for j_nz in J_colptr_cpu[col_global]:(J_colptr_cpu[col_global+1]-1)
                    row_global = J_rowval_cpu[j_nz]
                    if ws_cl <= row_global <= we_cl
                        local_row = row_global - ws_cl + 1
                        # Column-major linear index in w_k × w_k
                        local_pos = (local_col - 1) * wk + local_row
                        push!(A_jnz_list, Int32(j_nz))
                        # Encode: (ki, local_pos) — ki used by kernel to compute batch offset
                        # Pack as: flat_idx = (ki-1)*wk*wk + local_pos
                        push!(A_local_list, Int32((ki - 1) * wk * wk + local_pos))
                    end
                end
            end

            # B_k: rows in [ws_cl, we_cl], cols in {vr_col, vi_col}
            for (lcol, gcol) in enumerate((vr_col, vi_col))
                for j_nz in J_colptr_cpu[gcol]:(J_colptr_cpu[gcol+1]-1)
                    row_global = J_rowval_cpu[j_nz]
                    if ws_cl <= row_global <= we_cl
                        local_row = row_global - ws_cl + 1
                        # B_packed layout: w_k × 2 × batch, column-major
                        # local_pos = (lcol-1)*wk + local_row
                        local_pos = (lcol - 1) * wk + local_row
                        push!(B_jnz_list, Int32(j_nz))
                        push!(B_local_list, Int32((ki - 1) * wk * 2 + local_pos))
                    end
                end
            end

            # C_k: rows in {vr_col, vi_col}, cols in [ws_cl, we_cl]
            for col_global in ws_cl:we_cl
                local_col = col_global - ws_cl + 1
                for j_nz in J_colptr_cpu[col_global]:(J_colptr_cpu[col_global+1]-1)
                    row_global = J_rowval_cpu[j_nz]
                    if row_global == vr_col
                        # C_packed layout: 2 × w_k × batch
                        local_pos = (local_col - 1) * 2 + 1
                        push!(C_jnz_list, Int32(j_nz))
                        push!(C_local_list, Int32((ki - 1) * 2 * wk + local_pos))
                    elseif row_global == vi_col
                        local_pos = (local_col - 1) * 2 + 2
                        push!(C_jnz_list, Int32(j_nz))
                        push!(C_local_list, Int32((ki - 1) * 2 * wk + local_pos))
                    end
                end
            end
        end
        group_A_offsets[g + 1] = length(A_jnz_list)
        group_B_offsets[g + 1] = length(B_jnz_list)
        group_C_offsets[g + 1] = length(C_jnz_list)
    end

    cluster_A_jnz_g   = CuVector(A_jnz_list)
    cluster_A_local_g  = CuVector(A_local_list)
    cluster_B_jnz_g   = CuVector(B_jnz_list)
    cluster_B_local_g  = CuVector(B_local_list)
    cluster_C_jnz_g   = CuVector(C_jnz_list)
    cluster_C_local_g  = CuVector(C_local_list)
    n_A_entries_v = length(A_jnz_list)
    n_B_entries_v = length(B_jnz_list)
    n_C_entries_v = length(C_jnz_list)

    # D_k subtract positions: for each non-trivial cluster, the 4 S nzval
    # positions where D_k[1,1], D_k[2,1], D_k[1,2], D_k[2,2] subtract.
    D_S_nzpos_list = Int32[]
    for (g, group) in enumerate(nt_groups)
        for (ki, ci) in enumerate(group)
            cl = ct.clusters[ci]
            vr_g = net_ptr_v + 2*(cl.bus - 1) + 1
            vi_g = vr_g + 1
            vr_l = sw_tmp.global_to_reduced[vr_g]
            vi_l = sw_tmp.global_to_reduced[vi_g]
            # Find S nzval positions for (vr_l,vr_l), (vi_l,vr_l), (vr_l,vi_l), (vi_l,vi_l)
            for (col_l, row_l) in ((vr_l, vr_l), (vr_l, vi_l), (vi_l, vr_l), (vi_l, vi_l))
                found = false
                for s_nz in nzrange(sw_tmp.S, col_l)
                    if S_rows_cpu[s_nz] == row_l
                        push!(D_S_nzpos_list, Int32(s_nz))
                        found = true
                        break
                    end
                end
                @assert found "Missing S entry for D_k subtract at ($row_l, $col_l)"
            end
        end
    end
    cluster_D_S_nzpos_g = CuVector(D_S_nzpos_list)
    n_D_entries_v = length(D_S_nzpos_list)

    # Build GpuBatchedLU instances for each w_k-group
    schur_batched_lus_v = GpuBatchedLU[]
    schur_group_D_offsets_v = [0]
    schur_cluster_bus_reduced_v = Vector{Tuple{Int,Int}}[]
    schur_cluster_w_start_v = Vector{Int}[]
    for (g, group) in enumerate(nt_groups)
        nc = length(group)
        wk = ct.clusters[group[1]].w_size
        glu = GpuBatchedLU(wk, nc, M)
        push!(schur_batched_lus_v, glu)
        push!(schur_group_D_offsets_v, schur_group_D_offsets_v[end] + 4 * nc)
        bus_red = Tuple{Int,Int}[]
        ws_list = Int[]
        for (ki, ci) in enumerate(group)
            cl = ct.clusters[ci]
            vr_g = net_ptr_v + 2*(cl.bus - 1) + 1
            push!(bus_red, (sw_tmp.global_to_reduced[vr_g],
                            sw_tmp.global_to_reduced[vr_g + 1]))
            push!(ws_list, cl.w_start)
        end
        push!(schur_cluster_bus_reduced_v, bus_red)
        push!(schur_cluster_w_start_v, ws_list)
    end

    # cuDSS for reduced system S (symbolic analysis once)
    S_nnz_val = nnz(sw_tmp.S)
    S_perm_csc = SparseMatrixCSC(n_red, n_red,
                                  copy(sw_tmp.S.colptr), copy(rowvals(sw_tmp.S)),
                                  collect(Float64, 1:S_nnz_val))
    S_perm_csr = copy(S_perm_csc')
    csc_to_csr_perm_S_v = CuVector(Int32.(S_perm_csr.nzval))

    S_csc_ones = SparseMatrixCSC(n_red, n_red,
                                  copy(sw_tmp.S.colptr), copy(rowvals(sw_tmp.S)),
                                  ones(Float64, S_nnz_val))
    cudss_S_csr_v = CuSparseMatrixCSR(S_csc_ones)
    cudss_S_solver_v = CudssSolver(cudss_S_csr_v, "G", 'F')
    _tmp_sx = CUDA.zeros(Float64, n_red)
    _tmp_sb = CUDA.zeros(Float64, n_red)
    cudss("analysis", cudss_S_solver_v, _tmp_sx, _tmp_sb)
    cudss_S_rhs_v = CUDA.zeros(Float64, n_red)
    cudss_S_sol_v = CUDA.zeros(Float64, n_red)

    # Scratch buffers for GPU Schur Newton
    schur_fwk_packed_v = CuArray{Float64, 3}[]
    for (g, glu) in enumerate(schur_batched_lus_v)
        wk = glu.w_k; nc = glu.n_clusters
        push!(schur_fwk_packed_v, CUDA.zeros(Float64, wk, 1, nc * M))
    end
    schur_dz_gpu_v = CUDA.zeros(Float64, sys_dim_v)
    schur_S_csc_buf_v = CUDA.zeros(Float64, S_nnz_val)

    # GPU copies of per-cluster metadata for fused kernels
    schur_w_start_gpu_v = CuVector{Int}[]
    schur_vr_l_gpu_v = CuVector{Int}[]
    schur_vi_l_gpu_v = CuVector{Int}[]
    for (g, bus_red) in enumerate(schur_cluster_bus_reduced_v)
        push!(schur_w_start_gpu_v, CuVector(schur_cluster_w_start_v[g]))
        push!(schur_vr_l_gpu_v, CuVector([br[1] for br in bus_red]))
        push!(schur_vi_l_gpu_v, CuVector([br[2] for br in bus_red]))
    end

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

    # Batched cuDSS fields disabled — a single 217k LU factorization already
    # saturates the GPU, so batching provides no throughput gain (≤1.3×) and
    # becomes pathological at M≥16.  The sequential per-scenario path reuses
    # one cuDSS workspace and scales predictably.
    cudss_batch_solver = nothing
    cudss_batch_nzval  = nothing
    cudss_batch_rhs    = nothing
    cudss_batch_sol    = nothing

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
                             schur_batched_lus_v, nt_groups,
                             group_A_offsets, group_B_offsets, group_C_offsets,
                             schur_group_D_offsets_v,
                             schur_cluster_bus_reduced_v, schur_cluster_w_start_v,
                             cudss_S_solver_v, cudss_S_csr_v, csc_to_csr_perm_S_v,
                             cudss_S_rhs_v, cudss_S_sol_v,
                             schur_fwk_packed_v, schur_dz_gpu_v, schur_S_csc_buf_v,
                             schur_w_start_gpu_v, schur_vr_l_gpu_v, schur_vi_l_gpu_v,
                             cudss_solver, cudss_csr, csc_to_csr_perm,
                             cudss_rhs, cudss_sol,
                             cudss_batch_solver, cudss_batch_nzval,
                             cudss_batch_rhs, cudss_batch_sol,
                             nothing, nothing)
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
# GPU Schur extraction kernels
# -----------------------------------------------------------------------

# Gather A_k entries from J_nzval[m, :] into A_packed[:, :, batch_idx]
# for all scenarios m=1..M simultaneously.
# idx indexes into the flat A_jnz / A_local arrays (one entry per structural nz).
# M-loop inside kernel to avoid M × n_entries kernel launches.
@kernel function schur_gather_A_ka!(A_packed, J_nzval, A_jnz, A_local,
                                     @Const(wk_sq), @Const(M))
    idx = @index(Global)
    @inbounds begin
        jnz = A_jnz[idx]          # J nzval position (1-based)
        encoded = A_local[idx]     # (ki-1)*wk² + local_pos  (1-based)
        ki_m1 = div(encoded - 1, wk_sq)  # 0-based cluster-within-group
        local_pos = encoded - ki_m1 * wk_sq  # 1-based linear index in wk×wk
        for m in 1:M
            b = ki_m1 * M + m      # 1-based batch index
            val = J_nzval[m, jnz]
            A_packed[local_pos, b] = val  # A_packed viewed as (wk², batch)
        end
    end
end

# Gather B_k entries from J_nzval
@kernel function schur_gather_B_ka!(B_packed, J_nzval, B_jnz, B_local,
                                     @Const(wk_x2), @Const(M))
    idx = @index(Global)
    @inbounds begin
        jnz = B_jnz[idx]
        encoded = B_local[idx]     # (ki-1)*wk*2 + local_pos
        ki_m1 = div(encoded - 1, wk_x2)
        local_pos = encoded - ki_m1 * wk_x2
        for m in 1:M
            b = ki_m1 * M + m
            B_packed[local_pos, b] = J_nzval[m, jnz]
        end
    end
end

# Gather C_k entries from J_nzval
@kernel function schur_gather_C_ka!(C_packed, J_nzval, C_jnz, C_local,
                                     @Const(x2_wk), @Const(M))
    idx = @index(Global)
    @inbounds begin
        jnz = C_jnz[idx]
        encoded = C_local[idx]     # (ki-1)*2*wk + local_pos
        ki_m1 = div(encoded - 1, x2_wk)
        local_pos = encoded - ki_m1 * x2_wk
        for m in 1:M
            b = ki_m1 * M + m
            C_packed[local_pos, b] = J_nzval[m, jnz]
        end
    end
end

# Copy reduced block of J into S_nzval for one scenario
@kernel function schur_copy_reduced_ka!(S_nzval, J_nzval, J_to_S_jnz, J_to_S_snz,
                                         @Const(m))
    idx = @index(Global)
    @inbounds begin
        S_nzval[J_to_S_snz[idx]] = J_nzval[m, J_to_S_jnz[idx]]
    end
end

# Subtract D_k = C_k A_k⁻¹ B_k from S_nzval at precomputed positions.
# D_k is stored in B_packed after solving A_k⁻¹ B_k then multiplied by C_k.
# This kernel is called per cluster, writing 4 values (2×2 D_k).
# Actually, we'll compute D_k on GPU and subtract with a small kernel.
# Computes D_k = C_packed[:,:,b] * B_packed[:,:,b] (after solve, B holds A⁻¹B)
# and subtracts from S at 4 positions given in D_S_nzpos.
# One thread per cluster per scenario.
@kernel function schur_subtract_Dk_ka!(S_nzval, B_packed, C_packed, D_S_nzpos,
                                        @Const(wk), @Const(M), @Const(d_offset),
                                        @Const(ki_offset), @Const(m))
    ki_local = @index(Global)  # 1-based cluster within group
    @inbounds begin
        b = (ki_local - 1 + ki_offset) * M + m  # batch index (ki_offset=0 if starting from 1)
        d_base = d_offset + (ki_local - 1) * 4  # 0-based offset into D_S_nzpos
        # D_k[r,c] = Σ_j C[r,j,b] * B_solved[j,c,b]
        # C_packed is 2 × wk × batch, B_packed is wk × 2 × batch
        # After solve, B_packed[:, :, b] = A⁻¹ B
        for c in 1:2
            for r in 1:2
                dval = 0.0
                for j in 1:wk
                    # C_packed[r, j, b]: linear index = r + (j-1)*2 + (b-1)*2*wk
                    # But we store as CuArray{Float64,3} with dims (2, wk, batch)
                    dval += C_packed[r, j, b] * B_packed[j, c, b]
                end
                # Subtract from S: column-major order in D_S_nzpos is
                # (vr,vr), (vr,vi), (vi,vr), (vi,vi) — but actually stored as
                # (col=vr,row=vr), (col=vr,row=vi), (col=vi,row=vr), (col=vi,row=vi)
                # which maps to D[r,c] order: D[1,1], D[2,1], D[1,2], D[2,2]
                pos_idx = d_base + (c - 1) * 2 + r
                S_nzval[D_S_nzpos[pos_idx]] -= dval
            end
        end
    end
end

# -----------------------------------------------------------------------
# GPU Schur assembly
# -----------------------------------------------------------------------

"""
    _schur_extract_and_factor_gpu!(gbl)

Phase 1 of GPU Schur: extract A_k, B_k, C_k from J_nzval for ALL
scenarios, factor A_k, solve A_k⁻¹ B_k. All via single batched calls.
"""
function _schur_extract_and_factor_gpu!(gbl::GpuBatchedLayout)
    backend = CUDABackend()

    for (g, glu) in enumerate(gbl.schur_batched_lus)
        nc = glu.n_clusters
        wk = glu.w_k

        # Zero out packed buffers (stale values from previous iteration)
        fill!(glu.A_packed, 0.0)
        fill!(glu.B_packed, 0.0)
        fill!(glu.C_packed, 0.0)

        # Gather A_k (all clusters × all scenarios)
        n_a = gbl.schur_group_A_offsets[g + 1] - gbl.schur_group_A_offsets[g]
        if n_a > 0
            a_start = gbl.schur_group_A_offsets[g] + 1
            a_end = gbl.schur_group_A_offsets[g + 1]
            A_flat = reshape(glu.A_packed, wk * wk, nc * gbl.M)
            kernel_a = schur_gather_A_ka!(backend)
            kernel_a(A_flat, gbl.J_nzval,
                     view(gbl.cluster_A_jnz, a_start:a_end),
                     view(gbl.cluster_A_local, a_start:a_end),
                     Int32(wk * wk), Int32(gbl.M);
                     ndrange=n_a)
        end

        n_b = gbl.schur_group_B_offsets[g + 1] - gbl.schur_group_B_offsets[g]
        if n_b > 0
            b_start = gbl.schur_group_B_offsets[g] + 1
            b_end = gbl.schur_group_B_offsets[g + 1]
            B_flat = reshape(glu.B_packed, wk * 2, nc * gbl.M)
            kernel_b = schur_gather_B_ka!(backend)
            kernel_b(B_flat, gbl.J_nzval,
                     view(gbl.cluster_B_jnz, b_start:b_end),
                     view(gbl.cluster_B_local, b_start:b_end),
                     Int32(wk * 2), Int32(gbl.M);
                     ndrange=n_b)
        end

        n_c = gbl.schur_group_C_offsets[g + 1] - gbl.schur_group_C_offsets[g]
        if n_c > 0
            c_start = gbl.schur_group_C_offsets[g] + 1
            c_end = gbl.schur_group_C_offsets[g + 1]
            C_flat = reshape(glu.C_packed, 2 * wk, nc * gbl.M)
            kernel_c = schur_gather_C_ka!(backend)
            kernel_c(C_flat, gbl.J_nzval,
                     view(gbl.cluster_C_jnz, c_start:c_end),
                     view(gbl.cluster_C_local, c_start:c_end),
                     Int32(2 * wk), Int32(gbl.M);
                     ndrange=n_c)
        end

        KernelAbstractions.synchronize(backend)

        # Factor all A_k in one batched call
        gpu_batched_lu_factor!(glu)

        # Solve A_k⁻¹ B_k in one batched call (overwrites B_packed)
        gpu_batched_lu_solve!(glu)
    end

    return nothing
end

"""
    _schur_assemble_S_gpu!(gbl, m)

Phase 2 of GPU Schur: for scenario `m`, assemble S = D - Σ C_k A_k⁻¹ B_k.
Assumes `_schur_extract_and_factor_gpu!` already ran.
"""
function _schur_assemble_S_gpu!(gbl::GpuBatchedLayout, m::Int)
    backend = CUDABackend()

    # Copy reduced block: S_nzval[snz] = J_nzval[m, jnz]
    if gbl.J_nz_count_for_S > 0
        kernel_red = schur_copy_reduced_ka!(backend)
        kernel_red(gbl.S_nzval_gpu, gbl.J_nzval,
                   gbl.J_to_S_row, gbl.J_to_S_col, m;
                   ndrange=gbl.J_nz_count_for_S)
        KernelAbstractions.synchronize(backend)
    end

    # Subtract D_k for each group
    d_offset = 0
    for (g, glu) in enumerate(gbl.schur_batched_lus)
        nc = glu.n_clusters
        wk = glu.w_k
        kernel_d = schur_subtract_Dk_ka!(backend)
        kernel_d(gbl.S_nzval_gpu, glu.B_packed, glu.C_packed,
                 gbl.cluster_D_S_nzpos,
                 Int32(wk), Int32(gbl.M), Int32(d_offset),
                 Int32(0), Int32(m);
                 ndrange=nc)
        d_offset += 4 * nc
    end
    KernelAbstractions.synchronize(backend)

    return nothing
end

# -----------------------------------------------------------------------
# GPU Schur-complement Newton step with cuDSS on S
# -----------------------------------------------------------------------

# Gather reduced RHS (before negation): rhs_red[i] = f[m, reduced_idx[i]]
@kernel function schur_gather_rhs_ka!(rhs_red, f, reduced_idx, @Const(m))
    i = @index(Global)
    @inbounds rhs_red[i] = f[m, reduced_idx[i]]
end

# Scatter dz into z: z[m, reduced_idx[i]] += dv[i]
@kernel function schur_scatter_dv_ka!(z, dv, reduced_idx, @Const(m))
    i = @index(Global)
    @inbounds z[m, reduced_idx[i]] += dv[i]
end

# ── Fused kernels: one launch covers ALL clusters in a group ──

# Gather f_wk for ALL clusters in a group, ALL scenarios.
# ndrange = nc (one thread per cluster), M-loop inside.
@kernel function schur_gather_fwk_fused_ka!(fwk_packed, f, w_start_arr,
                                             @Const(wk), @Const(M))
    ki = @index(Global)
    @inbounds begin
        w_start = w_start_arr[ki]
        for m in 1:M
            b = (ki - 1) * M + m
            for j in 1:wk
                fwk_packed[j, 1, b] = f[m, w_start + j - 1]
            end
        end
    end
end

# Accumulate C_k * (A⁻¹ f_wk) into rhs_red for ALL clusters, one scenario m.
# ndrange = nc.
@kernel function schur_accum_Ck_fwk_fused_ka!(rhs_red, C_packed, fwk_solved,
                                                vr_l_arr, vi_l_arr,
                                                @Const(wk), @Const(M), @Const(m))
    ki = @index(Global)
    @inbounds begin
        batch = (ki - 1) * M + m
        vr_l = vr_l_arr[ki]
        vi_l = vi_l_arr[ki]
        c1 = 0.0; c2 = 0.0
        for j in 1:wk
            x = fwk_solved[j, 1, batch]
            c1 += C_packed[1, j, batch] * x
            c2 += C_packed[2, j, batch] * x
        end
        rhs_red[vr_l] -= c1
        rhs_red[vi_l] -= c2
    end
end

# Back-sub for ALL clusters in a group, one scenario m.
# ndrange = nc.
@kernel function schur_backsub_fused_ka!(z, fwk_solved, AinvBk, dv_sol,
                                          w_start_arr, vr_l_arr, vi_l_arr,
                                          @Const(wk), @Const(M), @Const(m))
    ki = @index(Global)
    @inbounds begin
        batch = (ki - 1) * M + m
        w_start = w_start_arr[ki]
        dv1 = dv_sol[vr_l_arr[ki]]
        dv2 = dv_sol[vi_l_arr[ki]]
        for j in 1:wk
            dw = -(fwk_solved[j, 1, batch] + AinvBk[j, 1, batch] * dv1 + AinvBk[j, 2, batch] * dv2)
            z[m, w_start + j - 1] += dw
        end
    end
end

"""
    _newton_step_schur_cudss_gpu!(gbl, dyn, L, dt; ...)

GPU Schur Newton step. All computation on GPU except per-scenario cuDSS
factorization of the small reduced system S.
"""
function _newton_step_schur_cudss_gpu!(
    gbl::GpuBatchedLayout,
    dyn::GradPower.PowerSystemDynamics,
    L::GradPower.SimulationLayout,
    dt::Float64;
    itermax::Int = 30,
    tol::Float64 = 1e-10,
)
    M       = gbl.M
    sys_dim = gbl.sys_dim
    n_red   = gbl.S_n
    backend = CUDABackend()

    f_flat = reshape(gbl.f, :)
    tol_l2 = tol * sqrt(Float64(sys_dim * M))

    for iter in 1:itermax
        # 1. Residual + backward Euler
        _beuler_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 2. Convergence check
        norm_f = CUDA.CUBLAS.nrm2(f_flat)
        norm_f < tol_l2 && return true

        # 3. Jacobian + backward Euler scaling
        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 4. Extract A/B/C, factor A, solve A⁻¹B (all clusters × all scenarios)
        _schur_extract_and_factor_gpu!(gbl)

        # 4b. Gather f_wk and solve A⁻¹ f_wk — one fused launch per group
        for (g, glu) in enumerate(gbl.schur_batched_lus)
            nc = glu.n_clusters; wk = glu.w_k
            fwk = gbl.schur_fwk_packed[g]
            kernel_fw = schur_gather_fwk_fused_ka!(backend)
            kernel_fw(fwk, gbl.f, gbl.schur_w_start_gpu[g], wk, M; ndrange=nc)
            KernelAbstractions.synchronize(backend)
            CUDA.CUBLAS.getrs_strided_batched!('N', glu.A_packed, fwk, glu.ipiv)
        end

        # 5. Per scenario: assemble S, build RHS, cuDSS solve, scatter+backsub
        for m in 1:M
            # 5a. Assemble S for scenario m
            _schur_assemble_S_gpu!(gbl, m)

            # 5b. Gather reduced RHS: rhs = f[m, reduced_idx]
            kernel_rhs = schur_gather_rhs_ka!(backend)
            kernel_rhs(gbl.cudss_S_rhs, gbl.f, gbl.reduced_idx_gpu, m;
                       ndrange=n_red)
            KernelAbstractions.synchronize(backend)

            # 5c. Subtract C_k (A⁻¹ f_wk) — one fused launch per group
            for (g, glu) in enumerate(gbl.schur_batched_lus)
                nc = glu.n_clusters; wk = glu.w_k
                kernel_acc = schur_accum_Ck_fwk_fused_ka!(backend)
                kernel_acc(gbl.cudss_S_rhs, glu.C_packed, gbl.schur_fwk_packed[g],
                           gbl.schur_vr_l_gpu[g], gbl.schur_vi_l_gpu[g],
                           wk, M, m; ndrange=nc)
            end
            KernelAbstractions.synchronize(backend)

            # Negate RHS: rhs = -(f_red - C_k A_k⁻¹ f_wk)
            CUDA.CUBLAS.scal!(n_red, -1.0, gbl.cudss_S_rhs)

            # 5d. CSC→CSR for S, cuDSS refactorize + solve
            copyto!(gbl.schur_S_csc_buf, gbl.S_nzval_gpu)
            kernel_csr = csc_to_csr_gather_ka!(backend)
            kernel_csr(gbl.cudss_S_csr.nzVal, gbl.schur_S_csc_buf, gbl.csc_to_csr_perm_S;
                       ndrange=gbl.S_nnz)
            KernelAbstractions.synchronize(backend)

            cudss("factorization", gbl.cudss_S_solver, gbl.cudss_S_sol, gbl.cudss_S_rhs)
            cudss("solve", gbl.cudss_S_solver, gbl.cudss_S_sol, gbl.cudss_S_rhs)

            # 5e. Scatter dv into z for reduced variables
            kernel_dv = schur_scatter_dv_ka!(backend)
            kernel_dv(gbl.z, gbl.cudss_S_sol, gbl.reduced_idx_gpu, m;
                      ndrange=n_red)

            # 5f. Back-sub — one fused launch per group
            for (g, glu) in enumerate(gbl.schur_batched_lus)
                nc = glu.n_clusters; wk = glu.w_k
                kernel_bs = schur_backsub_fused_ka!(backend)
                kernel_bs(gbl.z, gbl.schur_fwk_packed[g], glu.B_packed, gbl.cudss_S_sol,
                          gbl.schur_w_start_gpu[g], gbl.schur_vr_l_gpu[g], gbl.schur_vi_l_gpu[g],
                          wk, M, m; ndrange=nc)
            end
            KernelAbstractions.synchronize(backend)
        end
    end

    return false
end

# Copy z[m, :] into z_hist[:, m, step] on GPU (one kernel, no download)
@kernel function snapshot_z_ka!(z_hist, z, @Const(step), @Const(M))
    j = @index(Global)  # state index
    @inbounds for m in 1:M
        z_hist[j, m, step] = z[m, j]
    end
end

# CSC→CSR permute all M scenarios: J_nzval[M, nnz_csc] → batch_nzval[nnz_csr, M]
@kernel function csc_to_csr_batched_ka!(batch_nzval, J_nzval, perm, @Const(M))
    k = @index(Global)  # CSR nzval index
    @inbounds begin
        p = perm[k]  # corresponding CSC nzval index
        for m in 1:M
            batch_nzval[k, m] = J_nzval[m, p]
        end
    end
end

# Scatter batched solution into z: z[m, j] += sol[j, m]
@kernel function scatter_batched_sol_ka!(z, sol, @Const(M), @Const(n))
    j = @index(Global)
    @inbounds for m in 1:M
        z[m, j] += sol[j, m]
    end
end

# Build batched RHS: rhs[j, m] = -f[m, j]
@kernel function build_batched_rhs_ka!(rhs, f, @Const(M), @Const(n))
    j = @index(Global)
    @inbounds for m in 1:M
        rhs[j, m] = -f[m, j]
    end
end

# -----------------------------------------------------------------------
# GPU batched monolithic Newton step via cuDSS uniform batch
#
# All M scenarios factorized and solved in ONE cuDSS call.
# -----------------------------------------------------------------------

function _newton_step_cudss_batched_gpu!(
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

    f_flat = reshape(gbl.f, :)
    tol_l2 = tol * sqrt(Float64(sys_dim * M))

    # CudssMatrix wrappers (reuse across iterations)
    cb_rhs = CudssMatrix(Float64, sys_dim; nbatch=M)
    cb_sol = CudssMatrix(Float64, sys_dim; nbatch=M)

    first_factor = true
    for iter in 1:itermax
        # 1. Residual + backward Euler on GPU
        _beuler_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 2. Convergence check
        norm_f = CUDA.CUBLAS.nrm2(f_flat)
        norm_f < tol_l2 && return true

        # 3. Jacobian + backward Euler scaling on GPU
        _beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)

        # 4. Permute all M scenarios from CSC to CSR in one kernel
        kernel_perm = csc_to_csr_batched_ka!(backend)
        kernel_perm(gbl.cudss_batch_nzval, gbl.J_nzval, gbl.csc_to_csr_perm, M;
                    ndrange=nnz_J)
        KernelAbstractions.synchronize(backend)

        # 5. Build batched RHS = -f (transposed: sys_dim × M)
        kernel_rhs = build_batched_rhs_ka!(backend)
        kernel_rhs(gbl.cudss_batch_rhs, gbl.f, M, sys_dim; ndrange=sys_dim)
        KernelAbstractions.synchronize(backend)

        # 6. Update cuDSS with new nzvals and factorize all M systems at once
        cudss_update(gbl.cudss_batch_solver,
                     gbl.cudss_batch_rowPtr, gbl.cudss_batch_colVal,
                     gbl.cudss_batch_nzval)
        cudss_update(cb_rhs, gbl.cudss_batch_rhs)
        cudss_update(cb_sol, gbl.cudss_batch_sol)

        if first_factor
            cudss("factorization", gbl.cudss_batch_solver, cb_sol, cb_rhs)
            first_factor = false
        else
            cudss("refactorization", gbl.cudss_batch_solver, cb_sol, cb_rhs)
        end

        # 7. Solve all M systems at once
        cudss("solve", gbl.cudss_batch_solver, cb_sol, cb_rhs)

        # 8. Scatter solution: z[m, j] += sol[j, m]
        kernel_scatter = scatter_batched_sol_ka!(backend)
        kernel_scatter(gbl.z, gbl.cudss_batch_sol, M, sys_dim; ndrange=sys_dim)
        KernelAbstractions.synchronize(backend)
    end

    return false
end

# -----------------------------------------------------------------------
# GPU monolithic Newton step via cuDSS direct solve (per-scenario, legacy)
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

    first_factor = true
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

            # 4c. Numeric factorization (first) or refactorization (reuse symbolic)
            if first_factor
                cudss("factorization", gbl.cudss_solver, gbl.cudss_sol, gbl.cudss_rhs)
                first_factor = false
            else
                cudss("refactorization", gbl.cudss_solver, gbl.cudss_sol, gbl.cudss_rhs)
            end

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

    # GPU-resident trajectory buffer: z_hist[state, scenario, step]
    z_hist = CUDA.zeros(Float64, sys_dim, M, nsteps + 1)
    backend = CUDABackend()
    snap = snapshot_z_ka!(backend)
    snap(z_hist, gbl.z, 1, M; ndrange=sys_dim)
    KernelAbstractions.synchronize(backend)

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_cudss_gpu!(gbl, dyn, L, dt; tol = newton_tol)

        # Trajectory snapshot — GPU to GPU, no download
        snap(z_hist, gbl.z, k + 1, M; ndrange=sys_dim)
        KernelAbstractions.synchronize(backend)

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

    # Single bulk download at the end
    z_hist_cpu = Array(z_hist)
    trajs = [z_hist_cpu[:, m, :] for m in 1:M]

    return tvec, trajs
end

# -----------------------------------------------------------------------
# integrate_gpu_schur_cudss! — GPU Schur + cuDSS on S
# -----------------------------------------------------------------------

"""
    integrate_gpu_schur_cudss!(gbl, ps, tf; dt=1/120, newton_tol=1e-10)

GPU-resident batched integration using Schur-complement reduction.
A_k factored via cuBLAS batched, S solved via cuDSS.
"""
function integrate_gpu_schur_cudss!(
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

    # GPU-resident trajectory buffer: z_hist[state, scenario, step]
    z_hist = CUDA.zeros(Float64, sys_dim, M, nsteps + 1)
    backend = CUDABackend()
    snap = snapshot_z_ka!(backend)
    snap(z_hist, gbl.z, 1, M; ndrange=sys_dim)
    KernelAbstractions.synchronize(backend)

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_schur_cudss_gpu!(gbl, dyn, L, dt; tol=newton_tol)

        # Trajectory snapshot — GPU to GPU, no download
        snap(z_hist, gbl.z, k + 1, M; ndrange=sys_dim)
        KernelAbstractions.synchronize(backend)

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
            for (ei, ev) in enumerate(events)
                CUDA.@allowscalar gbl.event_status[ei] = ev.status
            end
            copyto!(gbl.zold, gbl.z)
            _newton_step_schur_cudss_gpu!(gbl, dyn, L, 0.0; tol=newton_tol)
        end
    end

    for event in events
        GradPower.deactivate!(event)
    end

    # Single bulk download at the end
    z_hist_cpu = Array(z_hist)
    trajs = [z_hist_cpu[:, m, :] for m in 1:M]

    return tvec, trajs
end

# -----------------------------------------------------------------------
# integrate_gpu_cudss_batched! — uses sequential per-scenario cuDSS (see note
# in GpuBatchedLayout constructor on why batched cuDSS is disabled)
# -----------------------------------------------------------------------

function integrate_gpu_cudss_batched!(
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

    z_hist = CUDA.zeros(Float64, sys_dim, M, nsteps + 1)
    backend = CUDABackend()
    snap = snapshot_z_ka!(backend)
    snap(z_hist, gbl.z, 1, M; ndrange=sys_dim)
    KernelAbstractions.synchronize(backend)

    copyto!(gbl.zold, gbl.z)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(gbl.zold, gbl.z)

        _newton_step_cudss_gpu!(gbl, dyn, L, dt; tol=newton_tol)

        snap(z_hist, gbl.z, k + 1, M; ndrange=sys_dim)
        KernelAbstractions.synchronize(backend)

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
            for (ei, ev) in enumerate(events)
                CUDA.@allowscalar gbl.event_status[ei] = ev.status
            end
            copyto!(gbl.zold, gbl.z)
            _newton_step_cudss_gpu!(gbl, dyn, L, 0.0; tol=newton_tol)
        end
    end

    for event in events
        GradPower.deactivate!(event)
    end

    z_hist_cpu = Array(z_hist)
    trajs = [z_hist_cpu[:, m, :] for m in 1:M]

    return tvec, trajs
end

end # module GradPowerCUDAExt
