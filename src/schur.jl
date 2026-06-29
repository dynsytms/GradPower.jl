# Schur complement reduction for the cluster-contiguous DAE system.
#
# After cluster-contiguous reordering, the backward-Euler Jacobian is:
#
#   J = [ A_nt  0    B_nt ]      A_nt = blkdiag of non-trivial clusters
#       [ 0     A_t  B_t  ]      A_t  = trivial clusters (singular)
#       [ C_nt  C_t  D    ]      D    = J_vv (voltage block)
#
# Non-trivial clusters have invertible A_k (Genrou + controllers).
# Trivial clusters (StaticGenerator PV/SLACK) have A_k = 0 because
# their alg equations depend only on voltages. These states are
# retained in the reduced system alongside the voltage variables.
#
# The reduced system has dimension n_red = n_trivial_states + 2*nbus
# and is solved with KLU.
#
# This file is included from src/GradPower.jl AFTER clusters.jl.

# -----------------------------------------------------------------------
# SchurWorkspace
# -----------------------------------------------------------------------

struct SchurWorkspace
    # Per non-trivial-cluster-group buffers.
    # nt_groups[g] = Vector{Int} of cluster indices (into ct.clusters)
    nt_groups::Vector{Vector{Int}}
    A_k_bufs::Vector{Vector{Matrix{Float64}}}
    B_k_bufs::Vector{Vector{Matrix{Float64}}}  # |w_k| x 2
    C_k_bufs::Vector{Vector{Matrix{Float64}}}  # 2 x |w_k|
    D_k_bufs::Vector{Vector{Matrix{Float64}}}  # 2 x 2
    lu_pivots::Vector{Vector{Vector{Int}}}
    tmp_w::Vector{Vector{Vector{Float64}}}

    # Reduced system S (sparse, dimension n_red x n_red)
    S::SparseMatrixCSC{Float64,Int}

    # Mapping between reduced and global indices.
    # reduced_idx[i] = global z-index for reduced variable i
    reduced_idx::Vector{Int}
    # global_to_reduced[g] = reduced index (0 if eliminated)
    global_to_reduced::Vector{Int}

    # KLU factorization of S
    S_fact::Base.RefValue{Any}

    # Reduced RHS vector
    rhs_red::Vector{Float64}

    # Full Newton correction
    dz::Vector{Float64}
end

function SchurWorkspace(ps::PowerSystem)
    ct = ps.dynamic.clusters::ClusterTable
    nbus = length(ps.buses)
    diff_dim = ps.dynamic.diff_dim
    alg_dim  = ps.dynamic.alg_dim
    net_ptr  = diff_dim + alg_dim
    sys_dim  = net_ptr + 2*nbus

    # Classify clusters
    nt_cluster_ids = Int[]   # non-trivial cluster indices
    for (ci, cl) in enumerate(ct.clusters)
        if !cl.trivial && cl.w_size > 0
            push!(nt_cluster_ids, ci)
        end
    end

    # Reduced index set: trivial cluster states + voltage states
    reduced_global = Int[]
    for (ci, cl) in enumerate(ct.clusters)
        if cl.trivial || cl.w_size == 0
            for j in cl.w_start:cl.w_end
                push!(reduced_global, j)
            end
        end
    end
    for i in (net_ptr + 1):sys_dim
        push!(reduced_global, i)
    end
    sort!(reduced_global)
    n_red = length(reduced_global)

    # Global -> reduced mapping
    g2r = zeros(Int, sys_dim)
    for (li, gi) in enumerate(reduced_global)
        g2r[gi] = li
    end

    # Build S sparsity from J's reduced subblock
    J_template = preallocate_jacobian(ps)
    # Fill with ones so klu gets a valid matrix
    fill!(J_template.nzval, 1.0)
    # Evaluate J at z0 for initial values
    # Actually, just build sparsity from the pattern
    I_s = Int[]; J_s = Int[]
    J_rows = rowvals(J_template)
    for (lj, gj) in enumerate(reduced_global)
        for nz in nzrange(J_template, gj)
            gi = J_rows[nz]
            li = g2r[gi]
            if li > 0
                push!(I_s, li)
                push!(J_s, lj)
            end
        end
    end
    # Ensure D_k diagonal positions for non-trivial clusters
    for ci in nt_cluster_ids
        cl = ct.clusters[ci]
        bus = cl.bus
        vr_g = net_ptr + 2*(bus - 1) + 1
        vi_g = vr_g + 1
        vr_l = g2r[vr_g]; vi_l = g2r[vi_g]
        for (r, c) in ((vr_l, vr_l), (vr_l, vi_l), (vi_l, vr_l), (vi_l, vi_l))
            push!(I_s, r); push!(J_s, c)
        end
    end
    # Ensure diagonal entries exist for all reduced positions (needed by
    # KLU for pivoting, even when the value is structurally zero, e.g.
    # PV StaticGenerator's alg state has J[ap,ap] = 0).
    for li in 1:n_red
        push!(I_s, li); push!(J_s, li)
    end
    S = sparse(I_s, J_s, zeros(Float64, length(I_s)), n_red, n_red)

    # Initial KLU symbolic factorization. Put nonzero values on the diagonal
    # to avoid a singular matrix error during symbolic analysis.
    fill!(S.nzval, 0.0)
    S_rows = rowvals(S)
    for col in 1:n_red
        for nz in nzrange(S, col)
            if S_rows[nz] == col
                S.nzval[nz] = 1.0
            end
        end
    end
    s_fact = Ref{Any}(klu(S))

    rhs_red = zeros(Float64, n_red)
    dz = zeros(Float64, sys_dim)

    # Group non-trivial clusters by w_size for uniform buffer allocation
    nt_groups = _group_by_wsize(ct, nt_cluster_ids)

    ng = length(nt_groups)
    A_k_bufs = Vector{Vector{Matrix{Float64}}}(undef, ng)
    B_k_bufs = Vector{Vector{Matrix{Float64}}}(undef, ng)
    C_k_bufs = Vector{Vector{Matrix{Float64}}}(undef, ng)
    D_k_bufs = Vector{Vector{Matrix{Float64}}}(undef, ng)
    lu_pivots = Vector{Vector{Vector{Int}}}(undef, ng)
    tmp_w = Vector{Vector{Vector{Float64}}}(undef, ng)

    for (g, group) in enumerate(nt_groups)
        nc = length(group)
        wk = ct.clusters[group[1]].w_size
        A_k_bufs[g] = [zeros(Float64, wk, wk) for _ in 1:nc]
        B_k_bufs[g] = [zeros(Float64, wk, 2)  for _ in 1:nc]
        C_k_bufs[g] = [zeros(Float64, 2, wk)  for _ in 1:nc]
        D_k_bufs[g] = [zeros(Float64, 2, 2)   for _ in 1:nc]
        lu_pivots[g] = [zeros(Int, wk) for _ in 1:nc]
        tmp_w[g] = [zeros(Float64, wk) for _ in 1:nc]
    end

    return SchurWorkspace(nt_groups, A_k_bufs, B_k_bufs, C_k_bufs, D_k_bufs,
                          lu_pivots, tmp_w, S, reduced_global, g2r,
                          s_fact, rhs_red, dz)
end

function _group_by_wsize(ct::ClusterTable, nt_ids::Vector{Int})
    isempty(nt_ids) && return Vector{Int}[]
    sorted = sort(nt_ids, by = ci -> ct.clusters[ci].w_size)
    groups = Vector{Int}[]
    cur = [sorted[1]]
    cur_wk = ct.clusters[sorted[1]].w_size
    for i in 2:length(sorted)
        ci = sorted[i]
        wk = ct.clusters[ci].w_size
        if wk == cur_wk
            push!(cur, ci)
        else
            push!(groups, cur)
            cur = [ci]
            cur_wk = wk
        end
    end
    push!(groups, cur)
    return groups
end

# -----------------------------------------------------------------------
# assemble_schur!
# -----------------------------------------------------------------------

function assemble_schur!(sw::SchurWorkspace, J::SparseMatrixCSC,
                          ct::ClusterTable, net_ptr::Int)
    # Step 1: copy reduced subblock of J into S
    _copy_reduced_block!(sw.S, J, sw.reduced_idx, sw.global_to_reduced)

    # Step 2: per non-trivial cluster — extract A_k, B_k, C_k, factor,
    # compute D_k = C_k A_k^{-1} B_k, subtract from S at bus diagonal
    for (g, group) in enumerate(sw.nt_groups)
        for (ki, ci) in enumerate(group)
            cl = ct.clusters[ci]
            wk = cl.w_size
            A = sw.A_k_bufs[g][ki]
            B = sw.B_k_bufs[g][ki]
            C = sw.C_k_bufs[g][ki]
            D = sw.D_k_bufs[g][ki]
            ipiv = sw.lu_pivots[g][ki]
            tmp = sw.tmp_w[g][ki]

            extract_Ak!(A, J, cl)
            extract_Bk_Ck!(B, C, J, cl, net_ptr)
            _lu_factor!(A, ipiv, wk)

            # D_k = C_k * A_k^{-1} * B_k (column by column)
            @inbounds for i in 1:wk; tmp[i] = B[i, 1]; end
            _lu_solve!(A, ipiv, tmp, wk)
            d11 = 0.0; d21 = 0.0
            @inbounds for i in 1:wk
                d11 += C[1, i] * tmp[i]
                d21 += C[2, i] * tmp[i]
            end

            @inbounds for i in 1:wk; tmp[i] = B[i, 2]; end
            _lu_solve!(A, ipiv, tmp, wk)
            d12 = 0.0; d22 = 0.0
            @inbounds for i in 1:wk
                d12 += C[1, i] * tmp[i]
                d22 += C[2, i] * tmp[i]
            end

            D[1,1] = d11; D[1,2] = d12
            D[2,1] = d21; D[2,2] = d22

            # Subtract D_k from S at bus diagonal (in reduced coordinates)
            bus = cl.bus
            vr_g = net_ptr + 2*(bus - 1) + 1
            vi_g = vr_g + 1
            vr_l = sw.global_to_reduced[vr_g]
            vi_l = sw.global_to_reduced[vi_g]
            _subtract_2x2_from_sparse!(sw.S, vr_l, vi_l, D)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------
# newton_step_schur!
# -----------------------------------------------------------------------

function newton_step_schur!(
    z0::AbstractVector,
    f0::AbstractVector,
    J0::SparseMatrixCSC,
    sw::SchurWorkspace,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    ps::PowerSystem,
    dt::Float64;
    itermax::Int=50,
    tol::Float64=1e-9,
    verbose::Bool=false,
    zwork::Union{Nothing,AbstractVector}=nothing,
    log::Union{Nothing,SolverLog}=nothing,
)
    z_buf = zwork === nothing ? similar(z0) : zwork
    copyto!(z_buf, z0)

    dyn  = ps.dynamic::PowerSystemDynamics
    net  = ps.network::Network
    L    = dyn.layout::SimulationLayout
    diff_dim = dyn.diff_dim
    ct   = dyn.clusters::ClusterTable
    ybus = net.ybus_real
    net_ptr = dyn.diff_dim + dyn.alg_dim
    n_red = length(sw.reduced_idx)

    success = false
    verbose && @printf("   Iter     Residual inf-norm\n")

    for iter = 1:itermax
        # 1. Evaluate backward-Euler residual
        if log !== nothing
            _t0 = time_ns()
            beuler_batched!(f0, z_buf, zold, u, p, dyn, ybus, L, diff_dim, dt, log)
            log.residual_ns += time_ns() - _t0
            log.residual_count += 1
        else
            beuler_batched!(f0, z_buf, zold, u, p, dyn, ybus, L, diff_dim, dt)
        end
        norm_f = norm(f0, Inf)
        verbose && @printf("   %2d     %.6e\n", iter-1, norm_f)
        if norm_f < tol
            success = true
            break
        end

        # 2. Evaluate backward-Euler Jacobian
        if log !== nothing
            _t0 = time_ns()
            beuler_jac_batched!(J0, z_buf, u, p, dyn, ybus, L, diff_dim, dt)
            log.jacobian_ns += time_ns() - _t0
            log.jacobian_count += 1
        else
            beuler_jac_batched!(J0, z_buf, u, p, dyn, ybus, L, diff_dim, dt)
        end

        # 3. Assemble Schur complement
        assemble_schur!(sw, J0, ct, net_ptr)

        # 4. Build reduced RHS: rhs_red[i] = f_red[i] - sum_k C_k A_k^{-1} f_wk
        # Start with the residual at reduced positions
        @inbounds for i in 1:n_red
            sw.rhs_red[i] = f0[sw.reduced_idx[i]]
        end

        # Subtract C_k * A_k^{-1} * f_wk for each non-trivial cluster
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size
                ws = cl.w_start
                bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1
                vi_g = vr_g + 1
                vr_l = sw.global_to_reduced[vr_g]
                vi_l = sw.global_to_reduced[vi_g]

                A = sw.A_k_bufs[g][ki]
                C = sw.C_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]
                tmp = sw.tmp_w[g][ki]

                # tmp = A_k^{-1} f_wk
                @inbounds for i in 1:wk; tmp[i] = f0[ws + i - 1]; end
                _lu_solve!(A, ipiv, tmp, wk)

                # rhs_red[vr/vi] -= C_k * tmp
                @inbounds for i in 1:wk
                    sw.rhs_red[vr_l] -= C[1, i] * tmp[i]
                    sw.rhs_red[vi_l] -= C[2, i] * tmp[i]
                end
            end
        end

        # 5. Solve S * d_red = -rhs_red
        @inbounds for i in 1:n_red; sw.rhs_red[i] = -sw.rhs_red[i]; end

        if log !== nothing
            _t0 = time_ns()
            if iter == 1
                sw.S_fact[] = klu(sw.S)
            else
                klu!(sw.S_fact[], sw.S)
            end
            log.lsolve_factor_ns += time_ns() - _t0
            log.lsolve_factor_count += 1

            _t0 = time_ns()
            ldiv!(sw.S_fact[], sw.rhs_red)
            log.lsolve_solve_ns += time_ns() - _t0
            log.lsolve_solve_count += 1
        else
            if iter == 1
                sw.S_fact[] = klu(sw.S)
            else
                klu!(sw.S_fact[], sw.S)
            end
            ldiv!(sw.S_fact[], sw.rhs_red)
        end
        # sw.rhs_red now holds d_red (Newton correction for reduced vars)

        # 6. Copy d_red into dz at reduced positions
        @inbounds for i in 1:n_red
            sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
        end

        # 7. Back-substitute for non-trivial clusters:
        # dw_k = -A_k^{-1}(f_wk + B_k * dv_{bus(k)})
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size
                ws = cl.w_start
                bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1
                vi_g = vr_g + 1

                A = sw.A_k_bufs[g][ki]
                B = sw.B_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]
                tmp = sw.tmp_w[g][ki]

                dv1 = sw.dz[vr_g]
                dv2 = sw.dz[vi_g]

                @inbounds for i in 1:wk
                    tmp[i] = f0[ws + i - 1] + B[i, 1] * dv1 + B[i, 2] * dv2
                end
                _lu_solve!(A, ipiv, tmp, wk)

                @inbounds for i in 1:wk
                    sw.dz[ws + i - 1] = -tmp[i]
                end
            end
        end

        # 8. Apply Newton update: z_new = z + delta
        @inbounds for k in eachindex(z_buf)
            z_buf[k] += sw.dz[k]
        end
    end

    z0 .= z_buf
    return success
end

# -----------------------------------------------------------------------
# Dense LU (pure Julia, zero allocation)
#
# For the small cluster sizes (6-20 states) in power-system DAE models,
# pure-Julia LU is equally fast as LAPACK and avoids the 16-byte Ref
# heap allocation per ccall that LAPACK wrappers produce.
# -----------------------------------------------------------------------

function _lu_factor!(A::Matrix{Float64}, ipiv::Vector, n::Int)
    @inbounds for k in 1:n
        # Partial pivoting: find max |A[i,k]| for i >= k
        maxval = abs(A[k, k])
        maxrow = k
        for i in (k+1):n
            v = abs(A[i, k])
            if v > maxval
                maxval = v
                maxrow = i
            end
        end
        ipiv[k] = maxrow
        # Swap rows k and maxrow
        if maxrow != k
            for j in 1:n
                A[k, j], A[maxrow, j] = A[maxrow, j], A[k, j]
            end
        end
        # Elimination
        akk = A[k, k]
        if akk != 0.0
            inv_akk = 1.0 / akk
            for i in (k+1):n
                A[i, k] *= inv_akk
                for j in (k+1):n
                    A[i, j] -= A[i, k] * A[k, j]
                end
            end
        end
    end
    return nothing
end

function _lu_solve!(A::Matrix{Float64}, ipiv::Vector,
                     b::Vector{Float64}, n::Int)
    # Apply row permutations
    @inbounds for k in 1:n
        p = Int(ipiv[k])
        if p != k
            b[k], b[p] = b[p], b[k]
        end
    end
    # Forward substitution (L * y = Pb)
    @inbounds for k in 1:n
        for i in (k+1):n
            b[i] -= A[i, k] * b[k]
        end
    end
    # Backward substitution (U * x = y)
    @inbounds for k in n:-1:1
        b[k] /= A[k, k]
        for i in 1:(k-1)
            b[i] -= A[i, k] * b[k]
        end
    end
    return nothing
end

# -----------------------------------------------------------------------
# Sparse helpers
# -----------------------------------------------------------------------

function _copy_reduced_block!(S::SparseMatrixCSC, J::SparseMatrixCSC,
                                reduced_idx::Vector{Int},
                                g2r::Vector{Int})
    S_rows = rowvals(S)
    S_vals = nonzeros(S)
    J_rows = rowvals(J)
    J_vals = nonzeros(J)
    n_red = size(S, 1)

    # Zero S first
    fill!(S_vals, 0.0)

    @inbounds for lj in 1:n_red
        gj = reduced_idx[lj]
        for j_nz in nzrange(J, gj)
            gi = J_rows[j_nz]
            li = g2r[gi]
            li == 0 && continue
            # Find li in S's column lj
            for s_nz in nzrange(S, lj)
                if S_rows[s_nz] == li
                    S_vals[s_nz] = J_vals[j_nz]
                    break
                end
            end
        end
    end
    return nothing
end

function _subtract_2x2_from_sparse!(S::SparseMatrixCSC, vr::Int, vi::Int,
                                      D::Matrix{Float64})
    rows = rowvals(S)
    vals = nonzeros(S)
    @inbounds for nz in nzrange(S, vr)
        r = rows[nz]
        if r == vr; vals[nz] -= D[1, 1]; end
        if r == vi; vals[nz] -= D[2, 1]; end
    end
    @inbounds for nz in nzrange(S, vi)
        r = rows[nz]
        if r == vr; vals[nz] -= D[1, 2]; end
        if r == vi; vals[nz] -= D[2, 2]; end
    end
    return nothing
end

# -----------------------------------------------------------------------
# Y-Preconditioner for GMRES on the Schur system
# -----------------------------------------------------------------------

"""
    YPreconditioner

Right-preconditioner wrapping a KLU factorization of the base admittance
matrix (Y_base), projected into the reduced Schur coordinate system.

The preconditioner matrix P has the same sparsity as the Schur complement
S: its voltage-voltage block is −ybus_real (the constant network admittance
contribution), and trivial-cluster rows/columns carry the J_reduced entries
evaluated at z₀. P is factored once with KLU and reused across all Newton
iterations and time steps (recomputed only on topology change).

Implements `LinearAlgebra.ldiv!(y, P, x)` for Krylov.jl compatibility.
"""
struct YPreconditioner
    P::SparseMatrixCSC{Float64,Int}
    fact::Base.RefValue{Any}
end

function LinearAlgebra.ldiv!(y::AbstractVector, prec::YPreconditioner, x::AbstractVector)
    copyto!(y, x)
    ldiv!(prec.fact[], y)
    return y
end

"""
    build_y_preconditioner(sw, ps, dt) -> YPreconditioner

Build the Y-preconditioner from the network admittance matrix only.
The preconditioner P contains:
  - voltage-voltage block: −ybus (network admittance only, no device
    stator injection, no zipload, no fault shunts)
  - trivial cluster rows: their Jacobian entries from J (StaticGenerator
    algebraic equations that can't be Schur-eliminated)
  - backward-Euler diagonal scaling (1/dt on diff rows)

This ensures P ≈ Y so that P⁻¹·S ≈ I − small device perturbations,
giving fast GMRES convergence independent of system size.
"""
function build_y_preconditioner(sw::SchurWorkspace, ps::PowerSystem,
                                 z0::AbstractVector, p_vec::AbstractVector,
                                 dt::Float64)
    P = copy(sw.S)
    _fill_y_preconditioner!(P, sw, ps, z0, p_vec, dt)
    fact = Ref{Any}(klu(P))
    return YPreconditioner(P, fact)
end

function _fill_y_preconditioner!(P::SparseMatrixCSC, sw::SchurWorkspace,
                                  ps::PowerSystem, z0::AbstractVector,
                                  p_vec::AbstractVector, dt::Float64)
    dyn = ps.dynamic::PowerSystemDynamics
    ct  = dyn.clusters::ClusterTable
    ybus = ps.network.ybus_real
    diff_dim = dyn.diff_dim
    net_ptr  = diff_dim + dyn.alg_dim
    nbus = length(ps.buses)
    g2r = sw.global_to_reduced

    P_rows = rowvals(P)
    P_vals = nonzeros(P)
    fill!(P_vals, 0.0)

    # 1. Copy −ybus into the voltage-voltage subblock of P
    ybus_rows = rowvals(ybus)
    ybus_vals = nonzeros(ybus)
    for col_y in 1:size(ybus, 2)
        gcol = net_ptr + col_y
        lcol = g2r[gcol]
        lcol == 0 && continue
        for nz_y in nzrange(ybus, col_y)
            grow = ybus_rows[nz_y] + net_ptr
            lrow = g2r[grow]
            lrow == 0 && continue
            val = -ybus_vals[nz_y]
            for nz_p in nzrange(P, lcol)
                if P_rows[nz_p] == lrow
                    P_vals[nz_p] = val
                    break
                end
            end
        end
    end

    # 2. Trivial cluster entries: build a minimal Jacobian with only
    #    current_injection + static_gen + backward-Euler scaling, then
    #    extract the trivial rows/cols into P.
    has_trivial = false
    for (ci, cl) in enumerate(ct.clusters)
        if cl.trivial || cl.w_size == 0
            has_trivial = true
            break
        end
    end

    if has_trivial
        J_min = preallocate_jacobian(ps)
        fill!(J_min.nzval, 0.0)
        current_injection_jacobian!(J_min, ybus, net_ptr)
        L = dyn.layout::SimulationLayout
        static_gen_jacobian_batch!(J_min, z0, p_vec, L.static_gen)
        isd = dyn.is_diff
        if isd !== nothing
            _jacobian_beuler_indices!(J_min, isd, dt)
        else
            _jacobian_beuler!(J_min, diff_dim, dt)
        end

        J_rows_min = rowvals(J_min)
        J_vals_min = nonzeros(J_min)
        for (ci, cl) in enumerate(ct.clusters)
            (cl.trivial || cl.w_size == 0) || continue
            # Copy trivial cluster columns → all reduced rows
            for j in cl.w_start:cl.w_end
                lj = g2r[j]
                lj == 0 && continue
                for nz_j in nzrange(J_min, j)
                    gi = J_rows_min[nz_j]
                    li = g2r[gi]
                    li == 0 && continue
                    for nz_p in nzrange(P, lj)
                        if P_rows[nz_p] == li
                            P_vals[nz_p] = J_vals_min[nz_j]
                            break
                        end
                    end
                end
            end
            # Copy voltage columns → trivial cluster rows
            for vi in (net_ptr+1):(net_ptr+2*nbus)
                lvi = g2r[vi]
                lvi == 0 && continue
                for nz_j in nzrange(J_min, vi)
                    gi = J_rows_min[nz_j]
                    li = g2r[gi]
                    li == 0 && continue
                    (gi >= cl.w_start && gi <= cl.w_end) || continue
                    for nz_p in nzrange(P, lvi)
                        if P_rows[nz_p] == li
                            P_vals[nz_p] = J_vals_min[nz_j]
                            break
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
    refresh_y_preconditioner!(prec, sw, ps, z0, p_vec, dt)

Recompute and refactor the Y-preconditioner after a topology change
(e.g. TripLineEvent modifies ybus_real). Rebuilds from Y-only.
"""
function refresh_y_preconditioner!(prec::YPreconditioner, sw::SchurWorkspace,
                                    ps::PowerSystem, z0::AbstractVector,
                                    p_vec::AbstractVector, dt::Float64)
    _fill_y_preconditioner!(prec.P, sw, ps, z0, p_vec, dt)
    klu!(prec.fact[], prec.P)
    return nothing
end

# -----------------------------------------------------------------------
# GmresSchurWorkspace
# -----------------------------------------------------------------------

"""
    GmresSchurWorkspace

Extended SchurWorkspace with Krylov.jl GMRES workspace and Y-preconditioner.
"""
struct GmresSchurWorkspace
    sw::SchurWorkspace
    prec::YPreconditioner
    gmres_ws::GmresWorkspace{Float64,Float64,Vector{Float64}}
    gmres_atol::Float64
    gmres_rtol::Float64
    gmres_maxiter::Int
    gmres_restart::Int
end

function GmresSchurWorkspace(ps::PowerSystem, z0::AbstractVector,
                              p_vec::AbstractVector, dt::Float64;
                              gmres_atol::Float64=1e-12,
                              gmres_rtol::Float64=1e-10,
                              gmres_maxiter::Int=100,
                              gmres_restart::Int=80)
    sw = SchurWorkspace(ps)
    n_red = length(sw.reduced_idx)

    prec = build_y_preconditioner(sw, ps, z0, p_vec, dt)

    # Preallocate Krylov workspace
    gmres_ws = GmresWorkspace(n_red, n_red, Vector{Float64}; memory=gmres_restart)

    return GmresSchurWorkspace(sw, prec, gmres_ws, gmres_atol, gmres_rtol,
                                gmres_maxiter, gmres_restart)
end

# -----------------------------------------------------------------------
# newton_step_schur_gmres!
# -----------------------------------------------------------------------

function newton_step_schur_gmres!(
    z0::AbstractVector,
    f0::AbstractVector,
    J0::SparseMatrixCSC,
    gsw::GmresSchurWorkspace,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    ps::PowerSystem,
    dt::Float64;
    itermax::Int=50,
    tol::Float64=1e-9,
    verbose::Bool=false,
    zwork::Union{Nothing,AbstractVector}=nothing,
    log::Union{Nothing,SolverLog}=nothing,
)
    sw = gsw.sw
    z_buf = zwork === nothing ? similar(z0) : zwork
    copyto!(z_buf, z0)

    dyn  = ps.dynamic::PowerSystemDynamics
    net  = ps.network::Network
    L    = dyn.layout::SimulationLayout
    diff_dim = dyn.diff_dim
    ct   = dyn.clusters::ClusterTable
    ybus = net.ybus_real
    net_ptr = dyn.diff_dim + dyn.alg_dim
    n_red = length(sw.reduced_idx)

    success = false
    verbose && @printf("   Iter     Residual inf-norm   GMRES iters\n")

    for iter = 1:itermax
        # 1. Evaluate backward-Euler residual
        if log !== nothing
            _t0 = time_ns()
            beuler_batched!(f0, z_buf, zold, u, p, dyn, ybus, L, diff_dim, dt, log)
            log.residual_ns += time_ns() - _t0
            log.residual_count += 1
        else
            beuler_batched!(f0, z_buf, zold, u, p, dyn, ybus, L, diff_dim, dt)
        end
        norm_f = norm(f0, Inf)
        if norm_f < tol
            verbose && @printf("   %2d     %.6e\n", iter-1, norm_f)
            success = true
            break
        end

        # 2. Evaluate backward-Euler Jacobian
        if log !== nothing
            _t0 = time_ns()
            beuler_jac_batched!(J0, z_buf, u, p, dyn, ybus, L, diff_dim, dt)
            log.jacobian_ns += time_ns() - _t0
            log.jacobian_count += 1
        else
            beuler_jac_batched!(J0, z_buf, u, p, dyn, ybus, L, diff_dim, dt)
        end

        # 3. Assemble Schur complement
        assemble_schur!(sw, J0, ct, net_ptr)

        # 4. Build reduced RHS: rhs_red[i] = f_red[i] - sum_k C_k A_k^{-1} f_wk
        @inbounds for i in 1:n_red
            sw.rhs_red[i] = f0[sw.reduced_idx[i]]
        end

        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size
                ws = cl.w_start
                bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1
                vi_g = vr_g + 1
                vr_l = sw.global_to_reduced[vr_g]
                vi_l = sw.global_to_reduced[vi_g]

                A = sw.A_k_bufs[g][ki]
                C = sw.C_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]
                tmp = sw.tmp_w[g][ki]

                @inbounds for i in 1:wk; tmp[i] = f0[ws + i - 1]; end
                _lu_solve!(A, ipiv, tmp, wk)

                @inbounds for i in 1:wk
                    sw.rhs_red[vr_l] -= C[1, i] * tmp[i]
                    sw.rhs_red[vi_l] -= C[2, i] * tmp[i]
                end
            end
        end

        # 5. Solve S * d_red = -rhs_red using GMRES with Y-preconditioner
        @inbounds for i in 1:n_red; sw.rhs_red[i] = -sw.rhs_red[i]; end

        gmres!(gsw.gmres_ws, sw.S, sw.rhs_red;
               N=gsw.prec, ldiv=true,
               atol=gsw.gmres_atol, rtol=gsw.gmres_rtol,
               itmax=gsw.gmres_maxiter,
               restart=true,
               verbose=0)

        gmres_niter = gsw.gmres_ws.stats.niter
        verbose && @printf("   %2d     %.6e   %d\n", iter-1, norm_f, gmres_niter)

        # Record GMRES iteration count
        if log !== nothing
            push!(log.gmres_iters, gmres_niter)
        end

        # Copy solution into rhs_red (gmres_ws.x holds the solution)
        copyto!(sw.rhs_red, gsw.gmres_ws.x)

        # 6. Copy d_red into dz at reduced positions
        @inbounds for i in 1:n_red
            sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
        end

        # 7. Back-substitute for non-trivial clusters
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size
                ws = cl.w_start
                bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1
                vi_g = vr_g + 1

                A = sw.A_k_bufs[g][ki]
                B = sw.B_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]
                tmp = sw.tmp_w[g][ki]

                dv1 = sw.dz[vr_g]
                dv2 = sw.dz[vi_g]

                @inbounds for i in 1:wk
                    tmp[i] = f0[ws + i - 1] + B[i, 1] * dv1 + B[i, 2] * dv2
                end
                _lu_solve!(A, ipiv, tmp, wk)

                @inbounds for i in 1:wk
                    sw.dz[ws + i - 1] = -tmp[i]
                end
            end
        end

        # 8. Apply Newton update
        @inbounds for k in eachindex(z_buf)
            z_buf[k] += sw.dz[k]
        end
    end

    z0 .= z_buf
    return success
end
