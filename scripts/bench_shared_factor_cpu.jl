#!/usr/bin/env julia
#
# Phase 1+2: CPU prototype — shared-factorization modified Newton
# with Woodbury rank-2 correction for bus faults.
#
# Validates convergence on CPU before GPU port.

using GradPower, SparseArrays, LinearAlgebra, Printf, KLU

# ═══════════════════════════════════════════════════════════════════════
# Woodbury-corrected solve: P_m⁻¹ b = P⁻¹b - W H⁻¹ Uᵀ P⁻¹b
# ═══════════════════════════════════════════════════════════════════════

struct WoodburyCorrection
    W::Matrix{Float64}          # (n_red, 2) — P⁻¹ U
    H_inv::Matrix{Float64}     # (2, 2) — (K⁻¹ + Uᵀ W)⁻¹
    vr_l::Int                  # reduced index for V_real at fault bus
    vi_l::Int                  # reduced index for V_imag at fault bus
    active::Ref{Bool}
end

"""
Precompute Woodbury columns for a bus fault at `fault_bus` with impedance `rfault`.

ΔY adds (1/rfault) to the Ybus diagonal at the fault bus.
In the 2n_b real-valued Schur complement, this is a rank-2 update:
  ΔS = -(1/rfault) * [e_vr  e_vi] * I₂ * [e_vr  e_vi]ᵀ
where e_vr, e_vi are unit vectors at the reduced indices for the bus.

P_m = P + ΔS. Woodbury: P_m⁻¹ = P⁻¹ - W H⁻¹ Uᵀ P⁻¹
  with U = [e_vr  e_vi], K = -(1/rfault) I₂
  W = P⁻¹ U  (2 sparse solves)
  H = K⁻¹ + Uᵀ W = -rfault I₂ + Uᵀ W
"""
function precompute_woodbury(P_fact, n_red::Int, fault_bus::Int,
                              rfault::Float64, net_ptr::Int,
                              g2r::Vector{Int})
    vr_g = net_ptr + 2*(fault_bus - 1) + 1
    vi_g = vr_g + 1
    vr_l = g2r[vr_g]
    vi_l = g2r[vi_g]

    # U = (n_red × 2), U[:, 1] = e_vr, U[:, 2] = e_vi
    # K = -(1/rfault) I₂
    # W = P⁻¹ U
    W = zeros(Float64, n_red, 2)

    # Solve P w₁ = e_vr
    rhs1 = zeros(Float64, n_red)
    rhs1[vr_l] = 1.0
    ldiv!(P_fact, rhs1)
    W[:, 1] .= rhs1

    # Solve P w₂ = e_vi
    rhs2 = zeros(Float64, n_red)
    rhs2[vi_l] = 1.0
    ldiv!(P_fact, rhs2)
    W[:, 2] .= rhs2

    # H = K⁻¹ + Uᵀ W = -rfault I₂ + [W[vr_l,:]; W[vi_l,:]]
    H = zeros(Float64, 2, 2)
    H[1, 1] = -rfault + W[vr_l, 1]
    H[1, 2] =           W[vr_l, 2]
    H[2, 1] =           W[vi_l, 1]
    H[2, 2] = -rfault + W[vi_l, 2]

    H_inv = inv(H)

    # Scale W by -(1/rfault) to absorb K into the correction:
    # The update is ΔS = -(1/rfault) * U * Uᵀ (with K = -(1/rfault)I)
    # But Woodbury formula: P_m⁻¹ = P⁻¹ - P⁻¹ U (K⁻¹ + Uᵀ P⁻¹ U)⁻¹ Uᵀ P⁻¹
    # We already have W = P⁻¹ U and H_inv = (K⁻¹ + Uᵀ W)⁻¹
    # So the correction is: x = y - W * H_inv * (Uᵀ y)
    # where y = P⁻¹ b

    return WoodburyCorrection(W, H_inv, vr_l, vi_l, Ref(false))
end

"""
Apply Woodbury correction in-place: x ← x - W H⁻¹ (Uᵀ x)
where x already holds P⁻¹ b.
"""
function apply_woodbury!(x::AbstractVector, wc::WoodburyCorrection)
    wc.active[] || return nothing

    # t = Uᵀ x  (2-vector: [x[vr_l], x[vi_l]])
    t1 = x[wc.vr_l]
    t2 = x[wc.vi_l]

    # s = H⁻¹ t
    s1 = wc.H_inv[1, 1] * t1 + wc.H_inv[1, 2] * t2
    s2 = wc.H_inv[2, 1] * t1 + wc.H_inv[2, 2] * t2

    # x -= W * s
    @inbounds for i in eachindex(x)
        x[i] -= wc.W[i, 1] * s1 + wc.W[i, 2] * s2
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════
# Modified Newton with frozen P + Woodbury correction
# ═══════════════════════════════════════════════════════════════════════

function newton_step_shared_factor!(
    z::AbstractVector,
    f::AbstractVector,
    J::SparseMatrixCSC,
    sw::GradPower.SchurWorkspace,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    ps::GradPower.PowerSystem,
    dt::Float64,
    P_fact,
    wc::WoodburyCorrection;
    itermax::Int=50,
    tol::Float64=1e-10,
)
    dyn  = ps.dynamic::GradPower.PowerSystemDynamics
    net  = ps.network::GradPower.Network
    L    = dyn.layout::GradPower.SimulationLayout
    ct   = dyn.clusters::GradPower.ClusterTable
    ybus = net.ybus_real
    diff_dim = dyn.diff_dim
    net_ptr = dyn.diff_dim + dyn.alg_dim
    n_red = length(sw.reduced_idx)

    for iter in 1:itermax
        GradPower.beuler_batched!(f, z, zold, u, p, dyn, ybus, L, diff_dim, dt)
        norm_f = norm(f, Inf)
        norm_f < tol && return (true, iter - 1)

        GradPower.beuler_jac_batched!(J, z, u, p, dyn, ybus, L, diff_dim, dt)
        GradPower.assemble_schur!(sw, J, ct, net_ptr)

        @inbounds for i in 1:n_red
            sw.rhs_red[i] = f[sw.reduced_idx[i]]
        end
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1; vi_g = vr_g + 1
                vr_l = sw.global_to_reduced[vr_g]
                vi_l = sw.global_to_reduced[vi_g]
                A = sw.A_k_bufs[g][ki]; C = sw.C_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
                @inbounds for i in 1:wk; tmp[i] = f[ws + i - 1]; end
                GradPower._lu_solve!(A, ipiv, tmp, wk)
                @inbounds for i in 1:wk
                    sw.rhs_red[vr_l] -= C[1, i] * tmp[i]
                    sw.rhs_red[vi_l] -= C[2, i] * tmp[i]
                end
            end
        end

        @inbounds for i in 1:n_red; sw.rhs_red[i] = -sw.rhs_red[i]; end

        # Solve P * d_red = rhs (frozen factorization)
        # Need a fresh copy of rhs_red since ldiv! overwrites
        ldiv!(P_fact, sw.rhs_red)

        # Apply Woodbury correction: d_red ← P_m⁻¹ rhs
        apply_woodbury!(sw.rhs_red, wc)

        fill!(sw.dz, 0.0)
        @inbounds for i in 1:n_red
            sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
        end

        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                vr_g = net_ptr + 2*(bus - 1) + 1; vi_g = vr_g + 1
                A = sw.A_k_bufs[g][ki]; B = sw.B_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
                dv1 = sw.dz[vr_g]; dv2 = sw.dz[vi_g]
                @inbounds for i in 1:wk
                    tmp[i] = f[ws + i - 1] + B[i, 1] * dv1 + B[i, 2] * dv2
                end
                GradPower._lu_solve!(A, ipiv, tmp, wk)
                @inbounds for i in 1:wk
                    sw.dz[ws + i - 1] = -tmp[i]
                end
            end
        end

        @inbounds for k in eachindex(z)
            z[k] += sw.dz[k]
        end
    end
    return (false, itermax)
end

function build_reference_P(ps::GradPower.PowerSystem, z_ref::Vector{Float64},
                            u::Vector{Float64}, p::Vector{Float64}, dt::Float64)
    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout
    ct = dyn.clusters::GradPower.ClusterTable
    diff_dim = dyn.diff_dim
    net_ptr = dyn.diff_dim + dyn.alg_dim
    ybus = ps.network.ybus_real

    J = GradPower.preallocate_jacobian(ps)
    GradPower.beuler_jac_batched!(J, z_ref, u, p, dyn, ybus, L, diff_dim, dt)

    sw = GradPower.SchurWorkspace(ps)
    GradPower.assemble_schur!(sw, J, ct, net_ptr)

    P_fact = klu(sw.S)
    return P_fact, sw
end

# ═══════════════════════════════════════════════════════════════════════
# Exact Schur Newton (for iteration counting)
# ═══════════════════════════════════════════════════════════════════════

function newton_step_exact_schur_counted!(
    z::AbstractVector,
    f::AbstractVector,
    J::SparseMatrixCSC,
    sw::GradPower.SchurWorkspace,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    ps::GradPower.PowerSystem,
    dt::Float64;
    itermax::Int=50,
    tol::Float64=1e-10,
)
    dyn  = ps.dynamic::GradPower.PowerSystemDynamics
    ct   = dyn.clusters::GradPower.ClusterTable
    ybus = ps.network.ybus_real
    L    = dyn.layout::GradPower.SimulationLayout
    diff_dim = dyn.diff_dim
    net_ptr = dyn.diff_dim + dyn.alg_dim
    n_red = length(sw.reduced_idx)

    for iter in 1:itermax
        GradPower.beuler_batched!(f, z, zold, u, p, dyn, ybus, L, diff_dim, dt)
        norm_f = norm(f, Inf)
        norm_f < tol && return (true, iter - 1)

        GradPower.beuler_jac_batched!(J, z, u, p, dyn, ybus, L, diff_dim, dt)
        GradPower.assemble_schur!(sw, J, ct, net_ptr)

        @inbounds for i in 1:n_red
            sw.rhs_red[i] = f[sw.reduced_idx[i]]
        end
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                vr_g = net_ptr + 2*(bus-1)+1; vi_g = vr_g+1
                vr_l = sw.global_to_reduced[vr_g]; vi_l = sw.global_to_reduced[vi_g]
                A = sw.A_k_bufs[g][ki]; C = sw.C_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
                @inbounds for i in 1:wk; tmp[i] = f[ws+i-1]; end
                GradPower._lu_solve!(A, ipiv, tmp, wk)
                @inbounds for i in 1:wk
                    sw.rhs_red[vr_l] -= C[1,i]*tmp[i]
                    sw.rhs_red[vi_l] -= C[2,i]*tmp[i]
                end
            end
        end

        @inbounds for i in 1:n_red; sw.rhs_red[i] = -sw.rhs_red[i]; end
        if iter == 1
            sw.S_fact[] = klu(sw.S)
        else
            klu!(sw.S_fact[], sw.S)
        end
        ldiv!(sw.S_fact[], sw.rhs_red)

        fill!(sw.dz, 0.0)
        @inbounds for i in 1:n_red
            sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
        end
        for (g, group) in enumerate(sw.nt_groups)
            for (ki, ci) in enumerate(group)
                cl = ct.clusters[ci]
                wk = cl.w_size; ws = cl.w_start; bus = cl.bus
                vr_g = net_ptr + 2*(bus-1)+1; vi_g = vr_g+1
                A = sw.A_k_bufs[g][ki]; B = sw.B_k_bufs[g][ki]
                ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
                dv1 = sw.dz[vr_g]; dv2 = sw.dz[vi_g]
                @inbounds for i in 1:wk
                    tmp[i] = f[ws+i-1] + B[i,1]*dv1 + B[i,2]*dv2
                end
                GradPower._lu_solve!(A, ipiv, tmp, wk)
                @inbounds for i in 1:wk; sw.dz[ws+i-1] = -tmp[i]; end
            end
        end

        @inbounds for k in eachindex(z)
            z[k] += sw.dz[k]
        end
    end
    return (false, itermax)
end

# ═══════════════════════════════════════════════════════════════════════
# Test harness
# ═══════════════════════════════════════════════════════════════════════

_mean(x) = isempty(x) ? 0.0 : sum(x) / length(x)
function _median(x)
    isempty(x) && return 0.0
    s = sort(x); n = length(s)
    isodd(n) ? Float64(s[(n+1)÷2]) : (s[n÷2] + s[n÷2+1]) / 2.0
end

function run_comparison(case_raw, case_dyr, case_label;
                        fault_bus=1, rfault=0.02, ton=0.1, toff=0.2,
                        tf=1.0, dt=1.0/120.0, tol=1e-10)
    println("=" ^ 70)
    println("  $case_label")
    println("=" ^ 70)
    flush(stdout)

    ps = GradPower.from_psse(case_raw, case_dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout
    ct = dyn.clusters::GradPower.ClusterTable
    net = ps.network::GradPower.Network
    diff_dim = dyn.diff_dim
    net_ptr = dyn.diff_dim + dyn.alg_dim
    sys_dim = length(dp.zvec)
    nsteps = Int(round(tf / dt))

    # ── Run 1: Exact Schur Newton (reference) ──
    empty!(dyn.events)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, rfault, ton, toff))
    events = dyn.events
    event_schedule = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule, by = x -> x[1])

    sw_ex = GradPower.SchurWorkspace(ps)
    n_red = length(sw_ex.reduced_idx)
    J_ex = GradPower.preallocate_jacobian(ps)
    f_ex = zeros(Float64, sys_dim)
    z_ex = copy(dp.zvec); zold_ex = copy(dp.zvec)
    traj_exact = zeros(Float64, sys_dim, nsteps + 1)
    traj_exact[:, 1] .= dp.zvec
    iters_exact = zeros(Int, nsteps)

    sched_idx = 1
    for k in 1:nsteps
        copyto!(zold_ex, z_ex)
        ok, ni = newton_step_exact_schur_counted!(z_ex, f_ex, J_ex, sw_ex,
                    zold_ex, dp.uvec, dp.pvec, ps, dt; tol=tol)
        iters_exact[k] = ni
        traj_exact[:, k+1] .= z_ex

        any_event = false
        while sched_idx <= length(event_schedule) && event_schedule[sched_idx][1] == k
            _, idx, action = event_schedule[sched_idx]
            action === :on ? GradPower.activate!(events[idx]) : GradPower.deactivate!(events[idx])
            any_event = true; sched_idx += 1
        end
        if any_event
            copyto!(zold_ex, z_ex)
            newton_step_exact_schur_counted!(z_ex, f_ex, J_ex, sw_ex,
                zold_ex, dp.uvec, dp.pvec, ps, dt; tol=tol)
        end
    end
    for event in events; GradPower.deactivate!(event); end

    # ── Run 2: Modified Newton with frozen P + Woodbury ──
    empty!(dyn.events)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, rfault, ton, toff))
    events = dyn.events
    event_schedule2 = Tuple{Int,Int,Symbol}[]
    for (ei, ev) in enumerate(events)
        push!(event_schedule2, (Int(round(ev.ton / dt)), ei, :on))
        push!(event_schedule2, (Int(round(ev.toff / dt)), ei, :off))
    end
    sort!(event_schedule2, by = x -> x[1])

    P_fact, sw_mod = build_reference_P(ps, dp.zvec, dp.uvec, dp.pvec, dt)

    # Precompute Woodbury columns for the fault
    wc = precompute_woodbury(P_fact, n_red, fault_bus, rfault, net_ptr,
                              sw_mod.global_to_reduced)
    # Need a fresh P_fact since ldiv! in precompute consumed it
    # Actually klu factorizations are reusable — ldiv! doesn't destroy them
    # But we need to re-factor since the precompute_woodbury calls ldiv! which
    # modifies the rhs in-place, not the factorization. The factorization is fine.

    J_mod = GradPower.preallocate_jacobian(ps)
    f_mod = zeros(Float64, sys_dim)
    z_mod = copy(dp.zvec); zold_mod = copy(dp.zvec)
    traj_mod = zeros(Float64, sys_dim, nsteps + 1)
    traj_mod[:, 1] .= dp.zvec
    iters_mod = zeros(Int, nsteps)
    failed_steps = Int[]

    sched_idx = 1
    for k in 1:nsteps
        copyto!(zold_mod, z_mod)
        ok, ni = newton_step_shared_factor!(z_mod, f_mod, J_mod, sw_mod,
                    zold_mod, dp.uvec, dp.pvec, ps, dt, P_fact, wc; tol=tol)
        iters_mod[k] = ni
        if !ok; push!(failed_steps, k); end
        traj_mod[:, k+1] .= z_mod

        any_event = false
        while sched_idx <= length(event_schedule2) && event_schedule2[sched_idx][1] == k
            _, idx, action = event_schedule2[sched_idx]
            if action === :on
                GradPower.activate!(events[idx])
                wc.active[] = true
            elseif action === :off
                GradPower.deactivate!(events[idx])
                wc.active[] = false
            end
            any_event = true; sched_idx += 1
        end
        if any_event
            copyto!(zold_mod, z_mod)
            ok, ni = newton_step_shared_factor!(z_mod, f_mod, J_mod, sw_mod,
                        zold_mod, dp.uvec, dp.pvec, ps, dt, P_fact, wc; tol=tol)
            if !ok; push!(failed_steps, -k); end
        end
    end
    for event in events; GradPower.deactivate!(event); end

    # ── Report ──
    println()
    @printf("  System: %d unknowns, %d reduced (S), %d timesteps\n", sys_dim, n_red, nsteps)

    max_err = maximum(abs.(traj_mod .- traj_exact))
    @printf("  Max trajectory error (modified+Woodbury vs exact): %.2e\n", max_err)
    println()

    @printf("  Newton iterations per timestep:\n")
    @printf("    %-24s  %6s  %6s  %6s\n", "", "Mean", "Max", "Median")
    @printf("    %-24s  %6.1f  %6d  %6.0f\n", "Exact Schur",
            _mean(iters_exact), maximum(iters_exact), _median(iters_exact))
    @printf("    %-24s  %6.1f  %6d  %6.0f\n", "Modified+Woodbury",
            _mean(iters_mod), maximum(iters_mod), _median(iters_mod))
    ratio = _mean(iters_mod) / max(_mean(iters_exact), 1e-10)
    @printf("    Ratio (mod/exact):    %6.2fx\n", ratio)
    println()

    if !isempty(failed_steps)
        @printf("  ⚠ Modified Newton FAILED at %d step(s): %s\n",
                length(failed_steps), failed_steps[1:min(10, end)])
    else
        println("  ✓ Modified Newton converged at all timesteps")
    end
    println()

    fault_on_step = Int(round(ton / dt))
    fault_off_step = Int(round(toff / dt))
    pre = iters_mod[1:max(fault_on_step-1, 1)]
    pre_r = iters_exact[1:max(fault_on_step-1, 1)]
    dur = iters_mod[fault_on_step:min(fault_off_step, nsteps)]
    dur_r = iters_exact[fault_on_step:min(fault_off_step, nsteps)]
    post = fault_off_step < nsteps ? iters_mod[fault_off_step+1:end] : Int[]
    post_r = fault_off_step < nsteps ? iters_exact[fault_off_step+1:end] : Int[]

    @printf("  Iterations by phase:\n")
    @printf("    %-14s  %8s  %8s  %8s\n", "Phase", "Exact", "Modified", "Ratio")
    if !isempty(pre) && !isempty(pre_r)
        @printf("    %-14s  %8.1f  %8.1f  %8.2fx\n", "Pre-fault",
                _mean(pre_r), _mean(pre), _mean(pre)/max(_mean(pre_r), 1e-10))
    end
    if !isempty(dur) && !isempty(dur_r)
        @printf("    %-14s  %8.1f  %8.1f  %8.2fx\n", "Fault-on",
                _mean(dur_r), _mean(dur), _mean(dur)/max(_mean(dur_r), 1e-10))
    end
    if !isempty(post) && !isempty(post_r)
        @printf("    %-14s  %8.1f  %8.1f  %8.2fx\n", "Post-fault",
                _mean(post_r), _mean(post), _mean(post)/max(_mean(post_r), 1e-10))
    end
    println()
    flush(stdout)

    return (max_err=max_err, avg_mod=_mean(iters_mod), avg_ref=_mean(iters_exact),
            max_mod=maximum(iters_mod), max_ref=maximum(iters_exact),
            failed=length(failed_steps))
end

# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

println()
println("=" ^ 70)
println("PHASE 1+2: Shared-Factor + Woodbury — CPU Convergence Validation")
println("=" ^ 70)
println()
flush(stdout)

r1 = run_comparison("examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr",
                     "IEEE 9-bus (GENROU + IEESGO + ZIPLoad)")

r2 = run_comparison("examples/ACTIVSg200.raw", "examples/ACTIVSg200.dyr",
                     "ACTIVSg200 (200 buses)")

r3 = run_comparison("examples/ACTIVSg2000.raw", "examples/ACTIVSg2000.dyr",
                     "ACTIVSg2000 (2000 buses)")

println("=" ^ 70)
println("VERDICT")
println("=" ^ 70)
println()

all_pass = true
for (label, r) in [("IEEE 9", r1), ("ACTIVSg200", r2), ("ACTIVSg2000", r3)]
    ratio = r.avg_mod / max(r.avg_ref, 1e-10)
    status = (r.failed == 0 && r.max_err < 1e-3) ? "✓" : "✗"
    @printf("  %s %-14s  err=%.1e  iters=%.1f/%.1f (%.1fx)  failed=%d\n",
            status, label, r.max_err, r.avg_mod, r.avg_ref, ratio, r.failed)
    if r.failed > 0 || r.max_err > 1e-3
        global all_pass = false
    end
end
println()

if all_pass
    println("  ✓ Shared-factor + Woodbury converges on all cases")
    println("  → Proceed to Phase 3 (GPU implementation)")
else
    println("  ⚠ Issues remain — investigate before GPU port")
end
println()
flush(stdout)
