# Tests for control signal routing (src/coupling.jl).
#
# Verifies wire_controls! and the trait system: attaches_to,
# produces_signals, consumes_signals, and PSS two-hop wiring.

const EX_CP = joinpath(@__DIR__, "..", "examples")

# -----------------------------------------------------------------------
# Trait correctness
# -----------------------------------------------------------------------

@testset "Coupling traits" begin
    @test GradPower.attaches_to(GradPower.IEESGO) === GradPower.Genrou
    @test GradPower.attaches_to(GradPower.TGOV1)  === GradPower.Genrou
    @test GradPower.attaches_to(GradPower.SEXS)   === GradPower.Genrou
    @test GradPower.attaches_to(GradPower.ESDC1A) === GradPower.Genrou
    @test GradPower.attaches_to(GradPower.IEEEST) === GradPower.AbstractExciterType

    # Generators and loads are not controllers
    @test GradPower.attaches_to(GradPower.Genrou)  === nothing
    @test GradPower.attaches_to(GradPower.ZIPLoad) === nothing

    # IEESGO produces p_m (target_ctrl_offset=1, alg_first)
    sigs = GradPower.produces_signals(GradPower.IEESGO)
    @test length(sigs) == 1
    @test sigs[1].target_ctrl_offset == 1
    @test sigs[1].source_kind === :alg_first

    # TGOV1 same shape as IEESGO
    sigs_tg = GradPower.produces_signals(GradPower.TGOV1)
    @test length(sigs_tg) == 1
    @test sigs_tg[1].target_ctrl_offset == 1
    @test sigs_tg[1].source_kind === :alg_first

    # SEXS produces e_fd (target_ctrl_offset=0, diff_at offset=1)
    sigs_sx = GradPower.produces_signals(GradPower.SEXS)
    @test length(sigs_sx) == 1
    @test sigs_sx[1].target_ctrl_offset == 0
    @test sigs_sx[1].source_kind === :diff_at
    @test sigs_sx[1].source_offset == 1

    # ESDC1A produces e_fd (target_ctrl_offset=0, diff_at offset=2)
    sigs_ed = GradPower.produces_signals(GradPower.ESDC1A)
    @test length(sigs_ed) == 1
    @test sigs_ed[1].target_ctrl_offset == 0
    @test sigs_ed[1].source_kind === :diff_at
    @test sigs_ed[1].source_offset == 2

    # IEESGO consumes w (ctrl_offset=0, state_kind=:w)
    csigs = GradPower.consumes_signals(GradPower.IEESGO)
    @test length(csigs) == 1
    @test csigs[1].ctrl_offset == 0
    @test csigs[1].state_kind === :w

    # IEEEST produces/consumes nothing (wired specially in wire_controls!)
    @test isempty(GradPower.produces_signals(GradPower.IEEEST))
    @test isempty(GradPower.consumes_signals(GradPower.IEEEST))
end

# -----------------------------------------------------------------------
# Governor wiring: IEESGO
# -----------------------------------------------------------------------

@testset "IEESGO governor wiring (ieee9)" begin
    ps = from_psse(joinpath(EX_CP, "ieee9_v33.raw"),
                   joinpath(EX_CP, "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    uvec = dyn.uvec_idx

    # For each generator+governor pair: verify wiring
    L = dyn.layout
    for k in 1:L.ieesgo.n
        # Governor's alg state in z: diff_dim + alg_ptr
        gov_pm_z = dyn.diff_dim + Int(L.ieesgo.alg_ptr[k])

        # Find the matching generator (same bus)
        gov_bus = Int(L.ieesgo.bus[k])
        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == gov_bus
                gen_k = g
                break
            end
        end
        @test gen_k > 0

        # Generator's ctrl slot for p_m: ctrl_ptr + 1
        gen_ctrl_pm = Int(L.genrou.ctrl_ptr[gen_k]) + 1
        # uvec_idx should map this ctrl slot to governor's alg state
        @test uvec[gen_ctrl_pm] == gov_pm_z

        # Governor's ctrl slot for w: ctrl_ptr + 0
        gov_ctrl_w = Int(L.ieesgo.ctrl_ptr[k])
        # Generator's w state: diff_ptr + 4 (w is 5th diff state, 0-based offset 4)
        gen_w_z = Int(L.genrou.diff_ptr[gen_k]) + 4
        @test uvec[gov_ctrl_w] == gen_w_z
    end
end

# -----------------------------------------------------------------------
# Governor wiring: TGOV1
# -----------------------------------------------------------------------

@testset "TGOV1 governor wiring (2bus)" begin
    ps = from_psse(joinpath(EX_CP, "2bus.raw"),
                   joinpath(EX_CP, "2bus_TGOV1.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    uvec = dyn.uvec_idx
    L = dyn.layout

    @test L.tgov1.n >= 1
    for k in 1:L.tgov1.n
        gov_pm_z = dyn.diff_dim + Int(L.tgov1.alg_ptr[k])
        gov_bus = Int(L.tgov1.bus[k])

        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == gov_bus
                gen_k = g; break
            end
        end
        @test gen_k > 0

        gen_ctrl_pm = Int(L.genrou.ctrl_ptr[gen_k]) + 1
        @test uvec[gen_ctrl_pm] == gov_pm_z

        gov_ctrl_w = Int(L.tgov1.ctrl_ptr[k])
        gen_w_z = Int(L.genrou.diff_ptr[gen_k]) + 4
        @test uvec[gov_ctrl_w] == gen_w_z
    end
end

# -----------------------------------------------------------------------
# Exciter wiring: SEXS
# -----------------------------------------------------------------------

@testset "SEXS exciter wiring (2bus)" begin
    ps = from_psse(joinpath(EX_CP, "2bus.raw"),
                   joinpath(EX_CP, "2bus_SEXS.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    uvec = dyn.uvec_idx
    L = dyn.layout

    @test L.sexs.n >= 1
    for k in 1:L.sexs.n
        # SEXS e_fd source: diff_ptr + 1 (source_offset=1)
        sexs_efd_z = Int(L.sexs.diff_ptr[k]) + 1
        sexs_bus = Int(L.sexs.bus[k])

        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == sexs_bus
                gen_k = g; break
            end
        end
        @test gen_k > 0

        # Generator's ctrl slot for e_fd: ctrl_ptr + 0
        gen_ctrl_efd = Int(L.genrou.ctrl_ptr[gen_k])
        @test uvec[gen_ctrl_efd] == sexs_efd_z
    end
end

# -----------------------------------------------------------------------
# Exciter wiring: ESDC1A
# -----------------------------------------------------------------------

@testset "ESDC1A exciter wiring (2bus)" begin
    ps = from_psse(joinpath(EX_CP, "2bus.raw"),
                   joinpath(EX_CP, "2bus_ESDC1A.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    uvec = dyn.uvec_idx
    L = dyn.layout

    @test L.esdc1a.n >= 1
    for k in 1:L.esdc1a.n
        # ESDC1A e_fd source: diff_ptr + 2 (source_offset=2)
        esdc1a_efd_z = Int(L.esdc1a.diff_ptr[k]) + 2
        esdc1a_bus = Int(L.esdc1a.bus[k])

        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == esdc1a_bus
                gen_k = g; break
            end
        end
        @test gen_k > 0

        gen_ctrl_efd = Int(L.genrou.ctrl_ptr[gen_k])
        @test uvec[gen_ctrl_efd] == esdc1a_efd_z
    end
end

# -----------------------------------------------------------------------
# PSS two-hop wiring: IEEEST → SEXS → GENROU
# -----------------------------------------------------------------------

@testset "IEEEST PSS two-hop wiring (2bus)" begin
    ps = from_psse(joinpath(EX_CP, "2bus.raw"),
                   joinpath(EX_CP, "2bus_IEEEST.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    uvec = dyn.uvec_idx
    L = dyn.layout

    @test L.ieeest.n >= 1
    @test L.sexs.n >= 1
    @test L.genrou.n >= 1

    for k in 1:L.ieeest.n
        pss_bus = Int(L.ieeest.bus[k])

        # Find the generator on the same bus
        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == pss_bus
                gen_k = g; break
            end
        end
        @test gen_k > 0

        # PSS ctrl slot 0 should point to generator's w state (two-hop)
        pss_ctrl_w = Int(L.ieeest.ctrl_ptr[k])
        gen_w_z = Int(L.genrou.diff_ptr[gen_k]) + 4
        @test uvec[pss_ctrl_w] == gen_w_z

        # SEXS vs_idx should point to IEEEST's alg output (v_s)
        sexs_k = 0
        for s in 1:L.sexs.n
            if Int(L.sexs.bus[s]) == pss_bus
                sexs_k = s; break
            end
        end
        @test sexs_k > 0
        pss_vs_z = dyn.diff_dim + Int(L.ieeest.alg_ptr[k])
        @test L.sexs.vs_idx[sexs_k] == pss_vs_z
    end
end

# -----------------------------------------------------------------------
# Signal propagation: verify actual signal flow
# -----------------------------------------------------------------------

@testset "Signal propagation: uvec routing delivers correct values" begin
    ps = from_psse(joinpath(EX_CP, "ieee9_v33.raw"),
                   joinpath(EX_CP, "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic
    L = dyn.layout

    # Apply uvec routing
    z = dp.zvec
    u = copy(dp.uvec)
    GradPower._apply_uvec_routing!(u, z, dyn.uvec_idx)

    # For each generator with a governor, u[ctrl_ptr+1] should equal
    # z[governor_pm_state]
    for k in 1:L.ieesgo.n
        gov_bus = Int(L.ieesgo.bus[k])
        gov_pm_z = dyn.diff_dim + Int(L.ieesgo.alg_ptr[k])

        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == gov_bus
                gen_k = g; break
            end
        end
        gen_k == 0 && continue

        gen_ctrl_pm = Int(L.genrou.ctrl_ptr[gen_k]) + 1
        @test u[gen_ctrl_pm] == z[gov_pm_z]
    end

    # Governor w input should equal generator w state
    for k in 1:L.ieesgo.n
        gov_ctrl_w = Int(L.ieesgo.ctrl_ptr[k])
        gov_bus = Int(L.ieesgo.bus[k])

        gen_k = 0
        for g in 1:L.genrou.n
            if Int(L.genrou.bus[g]) == gov_bus
                gen_k = g; break
            end
        end
        gen_k == 0 && continue

        gen_w_z = Int(L.genrou.diff_ptr[gen_k]) + 4
        @test u[gov_ctrl_w] == z[gen_w_z]
    end
end
