# Phase 1 acceptance test for ROADMAP.md §5 Phase 1.
#
# Asserts that the per-device-type SoA tables (GenrouTable, IEESGOTable,
# ESDC1ATable, ZIPLoadTable) built at the end of `set_dynamics!` faithfully
# mirror the data in the heterogeneous `psd.devices` vector. In Phase 1 the
# tables are built but not used by the hot loop; this test fixes the contract
# that Phase 2 will rely on.
#
# Cases exercised:
#   - 2bus      (1 GENROU, 1 ZIPLoad, 0 IEESGO, 0 ESDC1A)
#   - ieee9_v33 (3 GENROU, 3 ZIPLoad, 0 IEESGO, 0 ESDC1A in tracked .dyr)
#
# An IEESGO-populated case is synthesized in-test by manually adding 3 IEESGOs
# to the ieee9 psd before `set_dynamics!`. ESDC1A's populated path is NOT
# exercised here: `src/exciters.jl` is untracked and `AbstractExciterType` is
# not defined in the module, so no live case can instantiate ESDC1A. Phase 2/3
# will add coverage once exciters are wired in.

@testset "layout: table existence and counts" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    @test ps.dynamic.layout !== nothing
    L = ps.dynamic.layout
    @test L.genrou.n == 1
    @test L.ieesgo.n == 0
    @test L.esdc1a.n == 0
    # examples/2bus.raw has 1 static load → auto-added ZIPLoad
    @test L.zipload.n == 1
end

@testset "layout: ieee9 — GenrouTable mirrors psd.devices" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout
    @test L.genrou.n == 3
    @test L.zipload.n == 3   # IEEE9 has 3 loads
    @test L.ieesgo.n == 0    # tracked ieee9bus.dyr has no IEESGO
    @test L.esdc1a.n == 0

    gen_devs = [d for d in ps.dynamic.devices if d.dtype isa GradPower.Genrou]
    @test length(gen_devs) == L.genrou.n
    for k in 1:L.genrou.n
        g = gen_devs[k].dtype
        @test L.genrou.bus[k] == ps.dynamic.map.bus[findfirst(d -> d === gen_devs[k], ps.dynamic.devices)]
        @test L.genrou.diff_ptr[k] == gen_devs[k].diff_ptr
        @test L.genrou.alg_ptr[k]  == gen_devs[k].alg_ptr
        @test L.genrou.ctrl_ptr[k] == gen_devs[k].ctrl_ptr
        @test L.genrou.par_ptr[k]  == gen_devs[k].par_ptr
        # parameters
        @test L.genrou.x_d[k]    ≈ g.x_d
        @test L.genrou.x_q[k]    ≈ g.x_q
        @test L.genrou.x_dp[k]   ≈ g.x_dp
        @test L.genrou.x_qp[k]   ≈ g.x_qp
        @test L.genrou.x_ddp[k]  ≈ g.x_ddp
        @test L.genrou.xl[k]     ≈ g.xl
        @test L.genrou.H[k]      ≈ g.H
        @test L.genrou.D[k]      ≈ g.D
        @test L.genrou.T_d0p[k]  ≈ g.T_d0p
        @test L.genrou.T_q0p[k]  ≈ g.T_q0p
        @test L.genrou.T_d0dp[k] ≈ g.T_d0dp
        @test L.genrou.T_q0dp[k] ≈ g.T_q0dp
        # control coupling — no gov/exc in ieee9bus.dyr
        @test L.genrou.has_gov[k] == false
        @test L.genrou.has_exc[k] == false
        @test L.genrou.pm_idx[k]  == 0
        @test L.genrou.efd_idx[k] == 0
    end
end

@testset "layout: ieee9 — ZIPLoadTable mirrors psd.devices" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout
    load_devs = [d for d in ps.dynamic.devices if d.dtype isa GradPower.ZIPLoad]
    @test length(load_devs) == L.zipload.n
    for k in 1:L.zipload.n
        z = load_devs[k].dtype
        @test L.zipload.bus[k] == ps.dynamic.map.bus[findfirst(d -> d === load_devs[k], ps.dynamic.devices)]
        @test L.zipload.par_ptr[k] == load_devs[k].par_ptr
        @test L.zipload.pinj[k]   ≈ z.pinj
        @test L.zipload.qinj[k]   ≈ z.qinj
        @test L.zipload.α[k]      ≈ z.α
        @test L.zipload.β[k]      ≈ z.β
        @test L.zipload.γ[k]      ≈ z.γ
        @test L.zipload.weight[k] ≈ z.weight
        @test L.zipload.v0mag[k]  ≈ z.v0mag
        @test L.zipload.yreal[k]  ≈ z.yreal
        @test L.zipload.yimag[k]  ≈ z.yimag
    end
    # ROADMAP §5b note: yreal/yimag are 0 at build_layout! time (set_dynamics!
    # snapshot), populated by initialize_dynamics! afterwards. Phase 2 will
    # decide whether to re-snapshot post-init or read live from dp.pvec.
    @test all(L.zipload.yreal .== 0.0)
    @test all(L.zipload.yimag .== 0.0)
end

@testset "layout: synthetic ieee9 + IEESGO populates IEESGOTable" begin
    # Hand-build a populated IEESGO case since no tracked .dyr has IEESGO.
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"), nothing)
    psd = GradPower.PowerSystemDynamics()

    # Read the 3 GENROU devices that the parser would have created
    raw_psd = GradPower.PowerSystemDynamics(joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    for dev in raw_psd.devices
        GradPower.add_device!(psd, dev.dtype)
    end
    # Add a stock IEESGO on each generator's bus (id matched to the generator)
    for dev in raw_psd.devices
        if dev.dtype isa GradPower.Genrou
            gov = GradPower.IEESGO(
                dev.dtype.bus, dev.dtype.id,
                0.5, 1.0, 0.6, 0.5, 1.0, 1.0,    # T1..T6
                10.0, 0.5, 0.5,                   # K1..K3
                1.0, 0.0,                         # pmax, pmin
            )
            GradPower.add_device!(psd, gov)
        end
    end
    GradPower.set_dynamics!(ps, psd)

    L = ps.dynamic.layout
    @test L.ieesgo.n == 3
    @test L.genrou.n == 3
    # NOTE: the existing `set_dynamics!` wiring at src/GradPower.jl:289-318
    # routes `p_m` to the generator's e_fd ctrl slot (off-by-one), so the
    # GENROU-side `has_gov`/`pm_idx` arrays currently mirror that bug. The
    # IEESGO side (`w_idx`) is correctly populated. Phase 2 fixes the wiring.
    # For Phase 1, we just assert the layout faithfully mirrors uvec_idx as-is.
    for k in 1:L.genrou.n
        cp = L.genrou.ctrl_ptr[k]
        @test L.genrou.has_exc[k] == (ps.dynamic.uvec_idx[cp] != 0)
        @test L.genrou.has_gov[k] == (ps.dynamic.uvec_idx[cp + 1] != 0)
        @test L.genrou.efd_idx[k] == Int32(ps.dynamic.uvec_idx[cp])
        @test L.genrou.pm_idx[k]  == Int32(ps.dynamic.uvec_idx[cp + 1])
    end

    gov_devs = [d for d in ps.dynamic.devices if d.dtype isa GradPower.IEESGO]
    @test length(gov_devs) == L.ieesgo.n
    for k in 1:L.ieesgo.n
        gv = gov_devs[k].dtype
        @test L.ieesgo.diff_ptr[k] == gov_devs[k].diff_ptr
        @test L.ieesgo.alg_ptr[k]  == gov_devs[k].alg_ptr
        @test L.ieesgo.ctrl_ptr[k] == gov_devs[k].ctrl_ptr
        @test L.ieesgo.par_ptr[k]  == gov_devs[k].par_ptr
        @test L.ieesgo.T1[k]   ≈ gv.T1
        @test L.ieesgo.T2[k]   ≈ gv.T2
        @test L.ieesgo.T3[k]   ≈ gv.T3
        @test L.ieesgo.T4[k]   ≈ gv.T4
        @test L.ieesgo.T5[k]   ≈ gv.T5
        @test L.ieesgo.T6[k]   ≈ gv.T6
        @test L.ieesgo.K1[k]   ≈ gv.K1
        @test L.ieesgo.K2[k]   ≈ gv.K2
        @test L.ieesgo.K3[k]   ≈ gv.K3
        @test L.ieesgo.pmax[k] ≈ gv.pmax
        @test L.ieesgo.pmin[k] ≈ gv.pmin
        # w_idx — generator's w state is at diff_ptr + 4 (per src/GradPower.jl:309)
        @test L.ieesgo.w_idx[k] != 0
    end
end

@testset "layout: per-type counts sum to total dynamic devices" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout
    n_total = L.genrou.n + L.ieesgo.n + L.esdc1a.n + L.zipload.n
    @test n_total == ps.dynamic.num_devices
end

@testset "layout: jac_pos is empty (Phase 1 contract; Phase 2 populates)" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout
    @test size(L.genrou.jac_pos, 2)  == 0
    @test size(L.ieesgo.jac_pos, 2)  == 0
    @test size(L.esdc1a.jac_pos, 2)  == 0
    @test size(L.zipload.jac_pos, 2) == 0
    @test length(L.net_jac_pos)      == 0
end
