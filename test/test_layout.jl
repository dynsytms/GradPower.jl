# Acceptance test for per-device-type SoA tables.
#
# Asserts that the per-device-type SoA tables (GenrouTable, IEESGOTable,
# ESDC1ATable, ZIPLoadTable) built at the end of `set_dynamics!` faithfully
# mirror the data in the heterogeneous `psd.devices` vector.
#
# Cases exercised:
#   - 2bus      (1 GENROU, 1 ZIPLoad, 0 IEESGO, 0 ESDC1A)
#   - ieee9_v33 (3 GENROU, 3 ZIPLoad, 0 IEESGO, 0 ESDC1A in tracked .dyr)
#
# An IEESGO-populated case is synthesized in-test by manually adding 3 IEESGOs
# to the ieee9 psd before `set_dynamics!`. ESDC1A's populated path is NOT
# exercised here: `src/exciters.jl` is untracked and `AbstractExciterType` is
# not defined in the module, so no live case can instantiate ESDC1A.

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
    # yreal/yimag are 0 at build_layout! time (set_dynamics! snapshot),
    # populated by initialize_dynamics! afterwards via refresh_zipload_table!.
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
    # Wiring fixed via `wire_controls!`. With an IEESGO attached,
    # `pm_idx` is populated (slot 2 = `ctrl_ptr + 1`) and `efd_idx`
    # stays 0 (no exciter wired). Both sides of the wiring mirror
    # `uvec_idx` correctly.
    for k in 1:L.genrou.n
        cp = L.genrou.ctrl_ptr[k]
        @test L.genrou.has_exc[k] == false                # no exciter in this synthetic case
        @test L.genrou.has_gov[k] == true                 # IEESGO wired
        @test L.genrou.efd_idx[k] == 0                    # e_fd slot stays 0
        @test L.genrou.pm_idx[k]  == Int32(ps.dynamic.uvec_idx[cp + 1])  # p_m slot populated
        @test L.genrou.pm_idx[k]  != 0                    # ...and nonzero
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

@testset "layout 2.0c: ZIPLoad table refreshed after initialize_dynamics!" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout

    # At build_layout! time (end of set_dynamics!) yreal/yimag are still 0
    # because they require the PF solution. v0mag IS set at build time via
    # the ZIPLoad constructor reading ps.buses[load.bus].v0m.
    @test all(L.zipload.yreal .== 0.0)
    @test all(L.zipload.yimag .== 0.0)

    # initialize_dynamics! mutates ZIPLoad device structs from PF solution,
    # then refresh_zipload_table! must mirror those into the SoA columns.
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    # Every ZIPLoad must now have a populated v0mag (= bus voltage magnitude)
    # and a yreal corresponding to pinj/v0mag^2 (the load admittance).
    for k in 1:L.zipload.n
        @test L.zipload.v0mag[k] > 0.0
        # yreal/yimag should match the device's struct values (refresh is faithful).
        load_devs = [d for d in ps.dynamic.devices if d.dtype isa GradPower.ZIPLoad]
        @test L.zipload.v0mag[k] ≈ load_devs[k].dtype.v0mag
        @test L.zipload.yreal[k] ≈ load_devs[k].dtype.yreal
        @test L.zipload.yimag[k] ≈ load_devs[k].dtype.yimag
    end
end

@testset "layout 1.5: registry shape and dispatcher type-stability" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    L = ps.dynamic.layout

    # Registry round-trip: every device class in DEVICE_ORDER is reachable
    # in L.tables and has the expected concrete type.
    @test issubset(GradPower.DEVICE_ORDER, propertynames(L.tables))
    for name in GradPower.DEVICE_ORDER
        entry = GradPower.DEVICE_REGISTRY[name]
        @test getproperty(L.tables, name) isa entry.table_type
    end

    # Tables is a concrete NamedTuple of concrete types (the load-bearing
    # invariant for hot-loop specialization).
    T = typeof(L.tables)
    @test T <: NamedTuple
    @test isconcretetype(T)

    # Back-compat: L.genrou must still work via getproperty forward.
    @test L.genrou === L.tables.genrou
    @test L.ieesgo === L.tables.ieesgo

    # Type-stability + zero-allocation of the @generated dispatcher.
    # Wrap in a function so @allocated doesn't measure global-binding boxing.
    alloc_check(L) = (tables = L.tables; @allocated GradPower._dispatch_count_n(tables))
    alloc_check(L)  # warmup
    @test alloc_check(L) == 0

    # Sanity: total matches sum of n's.
    @test GradPower._dispatch_count_n(L.tables) ==
        L.genrou.n + L.ieesgo.n + L.esdc1a.n + L.zipload.n
end

@testset "layout 1.5b: wire_controls! fixes governor off-by-one" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    L = ps.dynamic.layout
    @test L.genrou.n == 3
    @test L.ieesgo.n == 3
    # Every generator should have a wired p_m (slot 2) and no exciter.
    for k in 1:L.genrou.n
        @test L.genrou.has_gov[k] == true
        @test L.genrou.pm_idx[k]  != 0
        @test L.genrou.has_exc[k] == false
        @test L.genrou.efd_idx[k] == 0
        cp = L.genrou.ctrl_ptr[k]
        @test ps.dynamic.uvec_idx[cp]     == 0                # e_fd slot empty
        @test ps.dynamic.uvec_idx[cp + 1] == L.genrou.pm_idx[k]  # p_m wired
    end
    # IEESGOs all wired to generator w states (nonzero w_idx).
    for k in 1:L.ieesgo.n
        @test L.ieesgo.w_idx[k] != 0
    end
end

@testset "layout: jac_pos shape (populated per-kernel)" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus.dyr"))
    L = ps.dynamic.layout
    # GENROU jac_pos has GENROU_JAC_NENTRIES per device.
    # IEESGO jac_pos has IEESGO_JAC_NENTRIES per device.
    @test size(L.genrou.jac_pos, 2)  == GradPower.GENROU_JAC_NENTRIES
    @test size(L.genrou.jac_pos, 1)  == L.genrou.n
    @test size(L.ieesgo.jac_pos, 2)  == GradPower.IEESGO_JAC_NENTRIES
    @test size(L.ieesgo.jac_pos, 1)  == L.ieesgo.n
    @test size(L.zipload.jac_pos, 2) == GradPower.ZIPLOAD_JAC_NENTRIES
    @test size(L.zipload.jac_pos, 1) == L.zipload.n
    # Remaining kernels: empty n×0 placeholder.
    @test size(L.esdc1a.jac_pos, 2)  == 0
    @test length(L.net_jac_pos)      == 0
end
