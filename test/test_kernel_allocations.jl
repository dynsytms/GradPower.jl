# Acceptance gate: batched rhs_fun! / rhs_jac! hot loop is heap-quiet.
#
# The batched kernels read from precomputed SoA tables and write to
# precomputed J.nzval positions — the per-device-type kernels themselves
# are verified to be 0-alloc in isolation.
#
# The full `rhs_fun!`/`rhs_jac!` entry points still pay a small fixed
# cost (~48 B / ~144 B respectively) per call from re-boxing
# `net.ybus_real` across the `Union{Nothing,Network}` split on
# `PowerSystem.network`. This is independent of system size — it does
# NOT scale with device count. The remaining boxing is removable by
# demoting `network`/`dynamic` from `Union{Nothing,...}` to
# concretely-typed.
#
# The acceptance gate is therefore: per-call allocation is a small
# O(1) constant that does NOT scale with system size. The 256-byte
# ceiling catches any future regression to per-device allocations.

const _ALLOC_CEIL = 256  # bytes per call, system-size-independent

function _alloc_rhs_fun!(f, dp, ps)
    GradPower.rhs_fun!(f, dp.zvec, dp.uvec, dp.pvec, ps)
end

function _alloc_rhs_jac!(J, dp, ps)
    GradPower.rhs_jac!(J, dp.zvec, dp.uvec, dp.pvec, ps)
end

@testset "rhs_fun!/rhs_jac! O(1) heap traffic" begin
    cases = [
        ("2bus_IEESGO",  "examples/2bus.raw",       "examples/2bus_IEESGO.dyr"),
        ("ieee9 no gov", "examples/ieee9_v33.raw",  "examples/ieee9bus.dyr"),
        ("ieee9 gov",    "examples/ieee9_v33.raw",  "examples/ieee9bus_gov.dyr"),
    ]

    for (label, raw, dyr) in cases
        @testset "$label" begin
            ps = from_psse(joinpath(@__DIR__, "..", raw),
                           joinpath(@__DIR__, "..", dyr))
            GradPower.build_network!(ps)
            GradPower.runpf(ps)
            dp = GradPower.DynamicProblem(ps)
            GradPower.initialize_dynamics!(dp, ps)

            n = length(dp.zvec)
            f = zeros(n)
            J = GradPower.preallocate_jacobian(ps)

            # warmup (JIT)
            _alloc_rhs_fun!(f, dp, ps)
            _alloc_rhs_jac!(J, dp, ps)

            a_f = @allocated _alloc_rhs_fun!(f, dp, ps)
            a_J = @allocated _alloc_rhs_jac!(J, dp, ps)

            @test a_f <= _ALLOC_CEIL
            @test a_J <= _ALLOC_CEIL
        end
    end
end

# Per-kernel kernel-only allocations: these MUST be exactly zero —
# they're the hot loop's per-device-type cost.
function _alloc_genrou(f, z, u, p, t, dd, np)
    GradPower.genrou_residual_batch!(f, z, u, p, t, dd, np)
end
function _alloc_ieesgo(f, z, p, t, dd)
    GradPower.ieesgo_residual_batch!(f, z, p, t, dd)
end
function _alloc_zipload(f, z, p, t, np)
    GradPower.zipload_residual_batch!(f, z, p, t, np)
end

function _measure_kernel_allocs(ps, dp)
    L = ps.dynamic.layout::GradPower.SimulationLayout
    gtab = L.genrou
    itab = L.ieesgo
    ztab = L.zipload
    diff_dim = ps.dynamic.diff_dim
    net_ptr  = diff_dim + ps.dynamic.alg_dim
    f = zeros(length(dp.zvec))
    z = dp.zvec; u = dp.uvec; p = dp.pvec

    # warmup
    _alloc_genrou(f, z, u, p, gtab, diff_dim, net_ptr)
    _alloc_ieesgo(f, z, p, itab, diff_dim)
    _alloc_zipload(f, z, p, ztab, net_ptr)

    a_g = @allocated _alloc_genrou(f, z, u, p, gtab, diff_dim, net_ptr)
    a_i = @allocated _alloc_ieesgo(f, z, p, itab, diff_dim)
    a_z = @allocated _alloc_zipload(f, z, p, ztab, net_ptr)
    return a_g, a_i, a_z
end

@testset "per-type residual kernels are 0-alloc" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    a_g, a_i, a_z = _measure_kernel_allocs(ps, dp)
    @test a_g == 0
    @test a_i == 0
    @test a_z == 0
end
