# Unit tests for cluster infrastructure (src/clusters.jl).
#
# Tests build_clusters!, reorder_state!, extract_Ak!, extract_Bk_Ck!.
# Cases:
#   - 2-bus (1 GENROU, no controllers — single gen-only cluster)
#   - ieee9+gov (3 GENROU + 3 IEESGO — gen+gov clusters)

# -- helpers ----------------------------------------------------------

function load_case(raw, dyr)
    ps = from_psse(raw, dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    return ps
end

function load_2bus()
    load_case(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
              joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
end

function load_ieee9_gov()
    load_case(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
              joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
end

# -- 1. build_clusters! and ClusterTable structure --------------------

@testset "clusters: 2-bus gen-only cluster" begin
    ps = load_2bus()
    ct = ps.dynamic.clusters
    devices = ps.dynamic.devices

    gen_devs = [i for (i,d) in enumerate(devices)
                if d.dtype isa GradPower.AbstractGeneratorType ||
                   d.dtype isa GradPower.StaticGenerator]

    @test length(ct.clusters) == length(gen_devs)

    cl = ct.clusters[1]
    @test cl.gen_idx in gen_devs
    @test cl.gov_idx == 0
    @test cl.exc_idx == 0
    @test cl.pss_idx == 0

    gen = devices[cl.gen_idx]
    @test cl.d_k == gen.dtype.diff_size
    @test cl.a_k == gen.dtype.alg_size
    @test cl.w_size == cl.d_k + cl.a_k
    @test cl.w_end - cl.w_start + 1 == cl.w_size
end

@testset "clusters: ieee9+gov gen+gov clusters" begin
    ps = load_ieee9_gov()
    ct = ps.dynamic.clusters
    devices = ps.dynamic.devices

    gen_indices = [i for (i,d) in enumerate(devices)
                   if d.dtype isa GradPower.AbstractGeneratorType]
    @test length(ct.clusters) >= length(gen_indices)

    # Every generator in exactly one cluster
    gen_seen = Dict{Int,Int}()
    for cl in ct.clusters
        gen_seen[cl.gen_idx] = get(gen_seen, cl.gen_idx, 0) + 1
    end
    for gi in gen_indices
        @test get(gen_seen, gi, 0) == 1
    end

    # Each cluster with a governor: controller bus matches generator bus
    for cl in ct.clusters
        if cl.gov_idx > 0
            @test devices[cl.gov_idx].dtype.bus == devices[cl.gen_idx].dtype.bus
            # w_size includes both gen and gov states
            expected = devices[cl.gen_idx].dtype.diff_size +
                       devices[cl.gen_idx].dtype.alg_size +
                       devices[cl.gov_idx].dtype.diff_size +
                       devices[cl.gov_idx].dtype.alg_size
            @test cl.w_size == expected
        end
    end

    # Type-tuple groups contiguous
    for (tt, s, e) in ct.type_groups
        for ci in s:e
            @test GradPower.cluster_type_tuple(ct.clusters[ci], devices) == tt
        end
    end
end

@testset "clusters: ieee9+gov state counts and z-ranges" begin
    ps = load_ieee9_gov()
    ct = ps.dynamic.clusters
    devices = ps.dynamic.devices

    # w_size matches summed device sizes
    for cl in ct.clusters
        expected_d = 0; expected_a = 0
        for di in GradPower._cluster_device_order(cl)
            expected_d += devices[di].dtype.diff_size
            expected_a += devices[di].dtype.alg_size
        end
        @test cl.d_k == expected_d
        @test cl.a_k == expected_a
        @test cl.w_size == expected_d + expected_a
    end

    # No overlapping z-ranges
    for i in 1:length(ct.clusters)
        for j in i+1:length(ct.clusters)
            a = ct.clusters[i]; b = ct.clusters[j]
            @test a.w_end < b.w_start || b.w_end < a.w_start
        end
    end
end

# -- 2. reorder_state! and pointer consistency ------------------------

@testset "clusters: reorder pointer consistency (2-bus)" begin
    ps = load_2bus()
    psd = ps.dynamic
    ct = psd.clusters
    n_da = psd.diff_dim + psd.alg_dim

    # diff_indices length and is_diff consistency
    @test length(psd.diff_indices) == psd.diff_dim
    for i in psd.diff_indices
        @test psd.is_diff[i] == true
    end
    @test sum(psd.is_diff) == psd.diff_dim

    # Clusters tile 1:n_da without gaps
    covered = falses(n_da)
    for cl in ct.clusters
        for i in cl.w_start:cl.w_end
            @test covered[i] == false
            covered[i] = true
        end
    end
    @test all(covered)

    # SoA table diff_ptr matches DynamicDevice diff_ptr
    L = psd.layout
    gen_devs = [(i,d) for (i,d) in enumerate(psd.devices)
                if d.dtype isa GradPower.Genrou]
    for (k, (di, dev)) in enumerate(gen_devs)
        @test L.genrou.diff_ptr[k] == dev.diff_ptr
    end
end

@testset "clusters: reorder pointer consistency (ieee9+gov)" begin
    ps = load_ieee9_gov()
    psd = ps.dynamic
    ct = psd.clusters
    n_da = psd.diff_dim + psd.alg_dim

    # diff_indices / is_diff
    @test length(psd.diff_indices) == psd.diff_dim
    @test sum(psd.is_diff) == psd.diff_dim

    # Clusters tile 1:n_da
    covered = falses(n_da)
    for cl in ct.clusters
        for i in cl.w_start:cl.w_end
            @test covered[i] == false
            covered[i] = true
        end
    end
    @test all(covered)
end

@testset "clusters: flat-line after reorder (2-bus)" begin
    ps = load_2bus()
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    f = zeros(length(dp.zvec))
    GradPower.rhs_fun!(f, dp.zvec, dp.uvec, dp.pvec, ps)
    @test maximum(abs, f) < 1e-9
end

# -- 3. extract_Ak! and extract_Bk_Ck! -------------------------------

@testset "clusters: extract_Ak! matches global Jacobian (2-bus)" begin
    ps = load_2bus()
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    J = GradPower.preallocate_jacobian(ps)
    GradPower.rhs_jac!(J, dp.zvec, dp.uvec, dp.pvec, ps)

    ct = ps.dynamic.clusters
    for cl in ct.clusters
        wk = cl.w_size
        A = zeros(wk, wk)
        GradPower.extract_Ak!(A, J, cl)
        A_ref = Matrix(J[cl.w_start:cl.w_end, cl.w_start:cl.w_end])
        @test maximum(abs, A .- A_ref) <= 1e-15
    end
end

@testset "clusters: extract_Bk_Ck! matches global Jacobian (2-bus)" begin
    ps = load_2bus()
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    J = GradPower.preallocate_jacobian(ps)
    GradPower.rhs_jac!(J, dp.zvec, dp.uvec, dp.pvec, ps)

    ct = ps.dynamic.clusters
    net_ptr = ps.dynamic.diff_dim + ps.dynamic.alg_dim

    for cl in ct.clusters
        wk = cl.w_size
        B = zeros(wk, 2)
        C = zeros(2, wk)
        GradPower.extract_Bk_Ck!(B, C, J, cl, net_ptr)

        vr = net_ptr + 2*(cl.bus - 1) + 1
        vi = vr + 1
        B_ref = Matrix(J[cl.w_start:cl.w_end, [vr, vi]])
        C_ref = Matrix(J[[vr, vi], cl.w_start:cl.w_end])
        @test maximum(abs, B .- B_ref) <= 1e-15
        @test maximum(abs, C .- C_ref) <= 1e-15
    end
end
