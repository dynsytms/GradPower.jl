using Test, LinearAlgebra, SparseArrays

function cpf_two_bus(; shift=0.0)
    ps = PowerSystem(100.0,
        [Bus(101, "reference", 3, 230., 1., 0.), Bus(205, "load", 1, 230., 1., 0.)],
        [Gen(1, "1", 0., 0., 100., true)], [Load(2, "1", 0.2, 0.)],
        [Branch(1, 2, "1", 0., 1., 0., 1., shift)], GradPower.Shunt[], Dict(101=>1, 205=>2))
    return ps
end

# Explicit vector perturbations avoid dependence on implicit identity sizes.
function cpf_difference(w, x, lambda; h=1e-6)
    A = zeros(length(x), length(x))
    for j in eachindex(x)
        a, b = copy(x), copy(x)
        a[j] += h
        b[j] -= h
        fa, fb = zeros(length(x)), zeros(length(x))
        GradPower._cpf_residual!(fa, w, a, lambda)
        GradPower._cpf_residual!(fb, w, b, lambda)
        A[:,j] .= (fa-fb)/(2h)
    end
    return A
end

@testset "Continuation formulation" begin
    ps = cpf_two_bus(; shift=12.0)
    d = LoadingDirection(ps)
    @test d.dp == [0., -0.2]
    @test d.dq == [0., 0.]
    @test d.bus_numbers == [101, 205]
    @test ps.network === nothing
    w = GradPower._cpf_workspace(ps, d)
    x = [0.1, 0.95]
    J = copy(GradPower._cpf_jacobian!(w, x))
    @test Matrix(J) ≈ cpf_difference(w, x, 0.3) atol=1e-8
    f1, f2 = zeros(2), zeros(2)
    GradPower._cpf_residual!(f1, w, x, 0.3+1e-6)
    GradPower._cpf_residual!(f2, w, x, 0.3-1e-6)
    @test (f1-f2)/2e-6 ≈ -w.direction atol=1e-9
    @test ps.network === nothing
    @test ps.buses[2].v0m == 1.
    @test_throws ArgumentError LoadingDirection(ps; dp=[0.])
    @test_throws ArgumentError LoadingDirection(ps; dp=[1.,0.])
    @test_throws ArgumentError LoadingDirection(ps; dq=[0., NaN])
    dp = [0., -1.]
    explicit = LoadingDirection(ps; dp)
    dp[2] = 3.
    @test explicit.dp[2] == -1.
    @test explicit.dq == zeros(2)
    @test_throws ArgumentError GradPower._cpf_workspace(ps, d; limits=:invalid)
    @test_throws ArgumentError GradPower._cpf_workspace(ps, d; control_scale=0.)
    isolated = deepcopy(ps)
    empty!(isolated.branches)
    @test_throws ArgumentError GradPower._cpf_workspace(isolated, d)
    @test_throws ArgumentError GradPower._cpf_workspace(ps, LoadingDirection(cpf_two_bus(); dp=[0.,1.]); control_scale=NaN)
end

@testset "Reactive control projection" begin
    control(q,v,l,u) = GradPower._cpf_control(q, v, 1., l, u, 2., :complementarity)
    @test control(0., 1., -1., 1.) == (0., 0., 2.)
    @test control(1., .9, -1., 1.) == (0., 1., 0.)
    @test control(-1., 1.1, -1., 1.) == (0., 1., 0.)
    @test first(control(1., 1.1, -1., 1.)) > 0 # upper limit must recover
    @test first(control(-1., .9, -1., 1.)) < 0 # lower limit must recover
    @test control(100., 1., -Inf, Inf) == (0., 0., 2.)
    @test control(1., .9, 1., 1.) == (0., 1., 0.)
    @test control(1., 1.1, 1., 1.) == (0., 1., 0.)
    @test control(1., 1., -1., 1.) == (0., 1., 0.) # saturated tie rule

    ps = cpf_two_bus()
    ps.buses[2].type = 2
    push!(ps.gens, Gen(2, "1", .1, .2, 100., true, .3, -.2))
    push!(ps.gens, Gen(2, "2", .2, .1, 100., true, .4, -.1))
    push!(ps.gens, Gen(2, "off", 100., 100., 100., false, 100., -100.))
    d = LoadingDirection(ps; dp=[0.,-.1], dq=[0.,-.2])
    w = GradPower._cpf_workspace(ps, d)
    @test w.lower ≈ [-.3]
    @test w.upper ≈ [.7]
    @test w.pbase[2] ≈ .1
    @test w.qbase[2] == 0.
    pattern = (copy(w.jacobian.colptr), copy(w.jacobian.rowval))
    for x in ([.1, 1., .2], [.1, .9, .8], [.1, 1.1, -.4])
        @test Matrix(GradPower._cpf_jacobian!(w,x)) ≈ cpf_difference(w,x,.2) atol=1e-8
        @test (w.jacobian.colptr, w.jacobian.rowval) == pattern
    end
    no_limits = GradPower._cpf_workspace(ps, d; limits=:none)
    x = [.1,.9,.8]
    @test Matrix(GradPower._cpf_jacobian!(no_limits,x)) ≈ cpf_difference(no_limits,x,.2) atol=1e-8
    ps.gens[2].qmin = 2.
    @test_throws ArgumentError GradPower._cpf_workspace(ps,d)
    ps.gens[2].status = false
    ps.gens[3].status = false
    @test_throws ArgumentError GradPower._cpf_workspace(ps,d)
end

@testset "Smooth pseudo-arclength continuation" begin
    ps = cpf_two_bus()
    snapshot = deepcopy(ps)
    direction = LoadingDirection(ps)
    result = GradPower._cpf_trace_smooth(ps,direction)
    @test result.status == :smooth_fold
    nose = last(result.points)
    # Lossless X=1, E=1, Q=0: Pmax=E²/(2X)=0.5;
    # P(lambda)=0.2*(1+lambda), hence lambda*=1.5.
    @test nose.lambda ≈ 1.5 atol=1e-8
    @test nose.state[1] ≈ -pi/4 atol=1e-7
    @test nose.state[2] ≈ inv(sqrt(2)) atol=1e-7
    @test abs(nose.tangent[end]) < 1e-7
    w = GradPower._cpf_workspace(ps,direction;limits=:none)
    weights = GradPower._cpf_weights(w)
    for point in result.points
        f = zeros(length(point.state))
        GradPower._cpf_residual!(f,w,point.state,point.lambda)
        @test norm(f,Inf) < 1e-9
        @test GradPower._cpf_norm(point.tangent,weights) ≈ 1.
        J = copy(GradPower._cpf_jacobian!(w,point.state))
        @test norm(J*point.tangent[1:end-1]-w.direction*point.tangent[end],Inf) < 1e-8
    end
    @test ps.network === nothing
    @test [b.v0m for b in ps.buses] == [b.v0m for b in snapshot.buses]
    @test [b.v0a for b in ps.buses] == [b.v0a for b in snapshot.buses]
    @test ps.loads[1].pd == snapshot.loads[1].pd
    @test ps.gens[1].qsch == snapshot.gens[1].qsch
    # The full augmented system continues through the singular unbordered
    # Jacobian, then follows the lower-voltage branch.
    full = GradPower._cpf_trace_smooth(ps,direction;stop=:steps,max_steps=40)
    @test full.status == :max_steps
    @test last(full.points).lambda < maximum(p.lambda for p in full.points)
    @test last(full.points).state[2] < inv(sqrt(2))
    @test any(p -> p.tangent[end]<0,full.points)
    for shift in (-12.,12.)
        shifted = GradPower._cpf_trace_smooth(cpf_two_bus(;shift))
        @test shifted.status == :smooth_fold
        @test last(shifted.points).lambda ≈ 1.5 atol=1e-8
        @test last(shifted.points).state[1] ≈ -pi/4-deg2rad(shift) atol=1e-7
    end
    # Nonzero reactive direction: P=.2+lambda, Q=.2lambda;
    # the discriminant E⁴-4XE²Q-4X²P² vanishes at this root.
    reactive = GradPower._cpf_trace_smooth(ps,LoadingDirection(ps;dp=[0.,-1.],dq=[0.,-.2]))
    @test reactive.status == :smooth_fold
    @test last(reactive.points).lambda ≈ (-.6+sqrt(.6^2-4*(.2^2-.25)))/2 atol=1e-8

    retried = GradPower._cpf_trace_smooth(ps;ds0=1.,ds_max=1.,predictor_tol=1e-4)
    @test retried.rejected_steps > 0
    @test retried.status == :smooth_fold
    @test last(retried.points).lambda ≈ nose.lambda atol=1e-8
    failed = GradPower._cpf_trace_smooth(ps;ds0=1.,ds_min=1.,ds_max=1.,predictor_tol=1e-10)
    @test failed.status == :minimum_step
    @test length(failed.points) == 1
    @test only(failed.points).lambda == 0.
    exhausted = GradPower._cpf_trace_smooth(ps;ds0=1.,ds_max=1.,max_retries=0,predictor_tol=1e-10)
    @test exhausted.status == :retry_limit
    @test length(exhausted.points) == 1
    @test GradPower._cpf_trace_smooth(ps;max_steps=0).status == :max_steps
    impossible = deepcopy(ps)
    impossible.loads[1].pd = 1.
    @test GradPower._cpf_trace_smooth(impossible).status == :initial_failure
    @test isempty(GradPower._cpf_trace_smooth(impossible).points)
    @test_throws ArgumentError GradPower._cpf_trace_smooth(ps;ds_min=1.,ds_max=.1)
    @test_throws ArgumentError GradPower._cpf_trace_smooth(ps;tol=NaN)
    @test_throws ArgumentError GradPower._cpf_trace_smooth(ps;stop=:unknown)
    @test_throws ArgumentError GradPower._cpf_trace_smooth(ps;max_steps=-1)
    @test_throws ArgumentError GradPower._cpf_trace_smooth(ps,LoadingDirection(ps;dp=zeros(2)))
end

@testset "Continuation with voltage-controlled generation" begin
    ps = cpf_two_bus()
    ps.buses[2].type = 2
    push!(ps.gens,Gen(2,"1",.1,0.,100.,true))
    direction = LoadingDirection(ps)
    result = GradPower._cpf_trace_smooth(ps,direction;max_steps=10)
    @test result.status == :max_steps
    @test length(result.points) == 11
    w = GradPower._cpf_workspace(ps,direction;limits=:none)
    for point in result.points
        @test point.state[2] ≈ 1. atol=1e-9
        f = zeros(length(point.state))
        GradPower._cpf_residual!(f,w,point.state,point.lambda)
        @test norm(f,Inf) < 1e-9
    end
    # Input changes after workspace creation cannot affect this snapshot.
    before = GradPower._cpf_residual!(zeros(3),w,w.initial,.2)
    ps.loads[1].pd = 100.
    direction.dp[2] = 100.
    ps.branches[1].x = 100.
    after = GradPower._cpf_residual!(zeros(3),w,w.initial,.2)
    @test before == after
end
