@testset "Public continuation API and physical results" begin
    ps = cpf_controlled_bus()
    snapshot = deepcopy(ps)
    result = runcpf(ps)
    @test result isa ContinuationResult
    @test result.status == :smooth_fold
    @test result.success
    @test result.limits == :complementarity
    @test result.stop == :nose
    @test result.bus_numbers == [101,205]
    @test result.baseMVA == 100.
    @test result.maximum === last(result.points)
    @test result.sampled_maximum === result.maximum
    @test result.diagnostic.status == :simple_fold
    @test result.maximum.lambda ≈ sqrt(.45)/.2-1 atol=1e-8
    @test result.events isa Vector{ContinuationEvent}
    @test length(result.events) == 1
    @test only(result.events).point.control == [:reference,:qmax]
    for point in result.points
        @test point isa ContinuationPoint
        @test length(point.vm) == length(point.va) == length(point.pg) == length(point.qg) == 2
        @test point.vm[1] == 1.
        @test point.va[1] == 0.
        @test point.pg[2] == 0. # no fictitious active generation at the PV bus
        @test point.pinj[2] ≈ -.2*(1+point.lambda) atol=1e-7
        @test point.qinj[2] ≈ point.qg[2] atol=1e-7
        @test point.pg[1] ≈ point.pinj[1] atol=1e-7
        @test point.qg[1] ≈ point.qinj[1] atol=1e-7
        @test point.residual <= 2e-7
        @test point.iterations >= 0
    end
    event = only(result.events)
    @test event.bus == 205
    @test event.kind == :qmax
    @test event.point.lambda ≈ 2. atol=1e-6
    @test event.incoming.lambda > 0 && event.outgoing.lambda > 0
    @test event.incoming.vm[2] ≈ 0. atol=1e-10
    @test event.incoming.qg[2] > 0
    @test event.outgoing.qg[2] ≈ 0. atol=1e-10
    @test isnan(event.incoming.qg[1])
    @test event.error <= 1e-7
    @test ps.network === nothing
    @test [b.v0m for b in ps.buses] == [b.v0m for b in snapshot.buses]
    @test [b.type for b in ps.buses] == [b.type for b in snapshot.buses]
    @test [g.qsch for g in ps.gens] == [g.qsch for g in snapshot.gens]
    @test ps.loads[1].pd == snapshot.loads[1].pd
    # Preserves a prebuilt network too; the workspace constructs its own Ybus.
    build_network!(ps)
    network = ps.network
    ybus = copy(network.ybus)
    selected = runcpf(ps;buses=[205,101])
    @test ps.network === network
    @test ps.network.ybus == ybus
    @test selected.bus_numbers == [205,101]
    @test length(selected.points) == length(result.points)
    for (a,b) in zip(selected.points,result.points)
        for field in (:vm,:va,:pinj,:qinj,:pg,:qg,:control)
            @test getproperty(a,field) == reverse(getproperty(b,field))
        end
        @test a.lambda == b.lambda
    end
    reference_only = runcpf(ps;buses=[101])
    @test length(reference_only.points) == length(result.points)
    @test only(reference_only.events).bus == 205 # unrecorded event still identified
    @test length(only(reference_only.events).point.vm) == 1
    scalar_only = runcpf(ps;buses=Int[])
    @test isempty(scalar_only.bus_numbers)
    @test all(p -> isempty(p.vm) && isempty(p.pg) && isempty(p.control),scalar_only.points)
    @test scalar_only.maximum.lambda == result.maximum.lambda
    @test only(scalar_only.events).bus == 205
    @test isempty(only(scalar_only.events).incoming.vm)
    @test Base.summarysize(scalar_only.points) < Base.summarysize(result.points)
    # Results own their vectors, independently of the input and other runs.
    selected.points[1].vm[1] = -1.
    @test result.points[1].vm[2] > 0
    ps.buses[2].v0m = .9
    @test result.points[1].vm[2] == 1.
    @test_throws ArgumentError runcpf(ps;buses=[2]) # external, not internal
    @test_throws ArgumentError runcpf(ps;buses=[205,205])
    @test_throws ArgumentError runcpf(ps;buses=[205.])
    @test_throws ArgumentError runcpf(ps;buses=[true])
    @test_throws ArgumentError runcpf(ps;buses=205)
    @test_throws ArgumentError runcpf(ps;limits=:invalid)
    @test_throws ArgumentError runcpf(ps;event_tol=1e-12)
    @test_throws ArgumentError runcpf(ps;control_scale=0.)
    @test_throws ArgumentError runcpf(ps;max_steps=-1)
    @test_throws ArgumentError runcpf(ps;max_steps=2.5)
    @test_throws ArgumentError runcpf(ps;max_retries=true)
end

@testset "Partial traces and maximum semantics" begin
    ps = cpf_two_bus()
    incomplete = runcpf(ps;max_steps=0)
    @test incomplete.status == :max_steps
    @test !incomplete.success
    @test incomplete.maximum === nothing
    @test incomplete.sampled_maximum === only(incomplete.points)
    @test occursin("incomplete",incomplete.message)
    complete_steps = runcpf(ps;stop=:steps,max_steps=0)
    @test complete_steps.success
    @test complete_steps.maximum === nothing
    @test complete_steps.status == :max_steps
    lower = runcpf(ps;limits=:none,stop=:steps,max_steps=40)
    @test lower.success
    @test lower.maximum === nothing
    @test lower.sampled_maximum.lambda > last(lower.points).lambda
    @test lower.sampled_maximum.lambda < 1.5
    @test isempty(lower.events)
    failed = runcpf(ps;ds0=1.,ds_max=1.,max_retries=0,predictor_tol=1e-12)
    @test failed.status == :retry_limit
    @test failed.rejected_steps == 1
    @test !failed.success
    @test failed.maximum === nothing
    @test length(failed.points) == 1
    @test isempty(failed.events)
    ps.loads[1].pd = 1.
    initial_failure = runcpf(ps)
    @test initial_failure.status == :initial_failure
    @test isempty(initial_failure.points)
    @test initial_failure.maximum === nothing
    @test initial_failure.sampled_maximum === nothing
    @test !initial_failure.success
    kink = cpf_controlled_bus(;voltage=.6,qmin=-.4,qmax=-.1)
    kink.gens[2].qsch = -.2
    limited = runcpf(kink)
    @test limited.status == :limit_induced_maximum
    @test limited.success
    @test limited.maximum === last(limited.points)
    @test only(limited.events).incoming.lambda > 0
    @test only(limited.events).outgoing.lambda < 0
    @test limited.maximum.control[2] == :qmax
    coincident = runcpf(cpf_controlled_bus(;qmax=.5);tol=1e-11,event_tol=1e-8,buses=Int[])
    @test coincident.status == :fold_control_coincidence
    @test !coincident.success
    @test coincident.maximum === nothing
    @test coincident.sampled_maximum !== nothing
    @test !isempty(keys(coincident.diagnostic))
    @test all(p -> isempty(p.vm),coincident.points)
    # A deliberately flat/degenerate direction must not claim a maximum.
    @test_throws ArgumentError runcpf(cpf_two_bus(),LoadingDirection(cpf_two_bus();dp=zeros(2)))
end

@testset "Generation versus general net directions" begin
    ps = cpf_two_bus()
    ps.gens[1].psch = 10. # reference report must reflect actual balancing
    ps.gens[1].qsch = 10.
    push!(ps.gens,Gen(2,"offline",50.,50.,100.,false))
    push!(ps.gens,Gen(2,"online",.05,.02,100.,true))
    d = LoadingDirection(ps;dp=[0.,-.3],dq=[0.,-.1])
    r = runcpf(ps,d;stop=:steps,max_steps=3)
    @test r.success
    for p in r.points
        @test p.pg[2] == .05
        @test p.qg[2] == .02
        @test p.pinj[2] ≈ .05-.2-.3p.lambda atol=1e-8
        @test p.qinj[2] ≈ .02-.1p.lambda atol=1e-8
        @test p.pg[1] ≈ p.pinj[1] atol=1e-8
        @test p.qg[1] ≈ p.qinj[1] atol=1e-8
    end
    pv = cpf_controlled_bus()
    pv.loads[1].qd = -.04
    d = LoadingDirection(pv;dp=[0.,-.2],dq=[0.,-.05])
    r = runcpf(pv,d;stop=:steps,max_steps=5)
    @test r.success
    for p in r.points
        @test p.qinj[2] ≈ p.qg[2]-.04-.05p.lambda atol=1e-8
    end
    unlimited = runcpf(cpf_controlled_bus();limits=:none,stop=:steps,max_steps=10)
    @test all(p -> p.control[2] == :voltage,unlimited.points)
    @test isempty(unlimited.events)
end

@testset "Selected recording does not retain solver histories" begin
    ps = cpf_controlled_bus()
    observed = Ref(0)
    observer(kind,w,value,branches) = (kind === :point && (observed[] += 1); nothing)
    trace = GradPower._cpf_trace_controls(ps;observer,retain_history=false)
    @test trace.status == :smooth_fold
    @test length(trace.points) == 1
    @test isempty(trace.events)
    @test observed[] > 1
    observed[] = 0
    smooth = GradPower._cpf_trace_smooth(ps;observer,retain_history=false,max_steps=5)
    @test length(smooth.points) == 1
    @test observed[] == 6
end
