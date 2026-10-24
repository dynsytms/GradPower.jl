# Uses the small synthetic fixture from test_cpf.jl, with closed-form control
# boundaries. No external executable or case data is needed by these tests.
function cpf_controlled_bus(; voltage=1., pd=.2, qmin=-.2, qmax=.2)
    ps = cpf_two_bus()
    ps.buses[2].type = 2
    ps.buses[2].v0m = voltage
    ps.loads[1].pd = pd
    push!(ps.gens, Gen(2,"1",0.,0.,100.,true,qmax,qmin))
    return ps
end

function cpf_check_control_trace(ps, direction, trace; tolerance=3e-7)
    w = GradPower._cpf_workspace(ps,direction)
    m = length(w.active)
    for point in trace.points
        f = GradPower._cpf_residual!(zeros(length(point.state)),w,point.state,point.lambda)
        @test norm(f,Inf) <= tolerance
        @test minimum(point.state[m+1:2m]) > 0
        for (k,bus) in enumerate(w.controlled)
            q,v = point.state[2m+k], point.state[m+w.index[bus]]
            @test w.lower[k]-tolerance <= q <= w.upper[k]+tolerance
            if q < w.upper[k]-tolerance && q > w.lower[k]+tolerance
                @test abs(v-w.vset[bus]) <= tolerance
            elseif w.lower[k] != w.upper[k]
                q >= w.upper[k]-tolerance && @test v <= w.vset[bus]+tolerance
                q <= w.lower[k]+tolerance && @test v >= w.vset[bus]-tolerance
            end
        end
    end
    for event in trace.events
        @test event.error <= tolerance
        @test event.bus in w.buses[w.controlled]
    end
end

@testset "Exact reactive-limit continuation" begin
    ps = cpf_controlled_bus()
    d = LoadingDirection(ps)
    result = GradPower._cpf_trace_controls(ps,d)
    @test result.status == :smooth_fold
    @test length(result.events) == 1
    event = only(result.events)
    @test event.kind == :qmax
    @test event.bus == 205
    # At V=1, Qg=1-cos(theta). Qg=.2 implies P=.6,
    # hence lambda=.6/.2-1=2. After switching, Q injection=.2;
    # Pmax=sqrt(.25+.2), lambda=Pmax/.2-1.
    @test event.lambda ≈ 2. atol=1e-6
    @test last(result.points).lambda ≈ sqrt(.45)/.2-1 atol=1e-7
    @test event.incoming[end] > 0 && event.outgoing[end] > 0
    cpf_check_control_trace(ps,d,result)
    @test ps.buses[2].type == 2
    @test ps.buses[2].v0m == 1.
    @test ps.gens[2].qsch == 0.
    @test ps.network === nothing
    for scale in (.1,10.)
        scaled = GradPower._cpf_trace_controls(ps,d;control_scale=scale)
        @test scaled.status == :smooth_fold
        @test only(scaled.events).lambda ≈ event.lambda atol=2e-6
        @test last(scaled.points).lambda ≈ last(result.points).lambda atol=1e-7
    end
    # Recovery from an initially saturated base point as demand falls.
    recovery_ps = cpf_controlled_bus(;pd=.65)
    recovery_d = LoadingDirection(recovery_ps;dp=[0.,.65])
    recovery = GradPower._cpf_trace_controls(recovery_ps,recovery_d)
    @test recovery.status == :smooth_fold
    @test first(recovery.events).kind == :recovery
    @test first(recovery.events).lambda ≈ 1-.6/.65 atol=1e-6
    @test last(recovery.events).kind == :qmax
    cpf_check_control_trace(recovery_ps,recovery_d,recovery)
    # Lower saturation driven by increasing reactive injection.
    lower_ps = cpf_controlled_bus(;qmin=-.01,qmax=.3)
    lower_d = LoadingDirection(lower_ps;dp=zeros(2),dq=[0.,.1])
    lower = GradPower._cpf_trace_controls(lower_ps,lower_d;max_steps=30)
    @test lower.status == :max_steps
    @test only(lower.events).kind == :qmin
    @test only(lower.events).lambda ≈ (1-sqrt(1-.2^2)+.01)/.1 atol=1e-6
    cpf_check_control_trace(lower_ps,lower_d,lower)
    # Lower-bound recovery as a capacitive load is reduced.
    lower_recovery_ps = cpf_controlled_bus(;qmin=-.01,qmax=.3)
    lower_recovery_ps.loads[1].qd = .1
    lower_recovery_d = LoadingDirection(lower_recovery_ps;dp=zeros(2),dq=[0.,-.1])
    lower_recovery = GradPower._cpf_trace_controls(lower_recovery_ps,lower_recovery_d;max_steps=20)
    @test only(lower_recovery.events).kind == :recovery
    @test only(lower_recovery.events).lambda ≈ 1-(1-sqrt(1-.2^2)+.01)/.1 atol=2e-6
    cpf_check_control_trace(lower_recovery_ps,lower_recovery_d,lower_recovery)
    # A base solution exactly on a bound must depart on a feasible side.
    boundary_ps = cpf_controlled_bus(;pd=.6)
    boundary = GradPower._cpf_trace_controls(boundary_ps;max_steps=10)
    @test boundary.status == :max_steps
    @test count(e -> e.kind == :qmax,boundary.events) <= 1
    cpf_check_control_trace(boundary_ps,LoadingDirection(boundary_ps),boundary)
    # Fixed-Q and unlimited control do not introduce artificial events.
    fixed_ps = cpf_controlled_bus(;qmin=0.,qmax=0.)
    fixed_d = LoadingDirection(fixed_ps)
    fixed = GradPower._cpf_trace_controls(fixed_ps,fixed_d)
    @test fixed.status == :smooth_fold
    @test isempty(fixed.events)
    @test last(fixed.points).lambda ≈ 1.5 atol=1e-7
    cpf_check_control_trace(fixed_ps,fixed_d,fixed)
    unlimited = cpf_controlled_bus(;qmin=-Inf,qmax=Inf)
    free = GradPower._cpf_trace_controls(unlimited;max_steps=10)
    @test free.status == :max_steps
    @test isempty(free.events)
    @test all(p -> p.state[2] ≈ 1.,free.points)

    @test_throws ArgumentError GradPower._cpf_trace_controls(ps;event_tol=1e-12)
    @test_throws ArgumentError GradPower._cpf_trace_controls(ps;max_retries=-1)
    @test_throws ArgumentError GradPower._cpf_trace_controls(ps;stop=:unknown)
    fail = GradPower._cpf_trace_controls(ps;ds0=1.,ds_max=1.,max_retries=0,predictor_tol=1e-12)
    @test fail.status == :retry_limit
    @test length(fail.points) == 1
    @test isempty(fail.events)
end

@testset "Limit-induced loading maximum" begin
    ps = cpf_controlled_bus(;voltage=.6,qmin=-.4,qmax=-.1)
    ps.gens[2].qsch = -.2
    d = LoadingDirection(ps)
    result = GradPower._cpf_trace_controls(ps,d)
    @test result.status == :limit_induced_maximum
    @test length(result.events) == 1
    event = only(result.events)
    # With V=.6, Qmax=-.1 gives cos(theta)=(.36+.1)/.6;
    # P=sqrt(.6²-(.36+.1)²), on the lower-voltage side of
    # the fixed-Q fold: outgoing lambda decreases immediately.
    expected = sqrt(.6^2-(.36+.1)^2)/.2-1
    @test event.lambda ≈ expected atol=1e-6
    @test event.incoming[end] > 0
    @test event.outgoing[end] < 0
    @test last(result.points).lambda == event.lambda
    cpf_check_control_trace(ps,d,result)
    traced = GradPower._cpf_trace_controls(ps,d;stop=:steps,max_steps=18)
    @test traced.status == :max_steps
    @test last(traced.points).lambda < event.lambda
    cpf_check_control_trace(ps,d,traced)
end

@testset "Simultaneous reactive limits" begin
    ps = cpf_controlled_bus()
    push!(ps.buses, Bus(309,"second",2,230.,1.,0.))
    ps.busmap[309] = 3
    push!(ps.gens,Gen(3,"1",0.,0.,100.,true,.2,-.2))
    push!(ps.loads,Load(3,"1",.2,0.))
    push!(ps.branches,Branch(1,3,"1",0.,1.,0.,1.,0.))
    d = LoadingDirection(ps)
    result = GradPower._cpf_trace_controls(ps,d;max_steps=25)
    # Two independent identical branches have a multiple fold later; test the
    # simultaneous ordinary control exchange only, before that degeneracy.
    @test length(result.events) == 2
    @test Set(e.bus for e in result.events) == Set([205,309])
    @test all(e -> e.kind == :qmax,result.events)
    @test result.events[1].lambda == result.events[2].lambda
    @test result.events[1].lambda ≈ 2. atol=1e-6
    cpf_check_control_trace(ps,d,result)
end
