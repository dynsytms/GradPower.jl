@testset "Analytic second derivatives and fold certificates" begin
    ps = cpf_controlled_bus()
    ps.branches[1].r = .07
    ps.branches[1].tap = 1.1
    ps.branches[1].shift = 13.
    w = GradPower._cpf_workspace(ps,LoadingDirection(ps))
    x,d = [.13,.94,.12],[.3,-.2,.4]
    h = 1e-5
    for branches in ([:free],[:lower],[:upper])
        jp = copy(GradPower._cpf_jacobian!(w,x+h*d;branches))
        jm = copy(GradPower._cpf_jacobian!(w,x-h*d;branches))
        @test GradPower._cpf_second_direction(w,x,d) ≈ (jp-jm)*d/(2h) atol=1e-9
    end
    certificate(J,b,h) = GradPower._cpf_fold_certificate(J,b,_ -> h)
    # Normal forms: x^2-lambda, x^4-lambda, and a nontransverse parameter.
    @test certificate(zeros(1,1),[-1.],[2.]).status == :simple_fold
    @test certificate(zeros(1,1),[-1.],[0.]).status == :fold_curvature_failure
    @test certificate(zeros(1,1),[0.],[2.]).status == :fold_transversality_failure
    @test certificate(zeros(2,2),[-1.,-1.],[2.,2.]).status == :fold_rank_failure
    @test certificate(ones(1,1),[-1.],[2.]).status == :fold_rank_failure
    @test certificate(zeros(1,1),[-1.],[1e-12]).status == :fold_curvature_failure
    ps = cpf_two_bus()
    r = GradPower._cpf_trace_smooth(ps)
    @test r.diagnostic.status == :simple_fold
    @test r.diagnostic.nullity == 1
    @test r.diagnostic.transversality > 1e-6
    @test r.diagnostic.curvature > 1e-6
    # Two identical PQ branches have two independent null vectors at the nose.
    push!(ps.buses,Bus(309,"second",1,230.,1.,0.))
    ps.busmap[309] = 3
    push!(ps.loads,Load(3,"1",.2,0.))
    push!(ps.branches,Branch(1,3,"1",0.,1.,0.,1.,0.))
    w = GradPower._cpf_workspace(ps,LoadingDirection(ps);limits=:none)
    x = [-pi/4,-pi/4,inv(sqrt(2)),inv(sqrt(2))]
    p = GradPower.CPFPoint(x,1.5,zeros(5),0.,0)
    @test GradPower._cpf_verify_fold(w,p).status == :fold_rank_failure
end

@testset "Interior exit and reentry on a smooth regime" begin
    # On V=1, Qg=1-sqrt(1-P^2). Starting at P=.2 and reducing
    # demand through zero gives a narrow Qmin violation near P=0 even
    # though both ends of the trial step are strictly feasible.
    ps = cpf_controlled_bus(;qmin=1e-5,qmax=Inf)
    d = LoadingDirection(ps;dp=[0.,.4])
    w = GradPower._cpf_workspace(ps,d)
    weights = GradPower._cpf_weights(w)
    x = [-asin(.2),1.,1-sqrt(.96)]
    reference = [0.,0.,0.,1.]
    p = GradPower._cpf_point(w,vcat(x,0.),reference,weights,0,0.;branches=[:free])
    step = 1.1
    sol = GradPower._cpf_correct(w,vcat(x,0.)+step*p.tangent,weights.*p.tangent;
                                tol=1e-11,max_newton=12,branches=[:free])
    @test sol !== nothing
    z,it,residual = sol
    q = GradPower._cpf_point(w,z,p.tangent,weights,it,residual;branches=[:free])
    guards = GradPower._cpf_guards(w,[:free])
    @test GradPower._cpf_margin(w,p.state,only(guards),[:free]) > 0
    @test GradPower._cpf_margin(w,q.state,only(guards),[:free]) > 0
    scan = GradPower._cpf_scan_controls(w,p,q,step,guards,[:free],weights;
                                       tol=1e-11,event_tol=1e-8,max_newton=12)
    @test scan.status == :ok
    @test length(scan.hits) == 1
    hit = only(scan.hits)
    located = GradPower._cpf_locate_control(w,p,hit.hi,hit.guard,[:free],weights;
        tol=1e-11,event_tol=1e-8,max_newton=12,lower_step=hit.lo)
    @test located !== nothing
    expected = (.2-sqrt(2e-5-1e-10))/.4
    @test located[2][end] ≈ expected atol=1e-8
    # Integration: a deliberately coarse accepted predictor must still stop at
    # the first activation, not silently jump to the recovered free segment.
    trace = GradPower._cpf_trace_controls(ps,d;ds0=step,ds_max=step,
        predictor_tol=.2,max_steps=1,tol=1e-11,event_tol=1e-8)
    @test length(trace.events) == 1
    @test only(trace.events).kind == :qmin
    @test only(trace.events).lambda ≈ expected atol=1e-8
    cpf_check_control_trace(ps,d,trace)
    # The limiting double root touches Qmin without crossing. First-order
    # event logic must not silently declare an ordinary exchange.
    ps.gens[2].qmin = 0.
    wtouch = GradPower._cpf_workspace(ps,d)
    touch = GradPower._cpf_scan_controls(wtouch,p,q,step,guards,[:free],weights;
                                        tol=1e-11,event_tol=1e-8,max_newton=12)
    @test touch.status == :control_guard_grazing
end

@testset "Event bracket preserves a feasible left endpoint" begin
    # The 3/4-step guard is negative by only half event_tol. Detection waits
    # until the next sample, but localization must bracket with an earlier
    # truly feasible point, not the slightly negative 3/4-step sample.
    ps = cpf_controlled_bus(;qmax=Inf)
    direction = LoadingDirection(ps)
    w = GradPower._cpf_workspace(ps,direction)
    weights = GradPower._cpf_weights(w)
    x = [-asin(.2),1.,1-sqrt(.96)]
    previous = GradPower._cpf_point(w,vcat(x,0.),[0.,0.,0.,1.],weights,0,0.;branches=[:free])
    step = .1
    function at_step(s)
        z,it,residual = GradPower._cpf_correct(w,vcat(x,0.)+s*previous.tangent,
            weights.*previous.tangent;tol=1e-12,max_newton=12,branches=[:free])
        return GradPower._cpf_point(w,z,previous.tangent,weights,it,residual;branches=[:free])
    end
    near = at_step(.75step)
    ps.gens[2].qmax = near.state[3]-5e-8
    w = GradPower._cpf_workspace(ps,direction)
    guard = only(filter(g -> g[2] === :upper,GradPower._cpf_guards(w,[:free])))
    @test -1e-7 < GradPower._cpf_margin(w,near.state,guard,[:free]) < 0
    scan = GradPower._cpf_scan_controls(w,previous,at_step(step),step,[guard],[:free],weights;
        tol=1e-11,event_tol=1e-7,max_newton=12)
    @test scan.status == :ok
    hit = only(scan.hits)
    @test GradPower._cpf_margin(w,at_step(hit.lo).state,guard,[:free]) >= 0
    @test GradPower._cpf_margin(w,at_step(hit.hi).state,guard,[:free]) < -1e-7
    located = GradPower._cpf_locate_control(w,previous,hit.hi,guard,[:free],weights;
        tol=1e-11,event_tol=1e-7,max_newton=12,lower_step=hit.lo)
    @test located !== nothing
    if located !== nothing
        @test located[5] <= 1e-11
        @test located[1] < .75step
    end
end

@testset "Coincident and closely separated fold/control events" begin
    # At V=1, Qmax=.5 places the outgoing PQ fold exactly at the
    # activation point: V^2=Qmax+.5. Either side gives a different outcome.
    for offset in (-1e-3,0.,1e-3)
        ps = cpf_controlled_bus(;qmax=.5+offset)
        r = GradPower._cpf_trace_controls(ps;tol=1e-11,event_tol=1e-8)
        if offset < 0
            @test r.status == :smooth_fold
            @test r.diagnostic.status == :simple_fold
            @test length(r.events) == 1
            @test last(r.points).lambda ≈ sqrt(.75+offset)/.2-1 atol=1e-8
        elseif offset > 0
            @test r.status == :limit_induced_maximum
            @test length(r.events) == 1
        else
            @test r.status == :fold_control_coincidence
            @test !isempty(keys(r.diagnostic))
        end
        cpf_check_control_trace(ps,LoadingDirection(ps),r)
    end
    # Incoming PV fold at P=1,Qg=1. Controls on either side of this
    # fold must be ordered in section coordinate, not by nearly equal lambda.
    for offset in (-1e-3,0.,1e-3), ds in (.03,.1)
        ps = cpf_controlled_bus(;qmax=1+offset)
        r = GradPower._cpf_trace_controls(ps;ds0=ds,tol=1e-11,event_tol=1e-8)
        expected_status = offset < 0 ? :limit_induced_maximum :
                          offset > 0 ? :smooth_fold : :fold_control_coincidence
        @test r.status == expected_status
        if offset < 0
            @test only(r.events).lambda ≈ sqrt(1-offset^2)/.2-1 atol=1e-8
        elseif offset > 0
            @test isempty(r.events)
            @test last(r.points).lambda ≈ 4. atol=1e-8
        end
        cpf_check_control_trace(ps,LoadingDirection(ps),r)
    end
end

@testset "Control-cone degeneracy diagnostics" begin
    ps = cpf_controlled_bus(;pd=0.,qmin=0.,qmax=.5)
    d = LoadingDirection(ps;dp=[0.,-.2])
    w = GradPower._cpf_workspace(ps,d)
    weights = GradPower._cpf_weights(w)
    x = [0.,1.,0.]
    p = GradPower._cpf_point(w,vcat(x,0.),[0.,0.,0.,1.],weights,0,0.;branches=[:free])
    # At P=Q=0 the lower-bound departure has zero first-order voltage
    # derivative: a tangent cone cannot decide feasibility to second order.
    dep = GradPower._cpf_depart(w,p,[:free],[(1,:lower,:qmin)],weights;event_tol=1e-8)
    @test dep.status == :control_cone_grazing
    @test !isempty(dep.visited)
    dep = GradPower._cpf_depart(w,p,[:free],[(1,:lower,:qmin)],weights;
                               event_tol=1e-8,max_iterations=0)
    @test dep.status == :control_cone_iteration_limit
    # An orthogonal orientation row is not a rank defect. The nullspace
    # fallback still identifies the grazing control cone in this example.
    p = GradPower.CPFPoint(x,0.,zeros(4),0.,0)
    dep = GradPower._cpf_depart(w,p,[:free],[(1,:lower,:qmin)],weights;event_tol=1e-8)
    @test dep.status == :control_cone_grazing
    # A genuine rank defect: fixed-Q fold with a zero parameter direction.
    ps = cpf_controlled_bus(;qmin=0.,qmax=0.)
    w = GradPower._cpf_workspace(ps,LoadingDirection(ps;dp=zeros(2)))
    p = GradPower.CPFPoint([-pi/4,inv(sqrt(2)),0.],0.,[0.,0.,0.,1.],0.,0)
    dep = GradPower._cpf_depart(w,p,[:free],[(1,:fixed,:qmin)],GradPower._cpf_weights(w);event_tol=1e-8)
    @test dep.status == :control_cone_singular
    @test dep.nullity == 2
    # Returning to the incoming active set is a cycle, not a new departure.
    ps = cpf_controlled_bus()
    w = GradPower._cpf_workspace(ps,LoadingDirection(ps))
    x = [-asin(.2),1.,1-sqrt(.96)]
    weights = GradPower._cpf_weights(w)
    p = GradPower._cpf_point(w,vcat(x,0.),[0.,0.,0.,1.],weights,0,0.;branches=[:free])
    dep = GradPower._cpf_depart(w,p,[:free],Tuple{Int,Symbol,Symbol}[],weights;event_tol=1e-8)
    @test dep.status == :control_cone_cycle

    # Coupled three-bus example with a genuine two-active-set cycle.
    # Dispatch is constructed from YV so the boundary state is an exact PF
    # solution; both incoming Q derivatives hit their upper limits.
    ps = cpf_controlled_bus()
    push!(ps.buses,Bus(309,"second",2,230.,1.,0.)); ps.busmap[309] = 3
    push!(ps.gens,Gen(3,"1",0.,0.,100.,true,1.,-1.))
    push!(ps.branches,Branch(1,3,"1",0.,1.,0.,1.,0.))
    push!(ps.branches,Branch(2,3,"1",0.,.5,0.,1.,0.))
    x = [1.2974366647847453,-.13985760022333063,
         .5387275010363611,1.173537694191047,0.,0.]
    ps.buses[2].v0m,ps.buses[3].v0m = x[3:4]
    v = [1.;x[3:4].*cis.(x[1:2])]
    s = v.*conj.(GradPower.create_ybus_complex(ps)*v)
    x[5:6] .= imag.(s[2:3])
    empty!(ps.loads)
    for k in 1:2
        ps.gens[k+1].psch = real(s[k+1])
        ps.gens[k+1].qmax = imag(s[k+1])
        ps.gens[k+1].qmin = imag(s[k+1])-2
    end
    d = LoadingDirection(ps;dp=[0.,.36243512058371474,-.7216246683771064],
                         dq=[0.,-.46175652363746816,-1.0533729242903629])
    w = GradPower._cpf_workspace(ps,d)
    weights = GradPower._cpf_weights(w)
    @test norm(GradPower._cpf_residual!(zeros(6),w,x,0.),Inf) < 1e-12
    p = GradPower._cpf_point(w,vcat(x,0.),[zeros(6);1.],weights,0,0.;branches=[:free,:free])
    @test all(p.tangent[5:6] .> 0)
    dep = GradPower._cpf_depart(w,p,[:free,:free],[(1,:upper,:qmax),(2,:upper,:qmax)],weights;event_tol=1e-8)
    @test dep.status == :control_cone_cycle
    @test length(dep.visited) == 2
end
