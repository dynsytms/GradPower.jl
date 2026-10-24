# Step/tolerance reproducibility on bundled and analytic cases. Independent
# local comparisons are documented separately; no external programs or data
# are required for this regression suite.
@testset "Continuation step and tolerance refinement" begin
    case9 = create_case9()
    for gen in case9.gens
        gen.qmin = -.5
        gen.qmax = .5
    end
    recovery = cpf_controlled_bus(;pd=.65)
    lower = cpf_controlled_bus(;qmin=-.01,qmax=.3)
    lower.loads[1].qd = .1
    kink = cpf_controlled_bus(;voltage=.6,qmin=-.4,qmax=-.1)
    kink.gens[2].qsch = -.2
    cases = [
        (case9, LoadingDirection(case9), :smooth_fold, [(2,:qmax),(3,:qmax)]),
        (recovery, LoadingDirection(recovery;dp=[0.,.65]), :smooth_fold,
         [(205,:recovery),(205,:qmax)]),
        (lower, LoadingDirection(lower;dp=zeros(2),dq=[0.,-.1]), :smooth_fold,
         [(205,:recovery),(205,:qmax)]),
        (kink, LoadingDirection(kink), :limit_induced_maximum, [(205,:qmax)]),
    ]
    for (ps,direction,status,order) in cases
        traces = [GradPower._cpf_trace_controls(ps,direction;
                    ds0=ds,ds_max=4ds,tol=tol,event_tol=event_tol,max_steps=500)
                  for (ds,tol,event_tol) in ((.05,1e-9,1e-7),(.015,1e-9,1e-7),(.025,1e-11,1e-9))]
        reference = last(traces)
        for trace in traces
            @test trace.status == status
            @test [(e.bus,e.kind) for e in trace.events] == order
            @test last(trace.points).lambda ≈ last(reference.points).lambda atol=2e-8 rtol=0
            @test last(trace.points).state ≈ last(reference.points).state atol=1e-7 rtol=0
            if length(trace.events) == length(reference.events)
                for (event,ref) in zip(trace.events,reference.events)
                    @test event.lambda ≈ ref.lambda atol=2e-8 rtol=0
                    @test event.state ≈ ref.state atol=1e-7 rtol=0
                end
            end
            cpf_check_control_trace(ps,direction,trace)
        end
    end
    # Closed-form margins are independent of continuation implementation.
    upper_nose = (sqrt(.45)+.65)/.65
    @test last(GradPower._cpf_trace_controls(recovery,LoadingDirection(recovery;dp=[0.,.65])).points).lambda ≈ upper_nose atol=1e-8
    @test last(GradPower._cpf_trace_controls(lower,LoadingDirection(lower;dp=zeros(2),dq=[0.,-.1])).points).lambda ≈ 6.1 atol=1e-8
end
