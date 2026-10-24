@testset "Continuation figures and animation" begin
    ps = PowerSystem(100.,
        [Bus(101,"reference",3,230.,1.,0.),Bus(205,"PV",2,230.,1.,0.)],
        [Gen(1,"1",0.,0.,100.,true),Gen(2,"1",0.,0.,100.,true,.2,-.2)],
        [Load(2,"1",.2,0.)],[Branch(1,2,"1",0.,1.,0.,1.,0.)],
        GradPower.Shunt[],Dict(101=>1,205=>2))
    r = runcpf(ps;buses=[205])
    original = [copy(p.vm) for p in r.points]
    mktempdir() do directory
        fig = plot_cpf(r;figure_kwargs=(size=(600,400),),axis_kwargs=(title="CPF",))
        @test only(_render_axes(fig)).title[] == "CPF"
        _test_render_files(fig,directory,"cpf")
        @test_throws ArgumentError plot_cpf(r;buses=[101])
        for fps in (0,1.5,true)
            @test_throws ArgumentError animate_cpf(joinpath(directory,"bad.gif"),r;framerate=fps)
        end
        short = runcpf(ps;buses=[205],stop=:steps,max_steps=3)
        path = joinpath(directory,"cpf.gif")
        @test animate_cpf(path,short;framerate=3,figure_kwargs=(size=(320,240),)) == path
        @test filesize(path)>256
        @test String(read(path)[1:6]) in ("GIF87a","GIF89a")
        one = runcpf(ps;buses=[205],max_steps=0)
        _test_render_files(plot_cpf(one),directory,"cpf-base")
        @test [p.vm for p in r.points] == original
    end
end
