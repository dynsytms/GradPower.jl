using GradPower

"""Small lossless network with an analytically checkable voltage nose."""
function continuation_example(; voltage=1.0, qmin=-0.2, qmax=0.2)
    buses = [Bus(101,"Reference",3,230.,1.,0.),
             Bus(205,"Controlled load",2,230.,voltage,0.)]
    gens = [Gen(1,"1",0.,0.,100.,true),
            Gen(2,"1",0.,0.,100.,true,qmax,qmin)]
    loads = [Load(2,"1",.2,0.)]
    branches = [Branch(1,2,"1",0.,1.,0.,1.,0.)]
    return PowerSystem(100.,buses,gens,loads,branches,GradPower.Shunt[],Dict(101=>1,205=>2))
end

# Default direction increases existing nonreference constant-power demands:
# P_d(lambda) = 0.2(1 + lambda). Generator reactive limits remain fixed.
system = continuation_example()
limited = runcpf(system; buses=[205])
@assert limited.status == :smooth_fold
@assert isapprox(limited.maximum.lambda, sqrt(.45)/.2-1; atol=1e-8)
println("Reactive-limited maximum: lambda = ",limited.maximum.lambda)
for event in limited.events
    println("Bus ",event.bus," ",event.kind," at lambda = ",event.point.lambda)
end

# A step-budget trace may pass the maximum and follow the lower-voltage branch.
# It deliberately does not claim a verified maximum (maximum === nothing).
branch = runcpf(system; buses=[205], stop=:steps, max_steps=55)
@assert branch.maximum === nothing

# Optional figures (run using the docs or plotting environment):
# using CairoMakie
# save("continuation.png", plot_cpf(branch))
# animate_cpf("continuation.gif", branch; framerate=12)
