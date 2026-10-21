using GradPower
using MadNLP
using NLPModels

ex = @__DIR__
sys = from_psse(joinpath(ex, "2bus.raw"), joinpath(ex, "2bus_IEESGO.dyr"))
build_network!(sys)
runpf!(sys; verbose=false)
dp = DynamicProblem(sys)
initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(2, 0.2, 0.2, 0.3))

lvar, uvar = copy(dp.pvec), copy(dp.pvec)
for dev in sys.dynamic.devices
    dev.dtype isa GradPower.Genrou || continue
    hidx = dev.par_ptr + findfirst(==("H"), get_param_names(dev.dtype)) - 1
    lvar[hidx] *= 0.8
    uvar[hidx] *= 1.2
end

# Forward objective evaluation only: grad! and optimization remain unavailable.
nlp = GradPower.DynamicNLP(sys, dp, 1.0, lvar, uvar)
println("Integrated squared speed deviation: ", NLPModels.obj(nlp, dp.pvec))
