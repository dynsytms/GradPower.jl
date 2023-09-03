using Plots
using Revise    
using GradPower
using SparseArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using BenchmarkTools

using Profile
using PProf

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

raw_file = "examples/ieee9_v33.raw"
dyr_file = "examples/ieee9bus.dyr"

#raw_file = "examples/ACTIVSg2000.raw"
#dyr_file = "examples/ACTIVSg2000.dyr"

# parse
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
tfinal = 3/120.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# compute Jacobian-vector products wrt parameters
pp_nom = copy(dprob.pvec) # nominal evaluation point
vdir = ones(length(pp_nom)) # direction of perturbation
#vdir = zeros(length(pp_nom)) # direction of perturbation
#vdir[2] = 1.0

# compute using finite differences
function rhs(pp_nom)
    f = similar(pp_nom, length(dprob.zvec))
    GradPower.rhs_fun!(f, dprob.zvec, dprob.uvec, pp_nom, sys)
    return f
end

function jvp_fd(pp, v)
    Jfd = FiniteDiff.finite_difference_jacobian(rhs, pp_nom)
    return Jfd*v
end

println("Computing Jacobian-vector product using finite differences...")
#jvp_fd(pp_nom, vdir)
jvpf = jvp_fd(pp_nom, vdir)

println("Computing Jacobian-vector product using ForwardDiff...")
jvp_ad = zeros(length(dprob.zvec))
GradPower.jacp_vec!(jvp_ad, vdir, dprob.zvec, dprob.uvec, pp_nom, sys, full_jac=false)
jvp_ad = zeros(length(dprob.zvec))
@time GradPower.jacp_vec!(jvp_ad, vdir, dprob.zvec, dprob.uvec, pp_nom, sys)
# compute difference
error = norm(jvpf - jvp_ad)/norm(jvpf)
println("error relative: ", error)
