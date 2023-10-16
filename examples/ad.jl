using GradPower
using SparseArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

# parse
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# compute Jacobian-vector products wrt parameters
pp_nom = copy(dprob.pvec) # nominal evaluation point
vdir = ones(length(pp_nom)) # direction of perturbation

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

function jvp_trans_fd(pp, v)
    Jfd = FiniteDiff.finite_difference_jacobian(rhs, pp_nom)
    return transpose(Jfd)*v
end

println("Jacobian vector product")
jvpf = jvp_fd(pp_nom, vdir)
jvp_ad = zeros(length(dprob.zvec))
GradPower.jacp_vec!(jvp_ad, vdir, dprob.zvec, dprob.uvec, pp_nom, sys, full_jac=false)
# compute difference
error = norm(jvpf - jvp_ad)/norm(jvpf)
println("error relative: ", error)

println("Jacobian^T vector product")
vdir = zeros(length(dprob.zvec))
vdir[1] = 1.0
jvpf_t = jvp_trans_fd(pp_nom, vdir)
