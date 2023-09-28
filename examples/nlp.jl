using Revise
using GradPower
using MadNLP
using NLPModels

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

#raw_file = "examples/ieee9_v33.raw"
#dyr_file = "examples/ieee9bus.dyr"

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
dt = 1.0/120.0
tfinal = 2.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# integrate dynamics
event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))

# build nlp
lvar = copy(dprob.pvec)
uvar = copy(dprob.pvec)

# inertia
lvar[7] = 0.8*lvar[7]
uvar[7] = 1.2*uvar[7]

nlp = GradPower.DynamicNLP(sys, dprob, tfinal, lvar, uvar)

# solve problem
solver = MadNLP.MadNLPSolver(nlp;
    print_level=MadNLP.INFO,
    kkt_system=MadNLP.DENSE_KKT_SYSTEM,
    hessian_approximation=MadNLP.DENSE_BFGS,
    linear_solver=LapackCPUSolver,
)
MadNLP.solve!(solver)
