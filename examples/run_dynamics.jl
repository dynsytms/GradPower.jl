using Plots
using Revise    
using GradPower
using SparseArrays
using FiniteDiff

using Profile
using PProf

raw_file = "data/2bus_33.raw"
dyr_file = "data/2bus.dyr"

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

#raw_file = "examples/ieee9_v33.raw"
#dyr_file = "examples/ieee9bus.dyr"

#raw_file = "examples/ACTIVSg2000.raw"
#dyr_file = "examples/ACTIVSg2000.dyr"

# parse
sys = GradPower.from_psse(raw_file, dyr_file)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))

# integrate dynamics
tfinal = 1.0
tvec, traj = GradPower.integrate!(dprob, sys, tfinal)
print(traj[:,end])
