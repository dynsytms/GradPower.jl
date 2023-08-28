using Plots
using Revise    
using GradPower
using SparseArrays

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"

raw_file = "examples/ieee9_v33.raw"
dyr_file = "examples/ieee9bus.dyr"

#raw_file = "examples/ACTIVSg2000.raw"
#dyr_file = "examples/ACTIVSg2000.dyr"

# parse
devices = GradPower.read_psse_dyr(dyr_file)
psse_devices = GradPower.create_device_vector(devices)
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=false);

# dynamic simulation
tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)

# add event
#event = GradPower.add_event!(sys, GradPower.ContingencyEvent(2, 0.2, 0.2, 0.3))
#tvec, traj = GradPower.integrate!(dprob, sys, tfinal)

Jsp = GradPower.preallocate_jacobian(sys)
