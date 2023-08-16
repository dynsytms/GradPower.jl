using Revise    
using GradPower

raw_file = "examples/2bus.raw"
dyr_file = "examples/2bus.dyr"


# parse
devices = GradPower.read_psse_dyr(dyr_file)
psse_devices = GradPower.create_device_vector(devices)
raw = GradPower.read_psse_raw(raw_file)
sys = GradPower.raw_to_grad(raw)
psd = GradPower.PowerSystemDynamics(dyr_file)
GradPower.set_dynamics!(sys, psd)

# power flow
GradPower.build_network!(sys)
GradPower.runpf!(sys, verbose=true);

# dynamic simulation
tfinal = 1.0
dprob = GradPower.DynamicProblem(sys)
GradPower.initialize_dynamics!(dprob, sys)
# perturb state manually
dprob.zvec[5] += 0.01
GradPower.integrate!(dprob, sys, tfinal)
