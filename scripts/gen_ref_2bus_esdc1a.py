"""Python uqgrid reference for 2bus GENROU+ESDC1A."""
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "uqgrid"))

from uqgrid import IntegrationConfig, add_dyr, integrate_system, load_psse

RAW = os.path.join(REPO, "examples", "2bus.raw")
DYR = os.path.join(REPO, "examples", "2bus_ESDC1A.dyr")
OUT = os.path.join(REPO, "examples", "refs", "2bus_esdc1a.npz")
FAULT_BUS, RFAULT, TON, TOFF, DT, TEND, ALPHA = 0, 0.02, 0.2, 0.3, 1/120, 5.0, 0.5

psys = load_psse(raw_filename=RAW)
add_dyr(psys, DYR)
psys.add_busfault(FAULT_BUS, RFAULT)
psys.createYbusComplex()
psys.set_load_parameters(np.full(psys.nloads, ALPHA))
config = IntegrationConfig(tend=TEND, dt=DT, ton=TON, toff=TOFF,
    power_injection=False, verbose=False, comp_sens=False, petsc=False)
results = integrate_system(psys, config)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, tvec=results["tvec"], history=results["history"],
    speed_idx=np.array(psys.genspeed_idx_set()),
    n_diff=psys.num_dof_dif, n_alg=psys.num_dof_alg, n_bus=psys.nbuses,
    fault_bus=FAULT_BUS, rfault=RFAULT, ton=TON, toff=TOFF, dt=DT, tend=TEND, zipload_alpha=ALPHA)
print(f"Saved: {OUT} history.shape={results['history'].shape} n_diff={psys.num_dof_dif} n_alg={psys.num_dof_alg}")
