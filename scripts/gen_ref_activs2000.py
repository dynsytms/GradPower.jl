"""Generate the ACTIVSg2000 reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="activs2000",
    raw="uqgrid/data/ACTIVSg2000.raw",
    dyr="uqgrid/data/ACTIVSg2000_genrou_only.dyr",
    output="examples/refs/activs2000.npz",
    generator_script="scripts/gen_ref_activs2000.py",
    fault_bus_external=1001,
    tend=2.0,
    include_power_flow_data=True,
)


if __name__ == "__main__":
    generate_reference(CASE)
