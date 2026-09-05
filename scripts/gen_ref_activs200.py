"""Generate the ACTIVSg200 reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="activs200",
    raw="uqgrid/data/ACTIVSg200.raw",
    dyr="uqgrid/data/ACTIVSg200.dyr",
    output="examples/refs/activs200.npz",
    generator_script="scripts/gen_ref_activs200.py",
    fault_bus_external=15,
    include_power_flow_data=True,
)


if __name__ == "__main__":
    generate_reference(CASE)
