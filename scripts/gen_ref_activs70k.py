"""Generate the ACTIVSg70k reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="activs70k",
    raw="uqgrid/data/ACTIVSg70k.raw",
    dyr="uqgrid/benchmarks/phase3_generated_dyrs/ACTIVSg70k_genrou_only.dyr",
    output="examples/refs/activs70k.npz",
    generator_script="scripts/gen_ref_activs70k.py",
    fault_bus_external=3,
    tend=1.0,
    include_power_flow_data=True,
)


if __name__ == "__main__":
    generate_reference(CASE)
