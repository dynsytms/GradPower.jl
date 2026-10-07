"""Generate the IEEE-39 GENROU+governor reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="ieee39_gov",
    raw="examples/IEEE39.raw",
    dyr="examples/IEEE39_gov.dyr",
    output="examples/refs/ieee39_gov.npz",
    generator_script="scripts/gen_ref_ieee39_gov.py",
    fault_bus=15,
)


if __name__ == "__main__":
    generate_reference(CASE)
