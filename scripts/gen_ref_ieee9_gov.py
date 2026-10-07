"""Generate the ieee9 GENROU+IEESGO reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="ieee9_gov",
    raw="examples/ieee9_v33.raw",
    dyr="examples/ieee9bus_gov.dyr",
    output="examples/refs/ieee9_gov.npz",
    generator_script="scripts/gen_ref_ieee9_gov.py",
    fault_bus=6,
)


if __name__ == "__main__":
    generate_reference(CASE)
