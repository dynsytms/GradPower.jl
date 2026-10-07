"""Python uqgrid reference for ieee9 GENROU+SEXS."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="ieee9_sexs",
    raw="examples/ieee9_v33.raw",
    dyr="examples/ieee9bus_SEXS.dyr",
    output="examples/refs/ieee9_sexs.npz",
    generator_script="scripts/gen_ref_ieee9_sexs.py",
    fault_bus=6,
)


if __name__ == "__main__":
    generate_reference(CASE)
