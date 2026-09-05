"""Python uqgrid reference for ieee9 GENROU+TGOV1."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="ieee9_tgov1",
    raw="examples/ieee9_v33.raw",
    dyr="examples/ieee9bus_TGOV1.dyr",
    output="examples/refs/ieee9_tgov1.npz",
    generator_script="scripts/gen_ref_ieee9_tgov1.py",
    fault_bus=6,
)


if __name__ == "__main__":
    generate_reference(CASE)
