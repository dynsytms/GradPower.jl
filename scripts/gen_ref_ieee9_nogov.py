"""Generate the ieee9 GENROU reference without governors."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="ieee9_nogov",
    raw="examples/ieee9_v33.raw",
    dyr="examples/ieee9bus.dyr",
    output="examples/refs/ieee9_nogov.npz",
    generator_script="scripts/gen_ref_ieee9_nogov.py",
    fault_bus=6,
)


if __name__ == "__main__":
    generate_reference(CASE)
