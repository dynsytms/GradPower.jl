"""Generate the 2-bus GENROU+TGOV1 reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_tgov1",
    raw="examples/2bus.raw",
    dyr="examples/2bus_TGOV1.dyr",
    output="examples/refs/2bus_tgov1.npz",
    generator_script="scripts/gen_ref_2bus_tgov1.py",
)


if __name__ == "__main__":
    generate_reference(CASE)
