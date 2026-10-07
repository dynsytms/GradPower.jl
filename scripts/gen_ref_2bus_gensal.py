"""Generate the 2-bus GENSAL reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_gensal",
    raw="examples/2bus.raw",
    dyr="examples/2bus_GENSAL.dyr",
    output="examples/refs/2bus_gensal.npz",
    generator_script="scripts/gen_ref_2bus_gensal.py",
)


if __name__ == "__main__":
    generate_reference(CASE)
