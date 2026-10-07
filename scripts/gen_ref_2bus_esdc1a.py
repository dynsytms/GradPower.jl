"""Generate the 2-bus GENROU+ESDC1A reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_esdc1a",
    raw="examples/2bus.raw",
    dyr="examples/2bus_ESDC1A.dyr",
    output="examples/refs/2bus_esdc1a.npz",
    generator_script="scripts/gen_ref_2bus_esdc1a.py",
)


if __name__ == "__main__":
    generate_reference(CASE)
