"""Generate the 2-bus GENROU+SEXS reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_sexs",
    raw="examples/2bus.raw",
    dyr="examples/2bus_SEXS.dyr",
    output="examples/refs/2bus_sexs.npz",
    generator_script="scripts/gen_ref_2bus_sexs.py",
)


if __name__ == "__main__":
    generate_reference(CASE)
