"""Generate the 2-bus GENROU+IEESGO reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_ieesgo",
    raw="examples/2bus.raw",
    dyr="examples/2bus_IEESGO.dyr",
    output="examples/refs/2bus_ieesgo.npz",
    generator_script="scripts/gen_ref_2bus_ieesgo.py",
    ton=0.1,
    toff=0.2,
    tend=2.0,
    zipload_alpha=1.0,
)


if __name__ == "__main__":
    generate_reference(CASE)
