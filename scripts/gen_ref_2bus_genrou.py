"""Generate the 2-bus GENROU reference."""

from reference_common import ReferenceCase, generate_reference


CASE = ReferenceCase(
    name="2bus_genrou",
    raw="examples/2bus.raw",
    dyr="examples/2bus.dyr",
    output="examples/refs/2bus_genrou.npz",
    generator_script="scripts/gen_ref_2bus_genrou.py",
    ton=0.1,
    toff=0.2,
    tend=2.0,
    zipload_alpha=1.0,
)


if __name__ == "__main__":
    generate_reference(CASE)
