# GENROU against PSS/E 35 (cross-check round 2, run 2026-11-12, 1/1920 s).
# Deck: IEEE 9-bus, three GENROU machines with no exciter or governor.
# The harness runs mech_input = :power (PSS/E's torque = Pm / speed) with
# Crank-Nicolson. With the default :torque form the speed error is 18-30 %.
# Ceilings: about 4x the measured error, at least angle 0.5 %, speed 0.25 %, voltage 5e-4 pu.
@testset "GENROU" begin
    tol = Dict("angle_pct" => 0.5, "speed_pct" => 0.25, "v_err" => 5e-4)
    # line 5-7 opened at 0.2 s. Measured: angle 0.04 %, speed 0.01 %, voltage 2e-5 pu.
    check_case("i9_ctrl_trip57"; tol)
    # line 8-9 opened at 0.2 s. Measured: angle 0.12 %, speed 0.02 %, voltage 3e-5 pu.
    check_case("i9_ctrl_trip89"; tol)
    # fault at bus 7, r = 0.01 pu, 0.2-0.3 s. Measured: angle 0.11 %, speed 0.06 %, voltage 3e-5 pu.
    check_case("i9_ctrl_flt7"; tol)
end
