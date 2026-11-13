# GENSAL against PSS/E 35 (cross-check round 2, run 2026-11-12, 1/1920 s).
# Deck: IEEE 9-bus, GENSAL on machines 1 and 3 (rows 3105 and 3106 of ACTIVSg2000_dynamics.dyr), GENROU on 2, no controllers.
# Needs the staged fix sets 01 (swing torque = Pm / speed) and 02 (GENSAL x'q = xq).
# Ceilings are the error measured with the fixes, plus 1 point or 25 %, whichever is larger
# (voltage: twice the error or +0.001 pu); the angle error is also held to ANGLE_TARGET_PCT.
# The remaining speed error (3.6-6.8 %) comes from GENROU's q-axis saturation term, which GENSAL
# (run as GENROU with x'q = xq) inherits; PSS/E's GENSAL saturates the d axis only. Open.
@testset "GENSAL" begin
    # line 5-7 opened at 0.2 s. With the fixes: angle 1.32 %, speed 3.58 %. Without: angle 25.3 %, speed 23.2 %.
    check_case("i9_gensal_trip57"; tol=Dict("angle_pct" => 2.5, "speed_pct" => 5.0, "v_err" => 0.01))
    # fault at bus 7, r = 0.01 pu, 0.2-0.3 s. With the fixes: angle 3.32 %, speed 6.81 %. Without: angle 74.3 %, speed 34.8 %.
    check_case("i9_gensal_flt7"; tol=Dict("angle_pct" => 4.5, "speed_pct" => 9.0, "v_err" => 0.005))
end
