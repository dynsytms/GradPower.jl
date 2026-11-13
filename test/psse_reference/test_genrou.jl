# GENROU against PSS/E 35 (cross-check round 2, run 2026-11-12, 1/1920 s).
# Deck: IEEE 9-bus, three GENROU machines, no controllers (the control deck).
# Needs the staged fix set 01 (swing torque = Pm / speed).
# Ceilings are the error measured with the fixes, plus 1 point or 25 %, whichever is larger
# (voltage: twice the error or +0.001 pu); the angle error is also held to ANGLE_TARGET_PCT.
@testset "GENROU" begin
    # line 5-7 opened at 0.2 s. With the fixes: angle 0.50 %, speed 0.09 %. Without: angle 2.0 %, speed 18.0 %.
    check_case("i9_ctrl_trip57"; tol=Dict("angle_pct" => 2.0, "speed_pct" => 1.5, "v_err" => 0.002))
    # line 8-9 opened at 0.2 s. With the fixes: angle 0.81 %, speed 0.08 %. Without: angle 6.2 %, speed 17.9 %.
    check_case("i9_ctrl_trip89"; tol=Dict("angle_pct" => 2.0, "speed_pct" => 1.5, "v_err" => 0.002))
    # fault at bus 7, r = 0.01 pu, 0.2-0.3 s. With the fixes: angle 0.91 %, speed 0.32 %. Without: angle 1.1 %, speed 29.6 %.
    check_case("i9_ctrl_flt7"; tol=Dict("angle_pct" => 2.0, "speed_pct" => 1.5, "v_err" => 0.002))
end
