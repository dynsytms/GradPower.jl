using Test
using GradPower

@testset "device state metadata" begin
    devices = GradPower.AbstractDeviceType[
        GradPower.GenericGenerator(1, "1"),
        GradPower.IEESGO(1, "1", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 2.0, 0.0),
        GradPower.TGOV1(1, "1", 0.05, 1.0, 2.0, 0.0, 0.5, 1.0, 0.0),
        GradPower.SEXS(1, "1", 0.1, 1.0, 10.0, 1.0, -5.0, 5.0),
        GradPower.ESDC1A(1, "1", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         0.0, 0.0, 0.0),
        GradPower.IEEEST(1, "1", 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         999.0, -999.0, 999.0, -999.0),
        GradPower.ZIPLoad(1, "1", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                          1.0, 0.0, 0.0),
        GradPower.StaticGenerator(Int64(1), Int64(2), Int64[1], 1.0, 0.0),
    ]

    for device in devices
        @test length(get_diff_names(device)) == device.diff_size
        @test length(get_alg_names(device)) == device.alg_size
        @test length(get_param_names(device)) == device.par_size
        @test length(unique(get_diff_names(device))) == device.diff_size
        @test length(unique(get_alg_names(device))) == device.alg_size
    end

    genrou = devices[1]
    @test get_diff_names(genrou) == ["e_qp", "e_dp", "phi_1d", "phi_2q", "w", "delta"]
    @test get_alg_names(genrou) == ["v_q", "v_d", "i_q", "i_d"]
end
