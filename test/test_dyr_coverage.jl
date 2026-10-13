using Test
using GradPower

const COVERAGE_DATA = joinpath(@__DIR__, "..", "uqgrid", "data")

@testset "DYR capability coverage" begin
    @test GradPower.native_dyr_models() == Set(["GENROU", "GENSAL", "IEESGO", "TGOV1",
                                                "SEXS", "ESDC1A", "IEEEST"])

    raw_source = read(joinpath(@__DIR__, "testdata", "2bus.raw"), String)
    inactive_gen = "     2,'2 ', 0, 0, 100, -100, 1, 0, 100, 0, 1, 0, 0, 1, 0, 100, 0, 1, 1, 1.0\n"
    raw_data = replace(raw_source,
                       "0 / END OF GENERATOR DATA" => inactive_gen * "0 / END OF GENERATOR DATA")
    mktemp() do raw_path, raw_io
        write(raw_io, raw_data)
        close(raw_io)
        mktemp() do dyr_path, io
            write(io, "1 'GENROU' 1 0 /\n")       # active native machine
            write(io, "1 'GENSAL' 1 0 /\n")       # duplicate machine family
            write(io, "1 'GGOV1' 1 0 /\n")        # unsupported by default
            write(io, "1 'TGOV1' 1 0 /\n")        # duplicate governor family
            write(io, "2 'SEXS' 2 0 /\n")         # inactive static generator
            write(io, "99999 'SEXS' 1 0 /\n")     # unmatched
            close(io)

            report = analyze_dyr_coverage(raw_path, dyr_path)
            @test getfield.(report.records, :status) ==
                  [:native, :duplicate, :unsupported, :duplicate, :inactive, :unmatched]
            @test getfield.(report.records, :active) == [true, true, true, true, false, nothing]
            @test report.records[1].device_id == "1"
            @test report.records[1].effective_model == "GENROU"
            @test coverage_counts(report) == Dict(:active => 4, :inactive => 1, :native => 1,
                                                  :redirected => 0, :unsupported => 1,
                                                  :unmatched => 1, :duplicate => 2)
            @test coverage_by_model(report, "gensal").duplicate == 1
            @test native_coverage(report) == 0.25
            @test (1, "1") ∉ report.active_generators_without_machine

            redirected = analyze_dyr_coverage(raw_path, dyr_path; redirects=Dict("GGOV1" => "TGOV1"))
            @test redirected.records[3].status == :redirected
            @test redirected.records[3].effective_model == "TGOV1"

            mktemp() do unsupported_path, unsupported_io
                write(unsupported_io, "1 'GENCLS' 1 0 /\n")
                close(unsupported_io)
                unsupported = analyze_dyr_coverage(raw_path, unsupported_path)
                @test unsupported.records[1].status == :unsupported
                @test unsupported.active_generators_without_machine == [(1, "1")]
            end
        end
    end
end

@testset "ACTIVS exact current coverage" begin
    expected = Dict(
        "200" => (active=114, inactive=33, native=114, unsupported=0, missing=0),
        "500" => (active=170, inactive=102, native=129, unsupported=41, missing=0),
        "2000" => (active=1335, inactive=404, native=677, unsupported=658, missing=98),
    )
    for case in ("200", "500", "2000")
        raw_path = joinpath(COVERAGE_DATA, "ACTIVSg$case.raw")
        dyr_path = joinpath(COVERAGE_DATA, "ACTIVSg$case.dyr")
        if !isfile(raw_path) || !isfile(dyr_path)
            @test_skip "ACTIVSg$case data are not available"
            continue
        end
        report = analyze_dyr_coverage(raw_path, dyr_path)
        counts = coverage_counts(report)
        want = expected[case]
        @test counts[:active] == want.active
        @test counts[:inactive] == want.inactive
        @test counts[:native] == want.native
        @test counts[:unsupported] == want.unsupported
        @test counts[:redirected] == 0
        @test counts[:unmatched] == 0
        @test counts[:duplicate] == 0
        @test length(report.active_generators_without_machine) == want.missing
    end
end
