#!/usr/bin/env julia
# Phase 2 G1 — All seed devices registered.
#
# Verifies that the seven seed devices (Genrou, IEESGO, TGOV1, SEXS,
# ESDC1A, ZIPLoad, StaticGenerator) appear in `list_contracts()` with
# diff/alg/ctrl/par sizes matching the sizes carried by a freshly
# instantiated placeholder struct of each type.
#
# Writes artifacts/phase2/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using Dates

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_PATH = joinpath(REPO, "artifacts", "phase2", "g1.json")
mkpath(dirname(OUT_PATH))

function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"' || c == '\\'; print(buf, '\\', c)
        elseif c == '\n'; print(buf, "\\n")
        elseif c == '\r'; print(buf, "\\r")
        elseif c == '\t'; print(buf, "\\t")
        elseif UInt32(c) < 0x20; print(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else; print(buf, c)
        end
    end
    String(take!(buf))
end
function json_value(v)
    if v === nothing; return "null"
    elseif v isa Bool; return v ? "true" : "false"
    elseif v isa AbstractString; return string('"', _json_escape(v), '"')
    elseif v isa Symbol; return string('"', _json_escape(String(v)), '"')
    elseif v isa Integer; return string(v)
    elseif v isa AbstractFloat; return isfinite(v) ? string(v) : "null"
    elseif v isa AbstractVector; return string("[", join(json_value.(v), ","), "]")
    elseif v isa AbstractDict
        ks = sort(collect(keys(v)), by=string); parts = String[]
        for k in ks
            push!(parts, string('"', _json_escape(String(string(k))), '"', ":", json_value(v[k])))
        end
        return string("{", join(parts, ","), "}")
    else; return string('"', _json_escape(string(v)), '"')
    end
end
function pretty(json::AbstractString)
    buf = IOBuffer(); indent = 0; in_string = false; prev = '\0'
    for c in json
        if in_string
            print(buf, c); (c == '"' && prev != '\\') && (in_string = false)
        else
            if c == '"'; in_string = true; print(buf, c)
            elseif c == '{' || c == '['; print(buf, c, '\n'); indent += 2; print(buf, ' '^indent)
            elseif c == '}' || c == ']'; indent -= 2; print(buf, '\n', ' '^indent, c)
            elseif c == ','; print(buf, c, '\n', ' '^indent)
            elseif c == ':'; print(buf, c, ' ')
            else; print(buf, c)
            end
        end
        prev = c
    end
    String(take!(buf))
end

# Placeholder instances used to read the canonical sizes off each struct.
# Must match the instances used in src/devices/seeds.jl so the registry
# is verified against the same source of truth.
function placeholder_instances()
    return [
        (:genrou,     GradPower.GenericGenerator(1, "1")),
        (:ieesgo,     GradPower.IEESGO(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (:tgov1,      GradPower.TGOV1(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (:sexs,       GradPower.SEXS(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (:esdc1a,     GradPower.ESDC1A(1, "1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (:zipload,    GradPower.ZIPLoad(1, "1", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
        (:static_gen, GradPower.StaticGenerator(Int64(1), Int64(2), Int64[1], 1.0, 0.0)),
    ]
end

function main()
    t_start = time()
    contracts = GradPower.list_contracts()
    by_name = Dict(c.name => c for c in contracts)
    insts = placeholder_instances()

    criteria = Any[]
    details = Dict{String,Any}()
    all_passed = true

    # Criterion 1: count
    expected_n = length(insts)
    actual_n = length(contracts)
    count_ok = actual_n == expected_n
    all_passed &= count_ok
    push!(criteria, Dict("name" => "contract_count",
                         "value" => actual_n, "threshold" => expected_n,
                         "passed" => count_ok))

    # Criterion 2: each expected device present with sizes matching its
    # placeholder instance.
    for (name, inst) in insts
        d, a, c, p = Int(inst.diff_size), Int(inst.alg_size), Int(inst.ctrl_size), Int(inst.par_size)
        present = haskey(by_name, name)
        sizes_ok = false
        if present
            con = by_name[name]
            sizes_ok = (con.diff_size == d && con.alg_size == a &&
                        con.ctrl_size == c && con.par_size == p)
        end
        ok = present && sizes_ok
        all_passed &= ok
        push!(criteria, Dict("name" => "$(name)_registered_with_correct_sizes",
                             "value" => ok ? 1 : 0, "threshold" => 1,
                             "passed" => ok))
        details[String(name)] = Dict("present" => present,
                                      "expected" => [d, a, c, p],
                                      "got" => present ? [by_name[name].diff_size,
                                                          by_name[name].alg_size,
                                                          by_name[name].ctrl_size,
                                                          by_name[name].par_size] : nothing)
    end

    git_sha = try strip(read(Cmd(`git rev-parse HEAD`, dir=REPO), String)) catch; "" end
    out = Dict{String,Any}(
        "phase" => 2,
        "gate" => "G1",
        "passed" => all_passed,
        "criteria" => criteria,
        "metadata" => Dict{String,Any}(
            "hardware" => string(Sys.cpu_info()[1].model),
            "git_sha" => git_sha,
            "wallclock_s" => time() - t_start,
            "contracts" => details,
        ),
    )
    open(OUT_PATH, "w") do io
        write(io, pretty(json_value(out))); write(io, "\n")
    end

    for c in contracts
        println("  ", c.name, "  ", c.class, "  ",
                c.diff_size, "/", c.alg_size, "/", c.ctrl_size, "/", c.par_size,
                "  attaches_to=", c.attaches_to)
    end
    println("\nPhase 2 G1: passed=$all_passed  artifact=$OUT_PATH")
    return all_passed
end

exit(main() ? 0 : 1)
