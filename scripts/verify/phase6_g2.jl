#!/usr/bin/env julia
# Phase 6 G2 — Baselines captured.
#
# Runs the timing baseline script and verifies that
# artifacts/phase6/baseline.json exists with wall-clock per case.
#
# Writes artifacts/phase6/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const BASELINE = joinpath(REPO, "artifacts", "phase6", "baseline.json")
const OUT  = joinpath(REPO, "artifacts", "phase6", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

# Minimal JSON parser (reused from phase5_g2.jl pattern)
mutable struct _P; s::String; i::Int; end
_skip(p::_P) = while p.i <= lastindex(p.s) && p.s[p.i] in (' ','\t','\n','\r'); p.i = nextind(p.s,p.i); end
function _expect(p::_P, c::Char); _skip(p); @assert p.s[p.i] == c; p.i = nextind(p.s,p.i); end
function _str(p::_P)
    _expect(p, '"'); buf = IOBuffer()
    while p.i <= lastindex(p.s)
        c = p.s[p.i]
        if c == '"'; p.i = nextind(p.s,p.i); return String(take!(buf))
        elseif c == '\\'
            p.i = nextind(p.s,p.i); e = p.s[p.i]
            print(buf, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e)
            p.i = nextind(p.s,p.i)
        else; print(buf, c); p.i = nextind(p.s,p.i)
        end
    end
    error("unterminated string")
end
function _num(p::_P)
    start = p.i
    while p.i <= lastindex(p.s) && (p.s[p.i] in ('-','+','.','e','E') || ('0' <= p.s[p.i] <= '9'))
        p.i = nextind(p.s,p.i)
    end
    s = p.s[start:prevind(p.s,p.i)]
    occursin('.',s) || occursin('e',s) || occursin('E',s) ? parse(Float64,s) : parse(Int,s)
end
function _val(p::_P)
    _skip(p); c = p.s[p.i]
    c == '{' ? _obj(p) : c == '[' ? _arr(p) : c == '"' ? _str(p) :
    c == 't' ? (p.i += 4; true) : c == 'f' ? (p.i += 5; false) :
    c == 'n' ? (p.i += 4; nothing) : _num(p)
end
function _obj(p::_P)
    _expect(p, '{'); d = Dict{String,Any}(); _skip(p)
    if p.s[p.i] == '}'; p.i = nextind(p.s,p.i); return d; end
    while true
        _skip(p); k = _str(p); _expect(p, ':'); d[k] = _val(p); _skip(p)
        c = p.s[p.i]
        c == ',' ? (p.i = nextind(p.s,p.i)) : c == '}' ? (p.i = nextind(p.s,p.i); return d) : error()
    end
end
function _arr(p::_P)
    _expect(p, '['); a = Any[]; _skip(p)
    if p.s[p.i] == ']'; p.i = nextind(p.s,p.i); return a; end
    while true
        push!(a, _val(p)); _skip(p); c = p.s[p.i]
        c == ',' ? (p.i = nextind(p.s,p.i)) : c == ']' ? (p.i = nextind(p.s,p.i); return a) : error()
    end
end
parse_json(s::AbstractString) = _val(_P(String(s), firstindex(s)))

const EXPECTED_CASES = Set(["2bus_genrou", "ieee9_nogov", "ieee39_gov", "activs200", "activs2000", "activs70k"])
const REQUIRED_BREAKDOWN_KEYS = Set(["residual", "jacobian", "lsolve_factor", "lsolve_solve", "ybus_mul", "init", "parse"])

function main()
    t_start = time()

    # Run the timing baseline script to generate/refresh baseline.json
    timing_script = joinpath(REPO, "scripts", "timing_baseline.jl")
    proc = run(ignorestatus(Cmd(`julia --project=$REPO $timing_script`)))
    script_ok = proc.exitcode == 0

    criteria = Any[]

    # Check that baseline.json exists
    file_exists = isfile(BASELINE)
    push!(criteria, Dict("name" => "baseline_file_exists",
                         "value" => file_exists, "threshold" => true, "passed" => file_exists))

    # Check that all cases are present and have wall-clock data
    cases_ok = false
    breakdown_ok = false
    if file_exists
        data = parse_json(read(BASELINE, String))
        all_completed = get(data, "all_completed", false)
        cases = get(data, "cases", Any[])
        found = Set{String}()
        has_wallclock = true
        has_breakdown = true
        for c in cases
            name = get(c, "name", "")
            push!(found, name)
            if !get(c, "completed", false) || !haskey(c, "t_min_s")
                has_wallclock = false
            end
            # Check breakdown
            bd = get(c, "breakdown", nothing)
            if bd === nothing || !isa(bd, Dict)
                has_breakdown = false
            else
                bd_keys = Set(collect(keys(bd)))
                if !issubset(REQUIRED_BREAKDOWN_KEYS, bd_keys)
                    has_breakdown = false
                end
            end
        end
        cases_ok = EXPECTED_CASES == found && has_wallclock && all_completed
        breakdown_ok = has_breakdown && cases_ok
    end
    push!(criteria, Dict("name" => "all_cases_with_wallclock",
                         "value" => cases_ok, "threshold" => true, "passed" => cases_ok))
    push!(criteria, Dict("name" => "all_cases_with_breakdown",
                         "value" => breakdown_ok, "threshold" => true, "passed" => breakdown_ok))

    passed = script_ok && file_exists && cases_ok && breakdown_ok
    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 6, "G2", passed, criteria, metadata)
    println("\nPhase 6 G2: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
