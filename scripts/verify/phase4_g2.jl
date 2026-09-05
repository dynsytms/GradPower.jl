#!/usr/bin/env julia
# Phase 4 G2 — Unconditional regression suite (covers ACTIVSg2000
# GENSAL machines at scale via the :gensal -> :genrou alias).
#
# Delegates to scripts/regression/run_all.jl, then republishes its
# aggregate verdict as artifacts/phase4/g2.json. Exits 0 iff every
# regression case passes.

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const RUNNER = joinpath(REPO, "scripts", "regression", "run_all.jl")
const AGG    = joinpath(REPO, "artifacts", "phase0", "g3.json")
const OUT    = joinpath(REPO, "artifacts", "phase4", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

# Vendored mini JSON reader (same shape as phase4_g1.jl).
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

function main()
    t0 = time()
    isfile(AGG) && rm(AGG)
    proc = run(ignorestatus(Cmd(`julia --project=$REPO $RUNNER`)))
    wall = time() - t0
    exit_ok = proc.exitcode == 0

    parsed = isfile(AGG) ? parse_json(read(AGG, String)) : nothing
    suite_passed = parsed !== nothing && parsed["passed"] === true
    n_cases = parsed !== nothing ? get(parsed["metadata"], "n_cases", -1) : -1
    n_ok = 0
    if parsed !== nothing
        for c in parsed["criteria"]
            if c["name"] == "all_artifacts_passed"; n_ok = Int(c["value"]); end
        end
    end

    passed = exit_ok && suite_passed
    criteria = Any[
        Dict("name" => "regression_runner_exit_zero",
             "value" => proc.exitcode, "threshold" => 0,
             "passed" => exit_ok),
        Dict("name" => "all_cases_passed",
             "value" => n_ok, "threshold" => n_cases,
             "passed" => suite_passed),
    ]
    metadata = Dict{String,Any}(
        "wallclock_s" => wall,
        "git_sha" => git_sha(REPO),
        "aggregate_artifact" => AGG,
        "runner" => RUNNER,
        "n_cases" => n_cases,
    )
    write_artifact(OUT, 4, "G2", passed, criteria, metadata)
    println("Phase 4 G2: passed=$passed  $n_ok/$n_cases cases  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
