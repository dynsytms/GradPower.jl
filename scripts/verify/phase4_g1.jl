#!/usr/bin/env julia
# Phase 4 G1 — GENSAL 2-bus uqgrid parity.
#
# Runs test/regression/test_gensal_2bus.jl (which dispatches to the
# existing compare_gensal_2bus.jl), reads its per-case artifact, and
# republishes the result as artifacts/phase4/g1.json. Exits 0 iff the
# trajectory parity criterion passes at the 2-bus tolerance.

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const SCRIPT  = joinpath(REPO, "test", "regression", "test_gensal_2bus.jl")
const CASE_ART = joinpath(REPO, "artifacts", "regression", "compare_gensal_2bus.json")
const OUT  = joinpath(REPO, "artifacts", "phase4", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

# Tiny JSON reader sufficient for the case artifact emitted by
# test/regression/common.jl (numbers, strings, bools, null, nested
# dicts/arrays). Vendored to avoid a JSON.jl dep at gate time.
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
    isfile(CASE_ART) && rm(CASE_ART)  # don't credit stale artifacts
    proc = run(ignorestatus(Cmd(`julia --project=$REPO $SCRIPT`)))
    wall = time() - t0
    exit_ok = proc.exitcode == 0

    parsed = isfile(CASE_ART) ? parse_json(read(CASE_ART, String)) : nothing
    art_passed = parsed !== nothing && parsed["passed"] === true
    worst = 0.0; tol = NaN
    if parsed !== nothing
        for c in parsed["criteria"]
            if c["name"] == "max_traj_err_inf"
                worst = Float64(c["value"]); tol = Float64(c["threshold"])
            end
        end
    end

    passed = exit_ok && art_passed
    criteria = Any[
        Dict("name" => "regression_script_exit_zero",
             "value" => proc.exitcode, "threshold" => 0,
             "passed" => exit_ok),
        Dict("name" => "max_traj_err_inf",
             "value" => worst, "threshold" => tol,
             "passed" => art_passed),
    ]
    metadata = Dict{String,Any}(
        "wallclock_s" => wall,
        "git_sha" => git_sha(REPO),
        "case_artifact" => CASE_ART,
        "script" => SCRIPT,
    )
    write_artifact(OUT, 4, "G1", passed, criteria, metadata)
    println("Phase 4 G1: passed=$passed  worst=$worst tol=$tol  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
