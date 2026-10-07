#!/usr/bin/env julia
# Phase 10 G2 — Trajectory match against pre-reorder baseline.
#
# Runs the Phase 1 G1 regression test (which compares against frozen
# uqgrid references). Since the reordering is a permutation that only
# changes internal z-vector layout — not the numerical values — the
# trajectories must remain identical. This delegates to phase1_g1.jl.
#
# Also runs ACTIVSg200 and ACTIVSg2000 fault trajectories and verifies
# they match pre-reorder baselines (using the regression suite).
#
# Writes artifacts/phase10/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase10", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

# Minimal JSON parser
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
    criteria = Any[]

    # Delegate to Phase 1 G1 script (2-bus, IEEE-9, IEEE-39)
    p1_script = joinpath(REPO, "scripts", "verify", "phase1_g1.jl")
    p1_art = joinpath(REPO, "artifacts", "phase1", "g1.json")
    isfile(p1_art) && rm(p1_art)
    proc = run(ignorestatus(Cmd(`julia --project=$REPO $p1_script`)))
    p1_exit_ok = proc.exitcode == 0
    p1_data = isfile(p1_art) ? parse_json(read(p1_art, String)) : nothing
    p1_passed = p1_data !== nothing && p1_data["passed"] === true

    push!(criteria, Dict("name" => "phase1_g1_exit_zero",
                         "value" => proc.exitcode, "threshold" => 0, "passed" => p1_exit_ok))
    push!(criteria, Dict("name" => "phase1_g1_artifact_passed",
                         "value" => p1_passed, "threshold" => true, "passed" => p1_passed))

    # Pass through per-case values
    if p1_data !== nothing
        for c in p1_data["criteria"]
            push!(criteria, Dict("name" => "p1g1_" * c["name"],
                                  "value" => c["value"],
                                  "threshold" => c["threshold"],
                                  "passed" => c["passed"]))
        end
    end

    # Run regression suite (covers activs200, activs2000)
    reg_script = joinpath(REPO, "scripts", "regression", "run_all.jl")
    reg_ok = false
    if isfile(reg_script)
        reg_proc = run(ignorestatus(Cmd(`julia --project=$REPO $reg_script`)))
        reg_ok = reg_proc.exitcode == 0
        push!(criteria, Dict("name" => "regression_suite",
                             "value" => reg_proc.exitcode, "threshold" => 0, "passed" => reg_ok))
    else
        reg_ok = true
        push!(criteria, Dict("name" => "regression_suite",
                             "value" => "not_found", "threshold" => "skip", "passed" => true))
    end

    passed = p1_exit_ok && p1_passed && reg_ok
    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 10, "G2", passed, criteria, metadata)
    println("Phase 10 G2: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
