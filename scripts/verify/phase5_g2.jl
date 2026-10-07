#!/usr/bin/env julia
# Phase 5 G2 — No parity regression.
#
# Re-runs the Phase 1 G1 trajectory-parity script (2-bus genrou,
# ieee9_nogov, ieee39_gov) on the current source tree. The phase 5
# allocation-elimination changes touch the Newton inner loop; this gate
# verifies the trajectories remain bit-identical (within the same
# 5e-9 / 0.02 tolerances Phase 1 G1 uses).
#
# Writes artifacts/phase5/g2.json. Exits 0 iff the embedded Phase 1 G1
# script exited 0 and its artifact reports passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const SCRIPT = joinpath(REPO, "scripts", "verify", "phase1_g1.jl")
const SRC_ART = joinpath(REPO, "artifacts", "phase1", "g1.json")
const OUT = joinpath(REPO, "artifacts", "phase5", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

# Reuse the tiny JSON reader from the phase4 gate (vendor-inlined here
# to avoid an additional include of that file).
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
    isfile(SRC_ART) && rm(SRC_ART)
    proc = run(ignorestatus(Cmd(`julia --project=$REPO $SCRIPT`)))
    wall = time() - t0
    exit_ok = proc.exitcode == 0

    parsed = isfile(SRC_ART) ? parse_json(read(SRC_ART, String)) : nothing
    art_passed = parsed !== nothing && parsed["passed"] === true

    passed = exit_ok && art_passed
    criteria = Any[
        Dict("name" => "phase1_g1_script_exit_zero",
             "value" => proc.exitcode, "threshold" => 0, "passed" => exit_ok),
        Dict("name" => "phase1_g1_artifact_passed",
             "value" => art_passed, "threshold" => true, "passed" => art_passed),
    ]
    # Pass through per-case worst values so reviewers can see them in
    # one place without opening the upstream artifact.
    if parsed !== nothing
        for c in parsed["criteria"]
            push!(criteria, Dict("name" => c["name"],
                                  "value" => c["value"],
                                  "threshold" => c["threshold"],
                                  "passed" => c["passed"]))
        end
    end
    metadata = Dict{String,Any}(
        "wallclock_s" => wall,
        "git_sha" => git_sha(REPO),
        "delegated_script" => SCRIPT,
        "delegated_artifact" => SRC_ART,
    )
    write_artifact(OUT, 5, "G2", passed, criteria, metadata)
    println("Phase 5 G2: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
