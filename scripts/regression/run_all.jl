#!/usr/bin/env julia
# Phase 0 G3 — Canonical regression-suite runner.
#
# Runs each test/regression/compare_*.jl script as a subprocess (so a
# crash in one case doesn't take down the whole suite) and aggregates
# their per-case JSON artifacts (artifacts/regression/<case>.json) into
# artifacts/phase0/g3.json. Exits 0 iff every per-case script exited 0
# and every aggregated entry reports passed: true.
#
# The list below is the canonical suite from phase-0-baseline.md#D5.

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const REG_DIR   = joinpath(REPO_ROOT, "test", "regression")
const ART_DIR   = joinpath(REPO_ROOT, "artifacts", "regression")
const OUT_PATH  = joinpath(REPO_ROOT, "artifacts", "phase0", "g3.json")

const CASES = [
    # Size ladder
    "compare_ieee9",
    "compare_2bus_genrou",
    "compare_activs200",
    "compare_activs2000",
    "compare_activs70k",
    # Governor / exciter coverage
    "compare_ieee9_gov",
    "compare_tgov1_ieee9",
    "compare_sexs_ieee9",
    "compare_2bus_ieesgo",
    "compare_gensal_2bus",
    "compare_tgov1_2bus",
    "compare_sexs_2bus",
    "compare_2bus_esdc1a_reduced",
]

mkpath(ART_DIR); mkpath(dirname(OUT_PATH))

# Minimal JSON (no JSON.jl dependency).
function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"' || c == '\\'; print(buf, '\\', c)
        elseif c == '\n'; print(buf, "\\n")
        elseif c == '\r'; print(buf, "\\r")
        elseif c == '\t'; print(buf, "\\t")
        elseif UInt32(c) < 0x20
            print(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else; print(buf, c)
        end
    end
    String(take!(buf))
end
function json_value(v)
    if v === nothing; return "null"
    elseif v isa Bool; return v ? "true" : "false"
    elseif v isa AbstractString; return string('"', _json_escape(v), '"')
    elseif v isa Integer; return string(v)
    elseif v isa AbstractFloat; return isfinite(v) ? string(v) : "null"
    elseif v isa AbstractVector; return string("[", join(json_value.(v), ","), "]")
    elseif v isa AbstractDict
        ks = sort(collect(keys(v)), by=string)
        parts = String[]
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
            print(buf, c); c == '"' && prev != '\\' && (in_string = false)
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

# Naive JSON parser — only handles the subset our per-case scripts emit
# (dicts, arrays, strings, numbers, true/false/null, plain ASCII keys).
mutable struct Parser; s::String; i::Int; end
function _skip(p::Parser)
    while p.i <= lastindex(p.s)
        c = p.s[p.i]
        (c == ' ' || c == '\t' || c == '\n' || c == '\r') || break
        p.i = nextind(p.s, p.i)
    end
end
function _expect(p::Parser, c::Char)
    _skip(p)
    @assert p.s[p.i] == c "expected $c at byte $(p.i), got $(p.s[p.i])"
    p.i = nextind(p.s, p.i)
end
function _parse_str(p::Parser)
    _expect(p, '"')
    start = p.i; buf = IOBuffer()
    while p.i <= lastindex(p.s)
        c = p.s[p.i]
        if c == '"'; p.i = nextind(p.s, p.i); return String(take!(buf))
        elseif c == '\\'
            p.i = nextind(p.s, p.i); esc = p.s[p.i]
            if     esc == 'n'; print(buf, '\n')
            elseif esc == 't'; print(buf, '\t')
            elseif esc == 'r'; print(buf, '\r')
            elseif esc == '"'; print(buf, '"')
            elseif esc == '\\'; print(buf, '\\')
            elseif esc == 'u'
                hex = p.s[p.i+1:p.i+4]; print(buf, Char(parse(Int, hex, base=16)))
                p.i = nextind(p.s, p.i, 4)
            else; print(buf, esc)
            end
            p.i = nextind(p.s, p.i)
        else
            print(buf, c); p.i = nextind(p.s, p.i)
        end
    end
    error("unterminated string starting at $start")
end
function _parse_num(p::Parser)
    start = p.i
    while p.i <= lastindex(p.s)
        c = p.s[p.i]
        if c in ('-', '+', '.', 'e', 'E') || ('0' <= c <= '9')
            p.i = nextind(p.s, p.i)
        else; break
        end
    end
    s = p.s[start:prevind(p.s, p.i)]
    occursin('.', s) || occursin('e', s) || occursin('E', s) ? parse(Float64, s) : parse(Int, s)
end
function _parse_val(p::Parser)
    _skip(p); c = p.s[p.i]
    if c == '{'; return _parse_obj(p)
    elseif c == '['; return _parse_arr(p)
    elseif c == '"'; return _parse_str(p)
    elseif c == 't'; p.i += 4; return true
    elseif c == 'f'; p.i += 5; return false
    elseif c == 'n'; p.i += 4; return nothing
    else;            return _parse_num(p)
    end
end
function _parse_obj(p::Parser)
    _expect(p, '{'); d = Dict{String,Any}(); _skip(p)
    if p.s[p.i] == '}'; p.i = nextind(p.s, p.i); return d; end
    while true
        _skip(p); k = _parse_str(p); _expect(p, ':'); v = _parse_val(p); d[k] = v
        _skip(p); c = p.s[p.i]
        if c == ','; p.i = nextind(p.s, p.i)
        elseif c == '}'; p.i = nextind(p.s, p.i); return d
        else; error("unexpected $c in obj")
        end
    end
end
function _parse_arr(p::Parser)
    _expect(p, '['); a = Any[]; _skip(p)
    if p.s[p.i] == ']'; p.i = nextind(p.s, p.i); return a; end
    while true
        push!(a, _parse_val(p)); _skip(p); c = p.s[p.i]
        if c == ','; p.i = nextind(p.s, p.i)
        elseif c == ']'; p.i = nextind(p.s, p.i); return a
        else; error("unexpected $c in arr")
        end
    end
end
parse_json(s::AbstractString) = _parse_val(Parser(String(s), firstindex(s)))

function main()
println("="^60)
println("Phase 0 G3 — regression suite")
println("Suite size: $(length(CASES))")
println("="^60)

case_results = Dict{String,Any}()
all_passed = true
n_scripts_exit_zero = 0
n_artifacts_passed = 0
exec_errors = String[]
total_wall = 0.0
for case in CASES
    script = joinpath(REG_DIR, "$case.jl")
    art = joinpath(ART_DIR, "$case.json")
    isfile(art) && rm(art)  # don't credit stale artifacts from prior runs
    println("\n-- $case --")
    t0 = time()
    proc = run(ignorestatus(Cmd(`julia --project=$REPO_ROOT $script`)))
    wall = time() - t0; total_wall += wall
    exit_ok = proc.exitcode == 0
    if exit_ok; n_scripts_exit_zero += 1; end
    parsed = nothing; passed_in_art = false
    if isfile(art)
        try
            parsed = parse_json(read(art, String))
            passed_in_art = parsed["passed"] === true
            if passed_in_art; n_artifacts_passed += 1; end
        catch e
            push!(exec_errors, "$case: artifact parse failed: $(sprint(showerror, e))")
        end
    else
        push!(exec_errors, "$case: artifact missing at $art")
    end
    case_results[case] = Dict{String,Any}(
        "exit_code" => proc.exitcode,
        "artifact_exists" => isfile(art),
        "artifact_passed" => passed_in_art,
        "wallclock_s" => wall,
        "artifact_path" => art,
    )
    if !(exit_ok && passed_in_art); all_passed = false; end
end

criteria = Any[
    Dict("name" => "all_scripts_exit_zero",
         "value" => n_scripts_exit_zero, "threshold" => length(CASES),
         "passed" => n_scripts_exit_zero == length(CASES)),
    Dict("name" => "all_artifacts_passed",
         "value" => n_artifacts_passed, "threshold" => length(CASES),
         "passed" => n_artifacts_passed == length(CASES)),
]

git_sha = try strip(read(Cmd(`git rev-parse HEAD`, dir=REPO_ROOT), String)) catch; "" end

out = Dict{String,Any}(
    "phase" => 0,
    "gate"  => "G3",
    "passed" => all_passed,
    "criteria" => criteria,
    "metadata" => Dict{String,Any}(
        "git_sha" => git_sha,
        "n_cases" => length(CASES),
        "total_wallclock_s" => total_wall,
        "cases" => case_results,
        "errors" => exec_errors,
    ),
)
open(OUT_PATH, "w") do io
    write(io, pretty(json_value(out))); write(io, "\n")
end
println("\n"*"="^60)
println("Regression suite: passed=$(all_passed)  scripts_ok=$(n_scripts_exit_zero)/$(length(CASES))  artifacts_ok=$(n_artifacts_passed)/$(length(CASES))")
println("Aggregate artifact: $OUT_PATH")
println("Total wall: $(round(total_wall, digits=2))s")
return all_passed
end

exit(main() ? 0 : 1)
