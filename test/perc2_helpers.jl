# Helpers for the PERC2 tests: a minimal JSON reader (no JSON package in the
# test environment), the reference CSV reader, and V/f profile playback.

const PERC2_REF_DIR = joinpath(@__DIR__, "perc2_reference")

# --- minimal recursive-descent JSON reader (objects, arrays, numbers, strings,
#     true/false/null) ------------------------------------------------------
function perc2_json_parse(s::AbstractString)
    pos = Ref(1)
    peek() = pos[] <= lastindex(s) ? s[pos[]] : '\0'
    function skipws()
        while pos[] <= lastindex(s) && isspace(s[pos[]])
            pos[] = nextind(s, pos[])
        end
    end
    function expect(c::Char)
        skipws()
        peek() == c || error("JSON: expected '$c' at $(pos[]), got '$(peek())'")
        pos[] = nextind(s, pos[])
    end
    function parse_string()
        expect('"')
        io = IOBuffer()
        while true
            c = peek()
            c == '\0' && error("JSON: unterminated string")
            pos[] = nextind(s, pos[])
            c == '"' && break
            if c == '\\'
                e = peek(); pos[] = nextind(s, pos[])
                if e == 'n'; print(io, '\n')
                elseif e == 't'; print(io, '\t')
                elseif e == 'r'; print(io, '\r')
                elseif e == 'b'; print(io, '\b')
                elseif e == 'f'; print(io, '\f')
                elseif e == 'u'
                    print(io, Char(parse(UInt16, s[pos[]:pos[]+3]; base = 16)))
                    pos[] += 4
                else
                    print(io, e)
                end
            else
                print(io, c)
            end
        end
        return String(take!(io))
    end
    function parse_value()
        skipws()
        c = peek()
        if c == '{'
            expect('{')
            d = Dict{String,Any}()
            skipws()
            if peek() == '}'; expect('}'); return d; end
            while true
                skipws()
                k = parse_string()
                expect(':')
                d[k] = parse_value()
                skipws()
                if peek() == ','; expect(','); else; expect('}'); break; end
            end
            return d
        elseif c == '['
            expect('[')
            a = Any[]
            skipws()
            if peek() == ']'; expect(']'); return a; end
            while true
                push!(a, parse_value())
                skipws()
                if peek() == ','; expect(','); else; expect(']'); break; end
            end
            return a
        elseif c == '"'
            return parse_string()
        elseif startswith(SubString(s, pos[]), "true")
            pos[] += 4; return true
        elseif startswith(SubString(s, pos[]), "false")
            pos[] += 5; return false
        elseif startswith(SubString(s, pos[]), "null")
            pos[] += 4; return nothing
        else
            m = match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?", SubString(s, pos[]))
            m === nothing && error("JSON: unexpected '$(peek())' at $(pos[])")
            pos[] += length(m.match)
            return parse(Float64, m.match)
        end
    end
    v = parse_value()
    skipws()
    pos[] <= lastindex(s) && error("JSON: trailing characters at $(pos[])")
    return v
end

const _PERC2_CASES = Ref{Any}(nothing)

perc2_cases() = (_PERC2_CASES[] === nothing &&
        (_PERC2_CASES[] = perc2_json_parse(read(joinpath(PERC2_REF_DIR, "cases.json"), String)));
    _PERC2_CASES[])

# CSV with header t,V,P,FracOn,f,Vtf → NamedTuple of vectors
function perc2_read_csv(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    hdr = Symbol.(split(lines[1], ','))
    cols = [Float64[] for _ in hdr]
    for ln in lines[2:end]
        for (j, x) in enumerate(split(ln, ','))
            push!(cols[j], parse(Float64, x))
        end
    end
    return NamedTuple{Tuple(hdr)}(Tuple(cols))
end

const PERC2_FLAGS = (:VLMflag, :VHMflag, :VLLflag, :VLHflag, :FLflag, :FHflag)

# CON-name dictionary → PERC2Params (flags 0/1 → Bool)
function params_from_cons(cons::AbstractDict)
    kw = Pair{Symbol,Any}[]
    for (k, v) in cons
        sk = Symbol(k)
        push!(kw, sk => (sk in PERC2_FLAGS ? Bool(round(v) == 1) : Float64(v)))
    end
    return GradPower.PERC2Params(; kw...)
end

# case = (name, description, cons, breakpoints, t_end, events, dt, ref)
function load_case(name::AbstractString)
    all = perc2_cases()
    c = all["cases"][name]
    ev = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in c["events"] if v !== nothing)
    bps = [(Float64(b[1]), Float64(b[2]), Float64(b[3])) for b in c["breakpoints"]]
    return (name = String(name), description = String(c["description"]),
            cons = c["cons"], breakpoints = bps, t_end = Float64(c["t_end"]),
            events = ev, dt = Float64(all["dt"]),
            ref = perc2_read_csv(joinpath(PERC2_REF_DIR, name * ".csv")))
end

perc2_case_names() = sort!(collect(keys(perc2_cases()["cases"])))

# piecewise-constant profile: last breakpoint with bt <= t + 1e-12
function perc2_profile(bps, t::Real)
    v, f = bps[1][2], bps[1][3]
    for (bt, bv, bf) in bps
        bt <= t + 1e-12 || break
        v, f = bv, bf
    end
    return v, f
end

# Playback of a V/f profile through the core; returns the snapshot after each
# call, k = 0..N, t_k = k*dt (P0 = 1, Q0 = 0, V0 = v(0)).
function playback(p, bps, t_end::Real, dt::Real; P0 = 1.0, Q0 = 0.0)
    N = round(Int, t_end / dt)
    v0, _ = perc2_profile(bps, 0.0)
    s = GradPower.perc2_init(p, P0, Q0, v0)
    snaps = Any[]
    for k in 0:N
        t = k * dt
        v, f = perc2_profile(bps, t)
        GradPower.perc2_step!(s, p, v, f, t, dt)
        push!(snaps, GradPower.perc2_snapshot(s, v))
    end
    return snaps
end

playback(p, case) = playback(p, case.breakpoints, case.t_end, case.dt)

# FracOn trace → Dict(:trip_time, :reconnect_start, :reconnect_end) (keys of
# events that do not occur are absent). Times are k*dt of a step:
#   trip_time        first step with FracOn < 1
#   reconnect_start  last step before FracOn first rises, i.e. the step on which
#                    the ramp starts (FracOn = level)
#   reconnect_end    first step FracOn reaches its final value after rising
# The reference events log reconnect_end from a ramp timer, which can lag the
# FracOn trace by one step (floating-point timer), so compare it to one step.
function perc2_events_from_fracon(frac::AbstractVector, dt::Real)
    ev = Dict{Symbol,Float64}()
    tol = 1e-12
    kt = findfirst(x -> x < 1 - tol, frac)
    kt === nothing && return ev
    ev[:trip_time] = (kt - 1) * dt
    kr = findfirst(k -> frac[k] > frac[k-1] + tol, kt+1:length(frac))
    kr === nothing && return ev
    kr += kt
    ev[:reconnect_start] = (kr - 2) * dt
    kend = findfirst(k -> frac[k] >= frac[end] - tol, kr:length(frac))
    kend === nothing && return ev
    kend += kr - 1
    ev[:reconnect_end] = (kend - 1) * dt
    return ev
end
