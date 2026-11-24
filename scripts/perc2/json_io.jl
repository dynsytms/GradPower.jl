# Minimal JSON writer/reader (no external packages) shared by the PERC2 study
# scripts. Supports Dict/NamedTuple/Vector/Tuple/String/Symbol/Number/Bool/nothing.
# Non-finite floats are written as null (and counted by `count_nonfinite`).

module PERC2JSON

export write_json, read_json, json_string, count_nonfinite

const NONFINITE = Ref(0)

function _esc(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"';      print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\t'; print(io, "\\t")
        elseif c == '\r'; print(io, "\\r")
        elseif c < ' ';   print(io, "\\u", string(UInt16(c), base = 16, pad = 4))
        else              print(io, c)
        end
    end
    return String(take!(io))
end

function _w(io, x, ind)
    pad = "  "^ind
    if x === nothing || x === missing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        if isfinite(x)
            print(io, x)
        else
            NONFINITE[] += 1
            print(io, "null")
        end
    elseif x isa Symbol || x isa AbstractString
        print(io, '"', _esc(String(x)), '"')
    elseif x isa NamedTuple
        _w(io, Dict{String,Any}(string(k) => v for (k, v) in pairs(x)), ind)
    elseif x isa AbstractDict
        isempty(x) && return print(io, "{}")
        println(io, "{")
        ks = sort!(collect(keys(x)); by = string)
        for (i, k) in enumerate(ks)
            print(io, pad, "  \"", _esc(string(k)), "\": ")
            _w(io, x[k], ind + 1)
            println(io, i < length(ks) ? "," : "")
        end
        print(io, pad, "}")
    elseif x isa AbstractVector || x isa Tuple
        if all(e -> e isa Number || e isa Bool || e === nothing, x)
            print(io, "[")
            for (i, e) in enumerate(x)
                i > 1 && print(io, ",")
                _w(io, e, ind + 1)
            end
            print(io, "]")
        else
            isempty(x) && return print(io, "[]")
            println(io, "[")
            for (i, e) in enumerate(x)
                print(io, pad, "  ")
                _w(io, e, ind + 1)
                println(io, i < length(x) ? "," : "")
            end
            print(io, pad, "]")
        end
    else
        throw(ArgumentError("cannot serialize $(typeof(x))"))
    end
end

function json_string(x)
    io = IOBuffer()
    _w(io, x, 0)
    return String(take!(io))
end

function write_json(path::AbstractString, x)
    mkpath(dirname(abspath(path)))
    NONFINITE[] = 0
    s = json_string(x)
    write(path, s * "\n")
    return NONFINITE[]
end

count_nonfinite() = NONFINITE[]

# ---- reader ----
mutable struct _P
    s::String
    i::Int
end

_ws!(p) = (while p.i <= ncodeunits(p.s) && p.s[p.i] in (' ', '\n', '\t', '\r'); p.i += 1; end)

function _val(p)
    _ws!(p)
    c = p.s[p.i]
    if c == '{'
        p.i += 1
        d = Dict{String,Any}()
        _ws!(p)
        if p.s[p.i] == '}'; p.i += 1; return d; end
        while true
            _ws!(p)
            k = _str(p)
            _ws!(p); p.s[p.i] == ':' || error("expected : at $(p.i)"); p.i += 1
            d[k] = _val(p)
            _ws!(p)
            if p.s[p.i] == ','; p.i += 1
            elseif p.s[p.i] == '}'; p.i += 1; return d
            else error("bad object at $(p.i)") end
        end
    elseif c == '['
        p.i += 1
        v = Any[]
        _ws!(p)
        if p.s[p.i] == ']'; p.i += 1; return v; end
        while true
            push!(v, _val(p))
            _ws!(p)
            if p.s[p.i] == ','; p.i += 1
            elseif p.s[p.i] == ']'; p.i += 1; return v
            else error("bad array at $(p.i)") end
        end
    elseif c == '"'
        return _str(p)
    elseif startswith(SubString(p.s, p.i), "true");  p.i += 4; return true
    elseif startswith(SubString(p.s, p.i), "false"); p.i += 5; return false
    elseif startswith(SubString(p.s, p.i), "null");  p.i += 4; return nothing
    else
        j = p.i
        while j <= ncodeunits(p.s) && p.s[j] in "+-0123456789.eE"; j += 1; end
        tok = p.s[p.i:j-1]
        p.i = j
        return occursin(r"[.eE]", tok) ? parse(Float64, tok) : parse(Int, tok)
    end
end

function _str(p)
    p.s[p.i] == '"' || error("expected string at $(p.i)")
    p.i += 1
    io = IOBuffer()
    while p.s[p.i] != '"'
        c = p.s[p.i]
        if c == '\\'
            p.i += 1
            e = p.s[p.i]
            if e == 'n'; print(io, '\n')
            elseif e == 't'; print(io, '\t')
            elseif e == 'r'; print(io, '\r')
            elseif e == 'u'; print(io, Char(parse(UInt16, p.s[p.i+1:p.i+4]; base = 16))); p.i += 4
            else print(io, e) end
        else
            print(io, c)
        end
        p.i += 1
    end
    p.i += 1
    return String(take!(io))
end

read_json(path::AbstractString) = _val(_P(read(path, String), 1))

end # module
