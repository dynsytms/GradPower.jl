# Shared helpers for Phase 3 gate scripts. No external dependencies beyond Base.

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

function write_artifact(path::AbstractString, phase::Int, gate::AbstractString,
                         passed::Bool, criteria::Vector, metadata::Dict)
    out = Dict{String,Any}(
        "phase" => phase, "gate" => gate, "passed" => passed,
        "criteria" => criteria, "metadata" => metadata,
    )
    open(path, "w") do io
        write(io, pretty(json_value(out))); write(io, "\n")
    end
end

git_sha(repo::AbstractString) = try
    strip(read(Cmd(`git rev-parse HEAD`, dir=repo), String))
catch; "" end
