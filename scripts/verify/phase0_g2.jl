#!/usr/bin/env julia
# Phase 0 G2 — Reference manifest frozen.
#
# Copies every .npz from examples/refs/ into artifacts/phase0/references/
# (byte-identical) and writes artifacts/phase0/references/manifest.json
# recording per-reference SHA-256 plus the uqgrid commit SHA and the
# GradPower main commit SHA at which the reference was last validated.
#
# Pass criterion (per phase-0-baseline.md):
#   - every .npz in examples/refs/ is also present under
#     artifacts/phase0/references/ with byte-identical content
#   - manifest.json exists, has one entry per reference, every entry has
#     non-null source_sha256, uqgrid_commit_sha, gradpower_validated_at_sha.
#
# Writes artifacts/phase0/g2.json. Exits 0 iff passed.
#
# Note: phase-0-baseline.md cites "scratch/results/" as the source; the
# actual on-disk location of the uqgrid reference traces in this repo
# is examples/refs/. Recorded as an open question in the builder report.

using SHA
using Dates
using NPZ

const REPO       = abspath(joinpath(@__DIR__, "..", ".."))
const REF_SRC    = joinpath(REPO, "examples", "refs")
const REF_DST    = joinpath(REPO, "artifacts", "phase0", "references")
const MANIFEST   = joinpath(REF_DST, "manifest.json")
const GATE_OUT   = joinpath(REPO, "artifacts", "phase0", "g2.json")

mkpath(REF_DST)
mkpath(dirname(GATE_OUT))

function sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function git_sha(dir::AbstractString)
    try
        return strip(read(Cmd(`git rev-parse HEAD`, dir=dir), String))
    catch
        return ""
    end
end

# JSON writer (no JSON.jl dependency).
function json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"' || c == '\\'
            print(buf, '\\', c)
        elseif c == '\n'; print(buf, "\\n")
        elseif c == '\r'; print(buf, "\\r")
        elseif c == '\t'; print(buf, "\\t")
        elseif UInt32(c) < 0x20
            print(buf, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else
            print(buf, c)
        end
    end
    String(take!(buf))
end

function json_value(v)
    if v === nothing
        return "null"
    elseif v isa Bool
        return v ? "true" : "false"
    elseif v isa AbstractString
        return string('"', json_escape(v), '"')
    elseif v isa Integer
        return string(v)
    elseif v isa AbstractFloat
        return isfinite(v) ? string(v) : "null"
    elseif v isa AbstractVector
        return string("[", join(json_value.(v), ","), "]")
    elseif v isa AbstractDict
        # Stable iteration: keys sorted by string form.
        ks = sort(collect(keys(v)), by=string)
        parts = String[]
        for k in ks
            push!(parts, string('"', json_escape(String(string(k))), '"', ":", json_value(v[k])))
        end
        return string("{", join(parts, ","), "}")
    else
        return string('"', json_escape(string(v)), '"')
    end
end

function pretty(json::AbstractString)
    # very small pretty-printer: insert newlines after commas/braces.
    buf = IOBuffer()
    indent = 0
    in_string = false
    prev = '\0'
    for c in json
        if in_string
            print(buf, c)
            if c == '"' && prev != '\\'
                in_string = false
            end
        else
            if c == '"'
                in_string = true
                print(buf, c)
            elseif c == '{' || c == '['
                print(buf, c, '\n')
                indent += 2
                print(buf, ' '^indent)
            elseif c == '}' || c == ']'
                indent -= 2
                print(buf, '\n', ' '^indent, c)
            elseif c == ','
                print(buf, c, '\n', ' '^indent)
            elseif c == ':'
                print(buf, c, ' ')
            else
                print(buf, c)
            end
        end
        prev = c
    end
    String(take!(buf))
end

function write_json(path::AbstractString, obj)
    open(path, "w") do io
        write(io, pretty(json_value(obj)))
        write(io, "\n")
    end
end

gradpower_sha = git_sha(REPO)
uqgrid_dir    = joinpath(REPO, "uqgrid")
uqgrid_sha    = isdir(joinpath(uqgrid_dir, ".git")) ? git_sha(uqgrid_dir) : ""

# ---- Copy step (byte-identical). ----
src_files = sort(filter(f -> endswith(f, ".npz"), readdir(REF_SRC)))
isempty(src_files) && error("no reference files found in $REF_SRC")
copied_ok = String[]
copy_errors = String[]
for f in src_files
    src = joinpath(REF_SRC, f)
    dst = joinpath(REF_DST, f)
    try
        if !isfile(dst) || sha256_file(src) != sha256_file(dst)
            cp(src, dst; force=true)
        end
        if sha256_file(src) == sha256_file(dst)
            push!(copied_ok, f)
        else
            push!(copy_errors, "$f: SHA mismatch after copy")
        end
    catch e
        push!(copy_errors, "$f: $(sprint(showerror, e))")
    end
end

# ---- Build manifest entries. ----
entries = Dict{String,Any}()
metadata_errors = String[]
for f in src_files
    src = joinpath(REF_SRC, f)
    sha = sha256_file(src)
    embedded = nothing
    if f in ("2bus_genrou.npz", "2bus_gensal.npz", "2bus_ieesgo.npz",
             "2bus_tgov1.npz", "2bus_sexs.npz", "2bus_esdc1a.npz",
             "2bus_esdc1a_reduced.npz")
        try
            arrays = npzread(src)
            haskey(arrays, "metadata_json") || error("metadata_json is missing")
            metadata_text = String(UInt8.(vec(arrays["metadata_json"])))
            occursin("\"schema_version\":1", metadata_text) || error("schema_version is missing")
            occursin("\"state_manifest\"", metadata_text) || error("state_manifest is missing")
            length(vec(arrays["z0"])) == size(arrays["history"], 1) || error("z0/history dimension mismatch")
            embedded = Dict("schema_version" => 1, "self_describing" => true)
        catch e
            push!(metadata_errors, "$f: $(sprint(showerror, e))")
        end
    end
    entries[f] = Dict{String,Any}(
        "source_sha256"              => sha,
        "uqgrid_commit_sha"          => isempty(uqgrid_sha) ? nothing : uqgrid_sha,
        "gradpower_validated_at_sha" => isempty(gradpower_sha) ? nothing : gradpower_sha,
        "source_path"                => relpath(src, REPO),
        "frozen_at"                  => string(Dates.now(Dates.UTC)),
        "embedded_metadata"          => embedded,
    )
end

manifest_obj = Dict{String,Any}(
    "schema_version" => 1,
    "phase" => 0,
    "uqgrid_commit_sha" => isempty(uqgrid_sha) ? nothing : uqgrid_sha,
    "gradpower_validated_at_sha" => isempty(gradpower_sha) ? nothing : gradpower_sha,
    "references" => entries,
)
write_json(MANIFEST, manifest_obj)

# ---- Verification. ----
# Criterion 1: every src .npz copied byte-identical.
all_copied = isempty(copy_errors) && length(copied_ok) == length(src_files)
# Criterion 2: manifest exists and every entry has the three required fields.
manifest_ok = isfile(MANIFEST)
missing_fields = String[]
for f in src_files
    e = entries[f]
    for k in ("source_sha256", "uqgrid_commit_sha", "gradpower_validated_at_sha")
        if e[k] === nothing || (e[k] isa AbstractString && isempty(e[k]))
            push!(missing_fields, "$f.$k")
        end
    end
end
fields_ok = isempty(missing_fields)
metadata_ok = isempty(metadata_errors)

criteria = Any[
    Dict("name" => "all_refs_copied_byte_identical",
         "value" => length(copied_ok), "threshold" => length(src_files),
         "passed" => all_copied),
    Dict("name" => "manifest_present",
         "value" => manifest_ok ? 1 : 0, "threshold" => 1,
         "passed" => manifest_ok),
    Dict("name" => "manifest_entries_complete",
         "value" => length(src_files) - length(missing_fields), "threshold" => length(src_files),
         "passed" => fields_ok),
    Dict("name" => "focused_refs_self_describing",
         "value" => 7 - length(metadata_errors), "threshold" => 7,
         "passed" => metadata_ok),
]

passed = all(c["passed"] for c in criteria)

result = Dict{String,Any}(
    "phase" => 0,
    "gate"  => "G2",
    "passed" => passed,
    "criteria" => criteria,
    "metadata" => Dict{String,Any}(
        "git_sha" => gradpower_sha,
        "uqgrid_commit_sha" => uqgrid_sha,
        "n_refs" => length(src_files),
        "copy_errors" => copy_errors,
        "missing_fields" => missing_fields,
        "metadata_errors" => metadata_errors,
        "ref_dst" => REF_DST,
        "manifest_path" => MANIFEST,
    ),
)

write_json(GATE_OUT, result)
println("phase0 G2: passed=", passed, "  refs=", length(src_files),
        "  manifest=", MANIFEST)
exit(passed ? 0 : 1)
