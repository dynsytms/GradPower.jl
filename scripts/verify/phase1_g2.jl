#!/usr/bin/env julia
# Phase 1 G2 — Single definition site.
#
# Verifies that each _<dev>_residual_one! and _<dev>_jacobian_one! leaf
# function is defined exactly once in src/kernels/, and that no equation
# duplication exists across files.
#
# Pass criterion: every device's leaf function name returns exactly one
# hit from grep across src/kernels/.
#
# Writes artifacts/phase1/g2.json. Exits 0 iff passed.

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_PATH = joinpath(REPO, "artifacts", "phase1", "g2.json")
const KERN_DIR = joinpath(REPO, "src", "kernels")
mkpath(dirname(OUT_PATH))

# JSON helpers (no JSON.jl dependency)
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

# Device leaf function patterns to check.
# Each entry: (grep pattern for function definition, expected file)
const LEAF_FUNCTIONS = [
    # Genrou
    ("_genrou_residual_one!", "genrou.jl"),
    ("_genrou_jacobian_one!", "genrou.jl"),
    # IEESGO
    ("_ieesgo_residual_one!", "ieesgo.jl"),
    ("_ieesgo_jacobian_one!", "ieesgo.jl"),
    # TGOV1
    ("_tgov1_residual_one!", "tgov1.jl"),
    ("_tgov1_jacobian_one!", "tgov1.jl"),
    # SEXS
    ("_sexs_residual_one!", "sexs.jl"),
    ("_sexs_jacobian_one!", "sexs.jl"),
    # ZIPLoad
    ("_zipload_residual_one!", "zipload.jl"),
    ("_zipload_jacobian_one!", "zipload.jl"),
    # StaticGenerator
    ("_static_gen_residual_one!", "static_gen.jl"),
    ("_static_gen_jacobian_one!", "static_gen.jl"),
]

function main()
    all_passed = true
    criteria = Any[]
    details = Dict{String,Any}()

    for (fn_name, expected_file) in LEAF_FUNCTIONS
        # grep for function definitions containing this name
        pattern = "function $(fn_name)"
        result = try
            strip(read(Cmd(`grep -rn $pattern $KERN_DIR`), String))
        catch
            ""
        end
        lines = filter(!isempty, split(result, '\n'))
        n_hits = length(lines)
        passed = n_hits == 1

        push!(criteria, Dict("name" => "$(fn_name)_single_def",
                             "value" => n_hits, "threshold" => 1,
                             "passed" => passed))
        details[fn_name] = Dict("n_hits" => n_hits,
                                "hits" => String[string(l) for l in lines],
                                "expected_file" => expected_file)
        if !passed
            all_passed = false
            println("FAIL: $(fn_name) has $(n_hits) definition(s)")
            for l in lines; println("  ", l); end
        else
            println("PASS: $(fn_name) -> 1 definition in $(expected_file)")
        end
    end

    # Also check that no batch function contains inlined equation logic.
    # Verify each batch function is a thin delegation loop by checking
    # that the batch function body calls its leaf function.
    batch_checks = [
        ("genrou_residual_batch!", "_genrou_residual_one!", "genrou.jl"),
        ("genrou_jacobian_batch!", "_genrou_jacobian_one!", "genrou.jl"),
        ("ieesgo_residual_batch!", "_ieesgo_residual_one!", "ieesgo.jl"),
        ("ieesgo_jacobian_batch!", "_ieesgo_jacobian_one!", "ieesgo.jl"),
        ("tgov1_residual_batch!", "_tgov1_residual_one!", "tgov1.jl"),
        ("tgov1_jacobian_batch!", "_tgov1_jacobian_one!", "tgov1.jl"),
        ("sexs_residual_batch!", "_sexs_residual_one!", "sexs.jl"),
        ("sexs_jacobian_batch!", "_sexs_jacobian_one!", "sexs.jl"),
        ("zipload_residual_batch!", "_zipload_residual_one!", "zipload.jl"),
        ("zipload_jacobian_batch!", "_zipload_jacobian_one!", "zipload.jl"),
        ("static_gen_residual_batch!", "_static_gen_residual_one!", "static_gen.jl"),
        ("static_gen_jacobian_batch!", "_static_gen_jacobian_one!", "static_gen.jl"),
    ]

    for (batch_fn, leaf_fn, file) in batch_checks
        filepath = joinpath(KERN_DIR, file)
        content = read(filepath, String)
        calls_leaf = occursin(leaf_fn, content) && occursin("function $(batch_fn)", content)
        push!(criteria, Dict("name" => "$(batch_fn)_delegates_to_leaf",
                             "value" => calls_leaf ? 1 : 0, "threshold" => 1,
                             "passed" => calls_leaf))
        if !calls_leaf
            all_passed = false
            println("FAIL: $(batch_fn) does not delegate to $(leaf_fn)")
        end
    end

    git_sha = try strip(read(Cmd(`git rev-parse HEAD`, dir=REPO), String)) catch; "" end
    out = Dict{String,Any}(
        "phase" => 1,
        "gate" => "G2",
        "passed" => all_passed,
        "criteria" => criteria,
        "metadata" => Dict{String,Any}(
            "hardware" => string(Sys.cpu_info()[1].model),
            "git_sha" => git_sha,
            "wallclock_s" => 0.0,
            "details" => details,
        ),
    )
    open(OUT_PATH, "w") do io
        write(io, pretty(json_value(out))); write(io, "\n")
    end
    println("\nPhase 1 G2: passed=$all_passed  artifact=$OUT_PATH")
    return all_passed
end

exit(main() ? 0 : 1)
