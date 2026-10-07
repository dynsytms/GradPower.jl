#!/usr/bin/env julia
# Phase 7 G3 — No dangling references.
#
# grep -rn 'adjoint\|jacp_vec\|jacpt_vec\|sensitivities' src/
# returns no hits outside comments and the nlp.jl stub.
#
# Writes artifacts/phase7/g3.json. Exits 0 iff passed.

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase7", "g3.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t_start = time()
    src_dir = joinpath(REPO, "src")

    # Run grep for the forbidden patterns
    patterns = ["adjoint", "jacp_vec", "jacpt_vec", "sensitivities"]
    combined = join(patterns, "\\|")
    cmd = `grep -rn $combined $src_dir`
    raw_hits = try
        read(cmd, String)
    catch
        # grep exits 1 when no matches — that's ideal
        ""
    end

    # Filter out comment lines and the nlp.jl stub (which is allowed)
    bad_hits = String[]
    for line in split(raw_hits, '\n')
        isempty(strip(line)) && continue
        # Allow comment lines (lines where first non-space after filename:lineno: is #)
        m = match(r"^[^:]+:\d+:\s*#", line)
        m !== nothing && continue
        # Allow the nlp.jl error stub
        if occursin("nlp.jl", line) && occursin("error(", lowercase(line))
            continue
        end
        # Allow the nlp.jl error string continuation line
        if occursin("nlp.jl", line) && occursin("Phase 15", line)
            continue
        end
        push!(bad_hits, line)
    end

    ok = isempty(bad_hits)
    detail = ok ? "no dangling references" : "$(length(bad_hits)) dangling hit(s)"
    if !ok
        for h in bad_hits
            println("  DANGLING: $h")
        end
    end

    criteria = [Dict("name" => "no_dangling_refs", "value" => length(bad_hits),
                     "threshold" => 0, "passed" => ok)]
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 7, "G3", ok, criteria, metadata)
    println("Phase 7 G3: passed=$ok  -> $OUT  $detail")
    return ok
end

exit(main() ? 0 : 1)
