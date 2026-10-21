# Build the documentation:  julia --project=docs docs/make.jl
# (first time: julia --project=docs -e 'using Pkg; Pkg.instantiate()')
using Documenter
using GradPower
using CairoMakie

DocMeta.setdocmeta!(GradPower, :DocTestSetup, :(using GradPower); recursive=true)

makedocs(;
    modules  = [GradPower],
    sitename = "GradPower.jl",
    authors  = "D. Adrian Maldonado, François Pacaud, Michel Schanen",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://dynsytms.github.io/GradPower.jl",
        edit_link  = "main",
        assets     = String[],
        size_threshold = 400 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "First simulation" => "tutorials/first_simulation.md",
            "Reading the results" => "tutorials/state_vector.md",
            "Contingencies and events" => "tutorials/events.md",
            "Many scenarios at once" => "tutorials/batched.md",
        ],
        "Guides" => [
            "Queries and parameter edits" => "guides/queries.md",
            "Plotting simulation results" => "guides/plotting.md",
            "Integrators and Newton options" => "guides/integrators.md",
            "Recording and early stop" => "guides/recording.md",
            "Linear solvers and GPU" => "guides/solvers_gpu.md",
            "Generating training data" => "guides/datasets.md",
            "Supported models and input files" => "guides/models.md",
        ],
        "How it works" => "theory.md",
        "API reference" => "api.md",
    ],
    # Reference only what is documented; do not fail the build on docstrings
    # that are not listed in a page (internal helpers).
    checkdocs = :none,
    warnonly  = [:missing_docs, :cross_references],
)

deploydocs(;
    repo = "github.com/dynsytms/GradPower.jl.git",
    devbranch = "main",
    push_preview = false,
)
