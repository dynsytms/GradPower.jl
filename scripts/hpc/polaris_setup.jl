# One-time build of the Polaris GPU Julia environment ($GP_ENV, default ~/gpenv):
# GradPower + CUDA + CUDSS.
#
#   source scripts/hpc/polaris_env.sh
#   julia --project=$GP_ENV scripts/hpc/polaris_setup.jl

using Pkg

gp_env  = get(ENV, "GP_ENV", joinpath(homedir(), "gpenv"))
gp_repo = get(ENV, "GRADPOWER_DIR", abspath(joinpath(@__DIR__, "..", "..")))

@info "GradPower checkout: $gp_repo"
@info "Julia environment:  $gp_env"
isfile(joinpath(gp_repo, "Project.toml")) ||
    error("no Project.toml at $gp_repo — set GRADPOWER_DIR to the repo root")

prefs = joinpath(gp_env, "LocalPreferences.toml")
if isfile(prefs)
    @warn "removing $prefs (may pin a CUDA artifact runtime)"
    rm(prefs)
end

mkpath(gp_env)
Pkg.activate(gp_env)
Pkg.develop(path = gp_repo)
Pkg.add(["CUDA", "CUDSS"])
Pkg.instantiate()
Pkg.precompile()

using CUDA
@info "CUDA.functional() = $(CUDA.functional())"
if CUDA.functional()
    @info "CUDA runtime = $(CUDA.runtime_version())"
    CUDA.versioninfo()
    using CUDSS
    @info "CUDSS loaded from $(get(ENV, "JULIA_CUDSS_LIBRARY_PATH", "<default>"))"
    using GradPower
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    ext === nothing ? error("GradPowerCUDAExt failed to load") :
        @info "GradPowerCUDAExt loaded — environment ready"
else
    @warn "CUDA is not functional here — are you on a compute node?"
end
