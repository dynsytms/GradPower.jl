mutable struct DynamicNLP{T, S} <: NLPModels.AbstractNLPModel{T, S}
    sys::PowerSystem
    prob::DynamicProblem
    tfinal::T
    λ0::Vector{T}
    meta::NLPModels.NLPModelMeta{T, S}
    counters::NLPModels.Counters
end

function DynamicNLP(
    sys::PowerSystem,
    dprob::DynamicProblem,
    tfinal::Float64,
    lvar::AbstractArray,
    uvar::AbstractArray;
    counters=NLPModels.Counters()
)
    nvar = length(dprob.pvec)
    ncon = 0
    λ0 = zeros(length(dprob.zvec))
    meta = NLPModels.NLPModelMeta(
        nvar,
        ncon=ncon,
        name="Optimal control problem",
        lvar=lvar,
        uvar=uvar,
        minimize=true,
    )
    return DynamicNLP(sys, dprob, tfinal, λ0, meta, counters)
end

function NLPModels.obj(nlp::DynamicNLP{T, S}, p::AbstractVector) where {T, S}
    prob = DynamicProblem(nlp.sys)
    GradPower.initialize_dynamics!(prob, nlp.sys)
    copyto!(prob.pvec, p)
    tvec, traj = integrate!(prob, nlp.sys, nlp.tfinal)
    tf = size(traj, 2)

    uvec, pvec = prob.uvec, prob.pvec
    # Integrate objective along time
    val = zero(T)
    for t in 1:tf-1
        x = view(traj, :, t+1)
        rfun = functional(x, uvec, pvec, nlp.sys)
        val += (tvec[t+1] - tvec[t]) * rfun
    end
    return val
end

function NLPModels.grad!(nlp::DynamicNLP, x::AbstractVector, g::AbstractVector)
    prob = DynamicProblem(nlp.sys)
    GradPower.initialize_dynamics!(prob, nlp.sys)
    copyto!(prob.pvec, x)
    tvec, traj = integrate!(prob, nlp.sys, nlp.tfinal)
    λ, μ = adjoint(nlp.λ0, prob, nlp.sys, traj, tvec; functional=true)
    copyto!(g, μ)
    return g
end

# Dummy function required to use dense BFGS
function MadNLP.jac_dense!(nlp::DynamicNLP, x::AbstractVector, jac::AbstractMatrix)
end

