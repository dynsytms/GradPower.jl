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
    counters=Counters()
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
    return DynamicNLP(sys, dprob, tfinal, meta, counters)
end

function NLPModels.obj(nlp::DynamicNLP{T, S}, x::AbstractVector) where {T, S}
    copyto!(nlp.prob.pvec, p)
    tvec, traj = integrate!(nlp.prob, nlp.sys, nlp.tfinal)
    tf = size(traj, 2)

    u, p = nlp.prob.uvec, nlp.prob.pvec
    # Integrate objective along time
    val = zero(T)
    for t in 1:tf-1
        x = view(traj, :, t+1)
        rfun = functional(x, u, p, nlp.sys)
        val += (tvec[t+1] - tvec[t]) * rfun
    end
    return val
end

function NLPModels.grad!(nlp::DynamicNLP, x::AbstractVector, g::AbstractVector)
    copyto!(nlp.prob.pvec, p)
    tvec, traj = integrate!(nlp.prob, nlp.sys, nlp.tfinal)
    state_idx = 4
    nlp.λ0[state_idx] = 1.0
    λ, μ = adjoint(nlp.λ0, nlp.prob, nlp.sys, traj, tvec; functional=true)
    return μ
end

# Dummy function required to use dense BFGS
function MadNLP.jac_dense!(nlp::DynamicNLP, x::AbstractVector, jac::AbstractMatrix)
end

