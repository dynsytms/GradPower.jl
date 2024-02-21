struct QuadraticCost <: CostFunctional
end

function functional(
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem,
    func::QuadraticCost
)
    idxs = gen_speeds(sys)
    val = 0.0
    for (i, idx) in enumerate(idxs)
        val += z[idx]^2.0
    end
    return val
end

function rdiff!(
    out::AbstractArray,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem,
    func::QuadraticCost
)
    idxs = gen_speeds(sys)
    for (i, idx) in enumerate(idxs)
        out[idx] = 2.0*z[idx]
    end
end
