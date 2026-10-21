"""
    GradPower.DynamicNLP(sys, dprob, tfinal, lvar, uvar; counters)

Construct the optional NLPModels model after loading `NLPModels` and `MadNLP`.
The concrete model type lives in `GradPowerNLPExt`; this entry point is a
constructor function, not a type. Forward objective evaluation is supported,
but gradients still raise an error until discrete adjoints are implemented.
"""
function DynamicNLP(args...; kwargs...)
    throw(ArgumentError("DynamicNLP requires the optional packages NLPModels and MadNLP; " *
                        "install them and run `using NLPModels, MadNLP` first."))
end
