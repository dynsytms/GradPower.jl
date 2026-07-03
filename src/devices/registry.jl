struct DeviceContract
    name::Symbol
    dtype::Type
    class::Symbol           # :generator, :governor, :exciter, :stabilizer, :load, :static_gen
    diff_size::Int
    alg_size::Int
    ctrl_size::Int
    par_size::Int
    residual_fn::Union{Nothing,Symbol}
    jacobian_fn::Union{Nothing,Symbol}
    residual_pullback::Any  # callable or nothing
    attaches_to::Symbol
end

const DEVICE_CONTRACTS = Dict{Symbol,DeviceContract}()
const DEVICE_CONTRACT_ORDER = Symbol[]

# Aliases map an alternate device name (e.g. :gensal) onto an existing
# canonical contract (e.g. :genrou). The alias resolves to the same
# DeviceContract — no kernels or sizes are duplicated.
const DEVICE_ALIASES = Dict{Symbol,Symbol}()
const DEVICE_ALIAS_ORDER = Symbol[]

function register_contract!(c::DeviceContract)
    if !haskey(DEVICE_CONTRACTS, c.name)
        push!(DEVICE_CONTRACT_ORDER, c.name)
    end
    DEVICE_CONTRACTS[c.name] = c
    return c
end

function register_alias!(alias::Symbol, target::Symbol)
    haskey(DEVICE_CONTRACTS, target) ||
        error("register_alias!: target contract :$target is not registered")
    haskey(DEVICE_CONTRACTS, alias) &&
        error("register_alias!: :$alias is already a canonical contract")
    if !haskey(DEVICE_ALIASES, alias)
        push!(DEVICE_ALIAS_ORDER, alias)
    end
    DEVICE_ALIASES[alias] = target
    return alias
end

get_contract(name::Symbol) = haskey(DEVICE_ALIASES, name) ?
    DEVICE_CONTRACTS[DEVICE_ALIASES[name]] : DEVICE_CONTRACTS[name]
list_contracts() = DeviceContract[DEVICE_CONTRACTS[n] for n in DEVICE_CONTRACT_ORDER]
list_aliases() = Pair{Symbol,Symbol}[a => DEVICE_ALIASES[a] for a in DEVICE_ALIAS_ORDER]
