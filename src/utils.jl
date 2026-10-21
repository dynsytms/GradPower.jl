"""
    gen_speeds(sys) -> Vector

Rows of the state vector holding the rotor speed deviation of every GENROU
machine (the rotor angle is the next row).
"""
function gen_speeds(sys::PowerSystem)
    if sys.dynamic == nothing
        return []
    end

    idxs = []
    for (i, device) in enumerate(sys.dynamic.devices)
        ptr = device.diff_ptr
        if device.dtype isa Genrou
            push!(idxs, ptr + 4)
        end
    end
    return idxs
end
