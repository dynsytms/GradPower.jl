function compare_matrix(A::AbstractMatrix, B::AbstractMatrix; rtol=1e-2, atol=1e-9)
    # Check if dimensions are the same
    if size(A) != size(B)
        println("Dimensions do not match.")
        return false
    end

    rows, cols = size(A)

    mism = 0

    # Check each element
    for i in 1:rows
        for j in 1:cols
            a, b = A[i, j], B[i, j]

            # Calculate absolute difference
            abs_diff = abs(a - b)

            # If both numbers are very close to zero, then proceed to next iteration
            if abs(a) < atol && abs(b) < atol
                continue
            end

            # Avoid division by zero or very small number
            max_val = max(abs(a), abs(b))
            if max_val < atol
                max_val = atol  # Prevents division by zero or very small number
            end
            
            # Calculate relative difference
            rel_diff = abs_diff / max_val

            # Check if the differences are within tolerances
            if abs_diff > atol && rel_diff > rtol
                println("Element mismatch at row $i, column $j: $a (A) vs $b (B)")
                mism += 1
            end
        end
    end

    valid = (mism == 0)
    return valid
end

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
