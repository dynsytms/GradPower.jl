# Create a mapping from row indices to the positions in the CSC data structure
function create_row_to_pos_array(A::SparseMatrixCSC)
    nrows, ncols = size(A)
    row_map = Vector{Vector{Int}}(undef, nrows)
    for i in 1:nrows
        row_map[i] = Vector{Int}()
    end
    for col in 1:ncols
        for j in nzrange(A, col)
            row = rowvals(A)[j]
            val_idx = j
            push!(row_map[row], val_idx)
        end
    end
    return row_map
end

"""
    csc_set_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, columns::Vector{Int}, values::Vector)

Set the values of the specified row in the CSC matrix `A` using the precomputed `row_map`.
The row to be updated is specified by `row`, and its new values are set according to the `columns` and `values` vectors.
"""
function csc_set_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, columns::Vector{Int}, values::Vector)
    relevant_indices = row_map[row]
    for idx in relevant_indices
        col = findfirst(x -> x > idx, A.colptr) - 1
        update_idx = findfirst(==(col), columns)
        if update_idx !== nothing
            A.nzval[idx] = values[update_idx]
        end
    end
end

"""
    csc_add_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, columns::Vector{Int}, values::Vector)

Add the `values` to the specified row in the CSC matrix `A` using the precomputed `row_map`.
The row to be updated is specified by `row`, and its new values are added according to the `columns` and `values` vectors.
"""
function csc_add_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, columns::Vector{Int}, values::Vector)
    relevant_indices = row_map[row]
    for idx in relevant_indices
        col = findfirst(x -> x > idx, A.colptr) - 1
        update_idx = findfirst(==(col), columns)
        if update_idx !== nothing
            A.nzval[idx] += values[update_idx]
        end
    end
end

"""
    csc_mult_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, scalar::Number)

Multiply all elements in the specified row in the CSC matrix `A` by `scalar` using the precomputed `row_map`.
The row to be updated is specified by `row`.
"""
function csc_mult_row!(A::SparseMatrixCSC, row_map::Vector{Vector{Int}}, row::Int, scalar::Number)
    relevant_indices = row_map[row]
    for idx in relevant_indices
        A.nzval[idx] *= scalar
    end
end
