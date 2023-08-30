function build_network!(psys::PowerSystem)
    # Create the adjacency list
    adjacency = adjacency_list(psys)

    # Create the ybus matrix
    ybus = create_ybus_complex(psys)

    # Create real ybus matrix
    ybus_re = realify_ybus(ybus)

    # Create the network
    psys.network = Network(adjacency, ybus, ybus_re)
end

function adjacency_list(sys::PowerSystem)
    # Initialize an empty Vector of Vectors to store the adjacency list
    adjacency_list = [Int[] for _ in 1:length(sys.buses)]

    # Populate the adjacency list
    for branch in sys.branches
        push!(adjacency_list[branch.fr], branch.to)
        push!(adjacency_list[branch.to], branch.fr)
    end
    # Remove duplicates and sort each list
    adjacency_list = [sort(unique(list)) for list in adjacency_list]
    return adjacency_list
end

function create_ybus_complex(psys::PowerSystem)
    dim = length(psys.buses)
    ybus_dict = Dict{Int, Dict{Int, ComplexF64}}()

    for branch in psys.branches
        tap = branch.tap
        shift = branch.shift

        if tap > 0.0
            tpsh = tap*exp(im*π/180.0*shift)
        else
            tpsh = 1.0
            tap = 1.0
        end

        fr = branch.fr
        to = branch.to
        y = 1.0/(branch.r + im*branch.x)

        ybus_dict[fr] = get(ybus_dict, fr, Dict{Int, ComplexF64}())
        ybus_dict[to] = get(ybus_dict, to, Dict{Int, ComplexF64}())

        ybus_dict[fr][fr] = get(ybus_dict[fr], fr, 0.0) + y/(tap*tap)
        ybus_dict[to][to] = get(ybus_dict[to], to, 0.0) + y
        ybus_dict[fr][to] = get(ybus_dict[fr], to, 0.0) - y/conj(tpsh)
        ybus_dict[to][fr] = get(ybus_dict[to], fr, 0.0) - y/tpsh

        # charging susceptance
        ybus_dict[to][to] += (im*0.5*branch.sh)/(tap*tap)
        ybus_dict[fr][fr] += im*0.5*branch.sh
    end

    for shunt in psys.shunts
        bus = shunt.bus
        ybus_dict[bus] = get(ybus_dict, bus, Dict{Int, ComplexF64}())
        ybus_dict[bus][bus] = get(ybus_dict[bus], bus, 0.0) + shunt.gsh + im*shunt.bsh
    end

    # Create CSC format arrays
    rows = Int[]
    cols = Int[]
    data = ComplexF64[]

    # iterate again to fill the arrays
    for (frbus, frbus_dict) in ybus_dict
        for (tobus, ybus_value) in frbus_dict
            push!(rows, frbus)
            push!(cols, tobus)
            push!(data, ybus_value)
        end
    end

    # Create sparse matrix
    ybus_spa = sparse(rows, cols, data, dim, dim)

    return ybus_spa
end

function realify_ybus(ybus::SparseMatrixCSC)
    nbuses = size(ybus, 1)
    nnz = length(ybus.nzval)

    new_val = zeros(Float64, 4*nnz)
    new_row = zeros(Int, 4*nnz)
    new_col = zeros(Int, 4*nnz)

    c_idx = 1
    for col in 1:nbuses
        for idx in ybus.colptr[col]:(ybus.colptr[col+1]-1)
            row = ybus.rowval[idx]
            entry = ybus.nzval[idx]
            
            zr = real(entry)
            zi = imag(entry)
            
            new_row[c_idx] = 2*row - 1
            new_col[c_idx] = 2*col - 1
            new_val[c_idx] = zr
            
            new_row[c_idx + 1] = 2*row
            new_col[c_idx + 1] = 2*col
            new_val[c_idx + 1] = zr
            
            new_row[c_idx + 2] = 2*row - 1
            new_col[c_idx + 2] = 2*col
            new_val[c_idx + 2] = -zi
            
            new_row[c_idx + 3] = 2*row
            new_col[c_idx + 3] = 2*col - 1
            new_val[c_idx + 3] = zi
            
            c_idx += 4
        end
    end

    rybus = sparse(new_row, new_col, new_val, 2*nbuses, 2*nbuses)
    return rybus
end
