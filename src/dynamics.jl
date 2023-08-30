include("generators.jl")
include("loads.jl")

function initialize_dynamics!(dp::DynamicProblem, ps::PowerSystem)

    z = dp.zvec
    p = dp.pvec
    u = dp.uvec

    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    ctrl_dim = ps.dynamic.ctrl_dim
    par_dim = ps.dynamic.par_dim
    nbus = length(ps.buses)

    x = @view z[1:diff_dim]
    y = @view z[diff_dim+1:diff_dim+alg_dim]

    # initialize voltages
    for i in 1:nbus
        vm = ps.buses[i].v0m
        va = ps.buses[i].v0a
        z[diff_dim + alg_dim + 2*(i - 1) + 1] = vm*cos(va)
        z[diff_dim + alg_dim + 2*(i - 1) + 2] = vm*sin(va)
    end

    # TODO: write initialization function for load
    # set initial voltage magnitude in ZIPLoad devices.
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa ZIPLoad
            device.dtype.v0mag = ps.buses[ps.dynamic.map.bus[i]].v0m
            yload = (device.dtype.pinj + 1im*device.dtype.qinj)/(device.dtype.v0mag^2.0)
            device.dtype.yreal = real(yload)
            device.dtype.yimag = imag(yload)
        end
    end
    
    # fill parameter vector.
    # TODO: I need to re-think this design. To fill pvec I first need to set the load parameters with the
    # updated power-flow solution. But Ideally I should initialize all the devices after I fill the pvec.
    # In this case the load becomes an special case.
    for (i, device) in enumerate(ps.dynamic.devices)
        fill_pvec!(@view(dp.pvec[device.par_ptr:device.par_ptr+device.dtype.par_size-1]), device.dtype)
    end

    # TODO: i can do this generic but I need to ensure that generators are initialized before
    # associated controllers (e.g., governor) such that these can retrieve torque and field voltage.
    map = ps.dynamic.map
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa AbstractGeneratorType
            # retrieve pointers and sizes
            diff_ptr = device.diff_ptr
            alg_ptr = device.alg_ptr
            ctrl_ptr = device.ctrl_ptr
            par_ptr = device.par_ptr

            diff_size = device.dtype.diff_size
            alg_size = device.dtype.alg_size
            ctrl_size = device.dtype.ctrl_size
            par_size = device.dtype.par_size
            # parameter view
            pview = @view(p[par_ptr:par_ptr+par_size-1])
            # find generator in static system and retrieve power injection.
            gen = ps.gens[map.gen[i]]
            pg = gen.psch
            qg = gen.qsch
            # retrieve voltage magnitude and angle.
            vm = ps.buses[map.bus[i]].v0m
            va = ps.buses[map.bus[i]].v0a
            # allocate initializing vector.
            # NOTE: I would prefer not to allocate. Not sure how to do it more efficiently since
            # I need to separate zvec and uvec.
            xinit = zeros(Float64, device.dtype.diff_size + device.dtype.alg_size + device.dtype.ctrl_size)
            # initial guess and initialization
            initial_guess!(xinit, pview, pg, qg, vm, va, device.dtype)
            rhs_fun!(f, x) = initialize_dynamics!(f, x, pview, pg, qg, vm, va, device.dtype)
            sol = nlsolve(rhs_fun!, xinit, ftol=1e-12, iterations=30, autodiff = :forward)
            xinit .= sol.zero
            # copy to zvec and uvec
            x[diff_ptr:diff_ptr+diff_size-1] .= xinit[1:diff_size]
            y[alg_ptr:alg_ptr+alg_size-1] .= xinit[diff_size+1:diff_size+alg_size]
            u[ctrl_ptr:ctrl_ptr+ctrl_size-1] .= xinit[diff_size+alg_size+1:diff_size+alg_size+ctrl_size]
        end
    end
end

function rhs_fun!(f::AbstractArray, z::AbstractArray, u::AbstractArray, p::AbstractArray, sys::PowerSystem)
    map = sys.dynamic.map

    diff_dim = sys.dynamic.diff_dim
    alg_dim = sys.dynamic.alg_dim
    ctrl_dim = sys.dynamic.ctrl_dim
    par_dim = sys.dynamic.par_dim

    x = @view z[1:diff_dim]
    y = @view z[diff_dim+1:diff_dim+alg_dim]
    v = @view z[diff_dim+alg_dim+1:end]

    # network balance
    #f[diff_dim+alg_dim+1:end] .= -sys.network.ybus_real*v
    fv = @view f[diff_dim+alg_dim+1:end]
    mul!(fv, sys.network.ybus_real, v, -1.0, 0.0)


    @inbounds for (i, device) in enumerate(sys.dynamic.devices)
        bus = map.bus[i]

        # retrieve pointers and sizes from map
        diff_ptr = map.diff_ptr[i]
        alg_ptr = map.alg_ptr[i]
        ctrl_ptr = map.ctrl_ptr[i]
        par_ptr = map.par_ptr[i]

        diff_size = map.diff_size[i]
        alg_size = map.alg_size[i]
        ctrl_size = map.ctrl_size[i]
        par_size = map.par_size[i]

        # retrieve local views
        diff = @view x[diff_ptr:diff_ptr+diff_size-1]
        alg = @view y[alg_ptr:alg_ptr+alg_size-1]
        ctrl = @view u[ctrl_ptr:ctrl_ptr+ctrl_size-1]
        par = @view p[par_ptr:par_ptr+par_size-1]
        vloc = @view v[2*bus-1:2*bus]

        f_diff = @view f[diff_ptr:diff_ptr+diff_size-1]
        f_alg = @view f[diff_dim+alg_ptr:diff_dim+alg_ptr+alg_size-1]
        f_net = @view f[diff_dim+alg_dim+2*(bus-1)+1:diff_dim+alg_dim+2*(bus-1)+2]

        # call rhs function
        cinject!(f_net, diff, alg, ctrl, par, vloc, device.dtype)
        rhs_fun!(f_diff, f_alg, diff, alg, ctrl, par, vloc, device.dtype)
    end

    @inbounds for (i, event) in enumerate(sys.dynamic.events)
        if event.status
            bus = event.bus
            vr = v[2*(bus-1)+1]
            vi = v[2*(bus-1)+2]
            yfault = 1.0/event.rfault
            f[diff_dim+alg_dim+2*(bus-1)+1] -= yfault*vr
            f[diff_dim+alg_dim+2*(bus-1)+2] -= yfault*vi
        end
    end
end

function beuler!(
    f::AbstractVector,
    z::AbstractVector,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    diff_dim::Int64,
    dt::Float64
)
    rhs_fun!(f, z, u, p, sys)
    @inbounds for i = 1:diff_dim
        f[i] = z[i] - zold[i] - dt*f[i]
    end
end

function beuler_jac!(
    J::SparseMatrixCSC,
    z::AbstractVector,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    diff_dim::Int64,
    dt::Float64
)
    # set all elements to zero
    fill!(J.nzval, 0.0)

    # evaluate Jacobian
    rhs_jac!(J, z, u, p, sys)

    # scale for backward Euler
    _jacobian_beuler!(J, diff_dim, dt)
end

function _jacobian_beuler!(J::SparseMatrixCSC, NDIFFEQ::Int, h::Float64)
    # Iterating through each column
    for col = 1:size(J, 2)
        # Flag to check if diagonal element for the column is found
        diagonal_found = false
        
        # Iterating through the non-zero elements in each column
        for row_index in nzrange(J, col)
            row = rowvals(J)[row_index]
            
            # Update values if the row index is less or equal to NDIFFEQ
            if row <= NDIFFEQ
                J.nzval[row_index] *= -h
                
                # Update diagonal element
                if row == col
                    J.nzval[row_index] += 1.0
                    diagonal_found = true
                end
           end
        end

        # If diagonal element was not found and col is within NDIFFEQ, then add it
        if !diagonal_found && col <= NDIFFEQ
            @warn "Diagonal element not found for column $col. Adding it."
            J[col, col] += 1.0
        end
    end
end

function integrate!(dp::DynamicProblem, ps::PowerSystem, tf::Float64; dt::Float64=(1.0/120.0))

    # TODO: here we should have some checks to ensure that the problem is initialized

    # calculate number of steps
    nsteps = Int(round(tf/dt))
    tvec = collect(0:dt:tf)

    # retrieve events.
    # TODO: we can only do one event right now.
    events = ps.dynamic.events
    @assert length(events) <= 1
    ton = toff = tf + dt
    step_on = step_off = nsteps + 1
    if length(events) == 1
        event = events[1]
        ton = event.ton
        toff = event.toff
        step_on = Int(round(ton/dt))
        step_off = Int(round(toff/dt))
    end

    # retrieve sizes
    nbus = length(ps.buses)
    system_size = ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*nbus
    @assert system_size == length(dp.zvec)

    # allocate solution trajectory.
    traj = zeros(Float64, system_size, nsteps+1)

    # allocate temporary vectors
    zold = zeros(Float64, system_size)
    println("Integrating from t = 0 s to t = $tf s with dt = $dt s.")
    # newton parameters
    ftol = 1e-9
    max_iter = 30

    # initial condition
    zold .= dp.zvec
    traj[:,1] .= dp.zvec

    # initialize residual and Jacobian
    f0 = zeros(Float64, system_size)
    J0 = preallocate_jacobian(ps)
    
    # time loop
    for k in 1:nsteps
        @printf("Time-stepping. t = %.2f s.\n", tvec[k])
        f_beuler!(f, z) = beuler!(f, z, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, dt)
        j_beuler!(J, z) = beuler_jac!(J, z, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, dt)
        df = OnceDifferentiable(f_beuler!, j_beuler!, zold, f0, J0)
        sol = nlsolve(df, zold, method=:newton, iterations=max_iter, ftol=ftol)
        #sol = nlsolve(f_beuler!, zold, ftol=ftol, iterations=max_iter)
        @printf("Converged in %d iterations. Residual norm: %.2e.\n", sol.iterations, sol.residual_norm)
        zold .= sol.zero
        traj[:,k+1] .= zold

        if k == step_on
            activate!(events[1])
            println("Event activated at t = $ton s. Fault at bus $(events[1].bus).")
            f_beuler_on!(f, z) = beuler!(f, z, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, 0.0)
            sol = nlsolve(f_beuler_on!, zold, ftol=ftol, iterations=max_iter)
            zold .= sol.zero
            @printf("ALG. Converged in %d iterations. Residual norm: %.2e.\n", sol.iterations, sol.residual_norm)
        elseif k == step_off
            deactivate!(events[1])
            println("Event deactivated at t = $toff s.")
            f_beuler_off!(f, z) = beuler!(f, z, zold, dp.uvec, dp.pvec, ps, ps.dynamic.diff_dim, 0.0)
            sol = nlsolve(f_beuler_off!, zold, ftol=ftol, iterations=max_iter)
            @printf("ALG. Converged in %d iterations. Residual norm: %.2e.\n", sol.iterations, sol.residual_norm)
            zold .= sol.zero
        end

    end
    # ensure fault is deactivated at the end
    for event in events
        deactivate!(event)
    end

    dp.zvec = traj[:,end]
    return tvec, traj
end


function fill_pvec!(pvec::AbstractArray, dtype::AbstractDeviceType)
    @warn "fill_pvec! not implemented for device type $(dtype)"
end

function initial_guess!(
        xinit::AbstractArray,
        p::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::AbstractDeviceType
)
    @warn "initial_guess! not implemented for device type $(dtype)"
end

function initialize_dynamics!(
        f::AbstractArray,
        xinit::AbstractArray,
        p::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::AbstractDeviceType
)
    @warn "initialize_dynamics! not implemented for device type $(dtype)"
end

function rhs_fun!(
        f_diff::AbstractArray,
        f_alg::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::AbstractDeviceType
)
    @warn "rhs_fun! not implemented for device type $(dtype)"
end

function cinject!(
        f::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::AbstractDeviceType
)
    @warn "cinject! not implemented for device type $(dtype)"
end

function preallocate_jacobian(ps::PowerSystem)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    ctrl_dim = ps.dynamic.ctrl_dim
    nbus = length(ps.buses)
    adj = ps.network.adjacency
    map = ps.dynamic.map

    # total system size
    sys_dim = diff_dim + alg_dim + 2*nbus

    # coordinate list for sparse jacobian
    coord_list = [Vector{Int}() for _ in 1:sys_dim]

    # diagonal non-zeros for integrators
    for i in 1:diff_dim
        push!(coord_list[i], i)
    end

    # network equations
    ptr = diff_dim + alg_dim + 1
    for fr in 1:nbus
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1) + 1)
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1) + 1)
        for to in adj[fr]
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1) + 1)
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1) + 1)
        end
    end

    # iterate over devices
    for (i, device) in enumerate(ps.dynamic.devices)
        bus = map.bus[i]
        diff_ptr = map.diff_ptr[i]
        alg_ptr = diff_dim + map.alg_ptr[i]
        volt_ptr = diff_dim + alg_dim + 2*(bus -1) + 1
        ctrl_ptr = map.ctrl_ptr[i]
        preallocate_jacobian!(coord_list, diff_ptr, alg_ptr, ctrl_ptr, volt_ptr, device.dtype)
    end

    # form coordinate lists (row, col, data)
    row = Int[]
    col = Int[]

    for i in 1:length(coord_list)
        if !isempty(coord_list[i])
            append!(row, fill(i, length(coord_list[i])))
            append!(col, coord_list[i])
        end
    end

    data = zeros(length(row))
    Jsp = sparse(row, col, data, sys_dim, sys_dim)
    return Jsp
end

function preallocate_jacobian!(
    coord_list::Vector{Vector{Int}},
    diff_ptr::Int,
    alg_ptr::Int,
    ctrl_ptr::Int,
    volt_ptr::Int,
    dtype::AbstractDeviceType
)
    @warn "preallocate_jacobian! not implemented for $(dtype)"
end


function rhs_jac!(
    jac::AbstractMatrix,
    z::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    sys::PowerSystem)

    diff_dim = sys.dynamic.diff_dim
    alg_dim = sys.dynamic.alg_dim
    ctrl_dim = sys.dynamic.ctrl_dim
    nbus = length(sys.buses)
    adj = sys.network.adjacency
    map = sys.dynamic.map
    
    x = @view z[1:diff_dim]
    y = @view z[diff_dim+1:diff_dim+alg_dim]
    v = @view z[diff_dim+alg_dim+1:end]

    # total system size
    current_injection_jacobian!(jac, sys.network.ybus_real, diff_dim + alg_dim)

    # index vector
    idx_dev = Array{Int}(undef, 7)
    
    # iterate over devices
    for (i, device) in enumerate(sys.dynamic.devices)
        bus = map.bus[i]
        diff_ptr = map.diff_ptr[i]
        alg_ptr = map.alg_ptr[i]
        ctrl_ptr = map.ctrl_ptr[i]
        par_ptr = map.par_ptr[i]
        
        diff_size = map.diff_size[i]
        alg_size = map.alg_size[i]
        ctrl_size = map.ctrl_size[i]
        par_size = map.par_size[i]
        
        diff = @view x[diff_ptr:diff_ptr+diff_size-1]
        alg = @view y[alg_ptr:alg_ptr+alg_size-1]
        ctrl = @view u[ctrl_ptr:ctrl_ptr+ctrl_size-1]
        par = @view p[par_ptr:par_ptr+par_size-1]
        vloc = @view v[2*bus-1:2*bus]

        idx_dev[1] = diff_ptr
        idx_dev[2] = diff_dim + alg_ptr
        idx_dev[3] = alg_dim + diff_dim
        idx_dev[4] = par_ptr
        idx_dev[5] = bus
        idx_dev[6] = ctrl_ptr

        rhs_jac!(jac, diff, alg, ctrl, par, vloc, idx_dev, device.dtype)
    end
end

function rhs_jac!(
    jac::AbstractMatrix,
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    v::AbstractArray,
    idx_dev::Vector{Int},
    dtype::AbstractDeviceType
)
    @warn "rhs_jac! not implemented for $(dtype)"
end

function rhs_jac!(
    jac::AbstractMatrix,
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    v::AbstractArray,
    idx_dev::Vector{Int},
    dtype::ZIPLoad
    )

    dp = idx_dev[1]
    ap = idx_dev[2]
    dev = idx_dev[3]
    pp = idx_dev[4]
    bus = idx_dev[5]

    vr = v[1]
    vi = v[2]

    pl = p[1]
    ql = p[2]
    α = p[3]
    β = p[4]
    γ = p[5]
    yload_real = p[8]
    yload_imag = p[9]

    vm = sqrt(vr^2 + vi^2)
    va = atan(vi, vr)

    row1 = dev + 2 * (bus - 1) + 1
    row2 = dev + 2 * (bus - 1) + 2
    col1 = dev + 2 * (bus - 1) + 1
    col2 = dev + 2 * (bus - 1) + 2

    # Constant admittance contribution
    vm2 = vr^2 + vi^2
    vm2_tld = 0.2

    val1 = -α * yload_real
    val2 = α * yload_imag
    jac[row1, col1] += val1
    jac[row1, col2] += val2
    #csc_add_row!(jac, row_map, row, col, val)

    val1 = -α * yload_imag
    val2 = -α * yload_real
    jac[row2, col1] += val1
    jac[row2, col2] += val2
    #csc_add_row!(jac, row_map, row, col, val)

    # Constant power contribution
    row = dev + 2 * (bus - 1) + 1
    if vm2 > vm2_tld
        val1 = (1-α) * ((ql * vr + pl * vi) * 2 * vr - ql * vm2) / vm2^2
        val2 = (1-α) * ((ql * vr + pl * vi) * 2 * vi - pl * vm2) / vm2^2
        jac[row1, col1] += val1
        jac[row1, col2] += val2
    else
        val1 = (1-α) * (-ql) / vm2_tld
        val2 = (1-α) * (-pl) / vm2_tld
        jac[row1, col1] += val1
        jac[row1, col2] += val2
    end
    #csc_add_row!(jac, row_map, row, col, val)

    row = dev + 2 * (bus - 1) + 2
    if vm2 > vm2_tld
        val1 = (1-α) * ((pl * vr - ql * vi) * 2 * vr - pl * vm2) / vm2^2
        val2 = (1-α) * ((pl * vr - ql * vi) * 2 * vi + ql * vm2) / vm2^2
        jac[row2, col1] += val1
        jac[row2, col2] += val2
    else
        val1 = (1-α) * (-pl) / vm2_tld
        val2 = (1-α) * ql / vm2_tld
        jac[row2, col1] += val1
        jac[row2, col2] += val2
    end
end




"""
    current_injection_jacobian!(ybus::SparseMatrixCSC, jac::SparseMatrixCSC, dev::Int)

Copy the data from `ybus` to `jac` with an offset `dev` for row and column indices.
"""
function current_injection_jacobian!(jac::SparseMatrixCSC, ybus::SparseMatrixCSC, dev::Int)
    n_cols = size(ybus, 2)
    
    ybus_rows = rowvals(ybus)
    ybus_vals = nonzeros(ybus)
    
    jac_rows = rowvals(jac)
    jac_vals = nonzeros(jac)
    
    # Loop through each column
    for col_idx in 1:n_cols
        for i in nzrange(ybus, col_idx)
            row_idx = ybus_rows[i]
            val = -ybus_vals[i]
            
            # Calculate new row and column indices with offset dev
            new_row = row_idx + dev
            new_col = col_idx + dev
            
            # Find corresponding position in jac using nzrange
            for j in nzrange(jac, new_col)
                if jac_rows[j] == new_row
                    jac_vals[j] = val
                    break
                end
            end
        end
    end
end


export Genrou
export from_data_field
