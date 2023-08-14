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

    # set initial voltage magnitude in ZIPLoad devices.
    for (i, device) in enumerate(ps.dynamic.devices)
        if device.dtype isa ZIPLoad
            device.dtype.v0mag = ps.buses[ps.dynamic.map.bus[i]].v0m
        end
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
            nlsolve(rhs_fun!, xinit, ftol=1e-12, iterations=30, autodiff = :forward)
            # copy to zvec and uvec
            x[diff_ptr:diff_ptr+diff_size-1] .= xinit[1:diff_size]
            y[alg_ptr:alg_ptr+alg_size-1] .= xinit[diff_size+1:diff_size+alg_size]
            u[ctrl_ptr:ctrl_ptr+ctrl_size-1] .= xinit[diff_size+alg_size+1:diff_size+alg_size+ctrl_size]
        end
    end
end


function fill_pvec!(pvec::AbstractArray, dtype::AbstractDeviceType)
    @warn "fill_pvec! not implemented for device type $(dtype)"
end

function initial_guess!(xinit::AbstractArray, p::AbstractArray, pg::Float64, qg::Float64, vm::Float64, va::Float64, dtype::AbstractDeviceType)
    @warn "initial_guess! not implemented for device type $(dtype)"
end

function initialize_dynamics!(f::AbstractArray, xinit::AbstractArray, p::AbstractArray, pg::Float64, qg::Float64, vm::Float64, va::Float64, dtype::AbstractDeviceType)
    @warn "initialize_dynamics! not implemented for device type $(dtype)"
end

export Genrou
export from_data_field
