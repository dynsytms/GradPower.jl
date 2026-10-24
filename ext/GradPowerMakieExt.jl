module GradPowerMakieExt

using GradPower
using Makie

function _outside_legend!(fig, ax, row)
    legend = Makie.axislegend(ax; bbox=nothing, framevisible=false,
                             tellwidth=true, tellheight=false)
    fig[row, 2] = legend
    return legend
end

function GradPower.plot_states(r::GradPower.SimulationResult, selectors;
                               scenarios=nothing, figure_kwargs=(;), axis_kwargs=(;))
    times = GradPower._plot_times(r)
    records = GradPower._plot_states_data(r, selectors; scenarios)
    fig = Makie.Figure(; figure_kwargs...)
    colors = Makie.wong_colors()
    for (row, record) in enumerate(records)
        kwargs = merge((title=record.label, xlabel="Time (s)", ylabel=record.ylabel),
                       axis_kwargs)
        ax = Makie.Axis(fig[row, 1]; kwargs...)
        for (i, series) in enumerate(record.series)
            color = colors[mod1(i, length(colors))]
            Makie.lines!(ax, times, series.values; color,
                         label="Scenario $(repr(series.scenario))")
            if length(times) == 1
                Makie.scatter!(ax, times, series.values; color, markersize=8)
            end
        end
        length(record.series) > 1 && _outside_legend!(fig, ax, row)
    end
    return fig
end

function _plot_curves(times, data; figure_kwargs, axis_kwargs)
    fig = Makie.Figure(; figure_kwargs...)
    kwargs = merge((xlabel="Time (s)", ylabel=data.ylabel), axis_kwargs)
    ax = Makie.Axis(fig[1, 1]; kwargs...)
    colors = Makie.wong_colors()
    for (i, curve) in enumerate(data.curves)
        color = colors[mod1(i, length(colors))]
        if curve.lower !== nothing && curve.upper !== nothing
            Makie.band!(ax, times, curve.lower, curve.upper; color=(color, 0.2))
        end
        Makie.lines!(ax, times, curve.values; color, label=curve.label)
        if length(times) == 1
            Makie.scatter!(ax, times, curve.values; color, markersize=8)
        end
    end
    isempty(data.curves) || _outside_legend!(fig, ax, 1)
    return fig
end

function GradPower.plot_voltage(r::GradPower.SimulationResult;
                                buses=nothing, scenarios=nothing, envelope=false,
                                component=:magnitude, figure_kwargs=(;), axis_kwargs=(;))
    times = GradPower._plot_times(r)
    data = GradPower._plot_voltage_data(r; buses, scenarios, envelope, component)
    return _plot_curves(times, data; figure_kwargs, axis_kwargs)
end

function GradPower.plot_speed(r::GradPower.SimulationResult;
                              generators=nothing, scenarios=nothing, envelope=false,
                              unit=:pu, nominal_frequency=60.0, model=:genrou,
                              figure_kwargs=(;), axis_kwargs=(;))
    times = GradPower._plot_times(r)
    data = GradPower._plot_speed_data(r; generators, scenarios, envelope,
                                    unit, nominal_frequency, model)
    return _plot_curves(times, data; figure_kwargs, axis_kwargs)
end

function _network_colorrange(lo, hi)
    isfinite(lo) && isfinite(hi) ||
        throw(ArgumentError("network color limits must be finite"))
    lo <= hi || throw(ArgumentError("network color limits must be increasing"))
    if lo == hi
        padding = max(abs(lo) * 0.01, 0.01)
        return (lo - padding, hi + padding)
    end
    return (lo, hi)
end

function _network_colorrange(colorrange)
    length(colorrange) == 2 ||
        throw(ArgumentError("network colorrange must contain two limits"))
    return _network_colorrange(Float64(first(colorrange)), Float64(last(colorrange)))
end

function _network_title(data, title)
    return data.time === nothing ? string(title) : "$title at t = $(data.time) s"
end

function _network_scene(data; colorrange, colormap, labels, figure_kwargs, axis_kwargs)
    fig = Makie.Figure(; figure_kwargs...)
    title = Makie.Observable(_network_title(data, get(axis_kwargs, :title, "Network")))
    kwargs = merge((aspect=Makie.DataAspect(),
                    xautolimitmargin=(0.12, 0.12), yautolimitmargin=(0.12, 0.12)),
                   axis_kwargs, (; title))
    ax = Makie.Axis(fig[1, 1]; kwargs...)
    points = Makie.Point2f.(data.positions)
    segments = Makie.Point2f[]
    for (a, b) in data.edges
        push!(segments, points[a], points[b])
    end
    Makie.linesegments!(ax, segments; color=:gray65, linewidth=1.5)
    colors = Makie.Observable(data.values)
    nodes = Makie.scatter!(ax, points; color=colors, colorrange, colormap,
                          markersize=15, strokecolor=:gray25, strokewidth=0.7)
    if labels
        Makie.text!(ax, points; text=string.(data.buses),
                    offset=(9, 9), align=(:left, :bottom), fontsize=12)
    end
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    Makie.Colorbar(fig[1, 2], nodes; label=data.label)
    return (; fig, colors, title)
end

function GradPower.plot_network(ps::GradPower.PowerSystem,
                                r::Union{Nothing,GradPower.SimulationResult}=nothing;
                                scenario=nothing, frame=nothing, positions=nothing,
                                quantity=:voltage, labels=true, colorrange=nothing,
                                colormap=:viridis, figure_kwargs=(;), axis_kwargs=(;))
    labels isa Bool || throw(ArgumentError("labels must be a Bool"))
    data = GradPower._plot_network_data(ps, r; scenario, frame, positions, quantity)
    limits = colorrange === nothing ?
             _network_colorrange(extrema(data.values)...) :
             _network_colorrange(colorrange)
    scene = _network_scene(data; colorrange=limits, colormap, labels,
                           figure_kwargs, axis_kwargs)
    return scene.fig
end

function GradPower.animate_network(path::AbstractString, ps::GradPower.PowerSystem,
                                   r::GradPower.SimulationResult;
                                   scenario=nothing, frames=nothing, framerate=20,
                                   positions=nothing, quantity=:voltage, labels=true,
                                   colorrange=nothing, colormap=:viridis,
                                   figure_kwargs=(;), axis_kwargs=(;))
    labels isa Bool || throw(ArgumentError("labels must be a Bool"))
    selected_frames = GradPower._plot_animation_frames(r, frames, framerate)
    initial = GradPower._plot_network_frame_data(
        ps, r; scenario, frame=first(selected_frames), positions, quantity)
    if colorrange === nothing
        # Retain only one frame's values and scalar extrema during the range scan.
        lo, hi = extrema(initial.values)
        for frame in Iterators.drop(selected_frames, 1)
            data = GradPower._plot_network_frame_data(ps, r; scenario, frame, positions, quantity)
            frame_lo, frame_hi = extrema(data.values)
            lo = min(lo, frame_lo)
            hi = max(hi, frame_hi)
        end
        limits = _network_colorrange(lo, hi)
    else
        limits = _network_colorrange(colorrange)
    end
    scene = _network_scene(initial; colorrange=limits, colormap, labels,
                           figure_kwargs, axis_kwargs)
    title = get(axis_kwargs, :title, "Network")
    Makie.record(scene.fig, path, selected_frames; framerate) do frame
        data = GradPower._plot_network_frame_data(ps, r; scenario, frame, positions, quantity)
        scene.colors[] = data.values
        scene.title[] = _network_title(data, title)
    end
    return String(path)
end

function _cpf_scene(r, indices; figure_kwargs, axis_kwargs)
    fig = Makie.Figure(;figure_kwargs...)
    kwargs = merge((xlabel="Loading factor λ",ylabel="Voltage magnitude (pu)",
                    title="Steady-state continuation — $(r.status)"),axis_kwargs)
    ax = Makie.Axis(fig[1,1];kwargs...)
    colors = Makie.wong_colors()
    return fig,ax,colors
end

function GradPower.plot_cpf(r::GradPower.ContinuationResult;
                            buses=nothing,figure_kwargs=(;),axis_kwargs=(;))
    indices = GradPower._plot_cpf_indices(r,buses)
    fig,ax,colors = _cpf_scene(r,indices;figure_kwargs,axis_kwargs)
    lambda = [p.lambda for p in r.points]
    for (k,i) in enumerate(indices)
        color = colors[mod1(k,length(colors))]
        Makie.lines!(ax,lambda,[p.vm[i] for p in r.points];color,linewidth=2,
                     label="Bus $(r.bus_numbers[i])")
        length(lambda)==1 && Makie.scatter!(ax,lambda,[r.points[1].vm[i]];color)
        for e in r.events
            e.bus == r.bus_numbers[i] || continue
            Makie.scatter!(ax,[e.point.lambda],[e.point.vm[i]];color,
                           marker=e.kind === :recovery ? :utriangle : :rect,markersize=12)
        end
        if r.maximum !== nothing
            Makie.scatter!(ax,[r.maximum.lambda],[r.maximum.vm[i]];color,
                           marker=:star5,markersize=18)
        end
    end
    _outside_legend!(fig,ax,1)
    return fig
end

function GradPower.animate_cpf(path::AbstractString,r::GradPower.ContinuationResult;
                               buses=nothing,framerate=12,figure_kwargs=(;),axis_kwargs=(;))
    framerate isa Integer && !(framerate isa Bool) && framerate>0 ||
        throw(ArgumentError("framerate must be a positive integer"))
    indices = GradPower._plot_cpf_indices(r,buses)
    fig,ax,colors = _cpf_scene(r,indices;figure_kwargs,axis_kwargs)
    lambda = [p.lambda for p in r.points]
    lo,hi = extrema(lambda)
    vlo,vhi = extrema(p.vm[i] for p in r.points for i in indices)
    xpad,ypad = max(.05*(hi-lo),.01),max(.05*(vhi-vlo),.01)
    Makie.xlims!(ax,lo-xpad,hi+xpad)
    Makie.ylims!(ax,vlo-ypad,vhi+ypad)
    frame = Makie.Observable(1)
    for (k,i) in enumerate(indices)
        color = colors[mod1(k,length(colors))]
        curve = Makie.lift(frame) do j
            Makie.Point2f[(r.points[t].lambda,r.points[t].vm[i]) for t in 1:j]
        end
        tip = Makie.lift(j -> [Makie.Point2f(r.points[j].lambda,r.points[j].vm[i])],frame)
        Makie.lines!(ax,curve;color,linewidth=2,label="Bus $(r.bus_numbers[i])")
        Makie.scatter!(ax,tip;color,markersize=12)
    end
    _outside_legend!(fig,ax,1)
    Makie.record(fig,path,eachindex(r.points);framerate) do j
        frame[] = j
    end
    return String(path)
end

end # module
