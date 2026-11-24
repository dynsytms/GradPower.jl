# Figures for the PERC2 case studies, read from artifacts/perc2/study<k>.json.
# All results use ASSUMED STUDY PARAMETERS (no facility validation).
#
#   julia +1.13 --project=test/plotting scripts/perc2/plot.jl
#
# Writes artifacts/perc2/study<k>.png (git-ignored).

using CairoMakie
include(joinpath(@__DIR__, "json_io.jl"))
using .PERC2JSON

CairoMakie.activate!()

const OUTDIR = joinpath(@__DIR__, "..", "..", "artifacts", "perc2")
const FOOT = "assumed study parameters - not facility values"

num(v) = Float64[x === nothing ? NaN : Float64(x) for x in v]
loadstudy(k) = (p = joinpath(OUTDIR, "study$k.json"); isfile(p) ? read_json(p) : nothing)
ok(r) = startswith(get(r, "status", ""), "ok")

function footer!(fig, doc)
    Label(fig[end+1, 1:end], FOOT; fontsize = 10,
          halign = :left, color = :gray40, tellwidth = false)
end

function skipped_fig(k, doc)
    fig = Figure(size = (700, 200))
    Label(fig[1, 1], "Study $k: $(doc["status"])"; fontsize = 16)
    return fig
end

function playback_fig(k, doc; title)
    runs = filter(ok, doc["runs"])
    fig = Figure(size = (950, 700))
    ax1 = Axis(fig[1, 1]; ylabel = "voltage (pu)", title = "Study $k: $title")
    ax2 = Axis(fig[2, 1]; ylabel = "P / P0")
    ax3 = Axis(fig[3, 1]; ylabel = "FracOn", xlabel = "time (s)")
    for (i, r) in enumerate(runs)
        tr = r["trace"]; t = num(tr["t"])
        i == 1 && lines!(ax1, t, num(tr["V"]); color = :black, linestyle = :dash, label = "V (input)")
        lines!(ax1, t, num(tr["Vtf"]); label = r["name"])
        lines!(ax2, t, num(tr["P_pu_of_P0"]); label = r["name"])
        lines!(ax3, t, num(tr["FracOn"]); label = r["name"])
    end
    axislegend(ax1; position = :rb, labelsize = 9); axislegend(ax2; position = :rb, labelsize = 9)
    linkxaxes!(ax1, ax2, ax3)
    footer!(fig, doc)
    return fig
end

function study5_fig(doc)
    fig = playback_fig(5, doc; title = "Loading factor Ldfact sweep")
    sw = get(doc, "ldfact_sweep", Any[])
    if !isempty(sw)
        ax = Axis(fig[1:3, 2]; xlabel = "Ldfact", ylabel = "ramp duration (s)", title = "ramp vs Tramp*Ldfact")
        ld = num([s["Ldfact"] for s in sw])
        scatter!(ax, ld, num([s["ramp_duration_s"] for s in sw]); label = "measured", markersize = 12)
        lines!(ax, ld, num([s["TrampUsed_s"] for s in sw]); color = :red, linestyle = :dash, label = "Tramp*Ldfact")
        axislegend(ax; position = :lt)
        colsize!(fig.layout, 2, Relative(0.3))
    end
    return fig
end

function study6_fig(doc)
    runs = filter(ok, doc["runs"])
    fig = Figure(size = (1000, 1000))
    ax = [Axis(fig[i, 1]; ylabel = l) for (i, l) in enumerate(["PERC2 bus voltage (pu)", "PERC2 P (MW)", "FracOn",
                                                              "max |gen speed dev| (Hz)"])]
    ax[4].xlabel = "time (s)"
    ax[1].title = "Study 6: IEEE 9, PERC2 = $(round(100 * doc["perc2_fraction"]; digits = 0))% of bus $(doc["host_bus"]), fault at bus $(doc["fault_bus"])"
    host = doc["host_bus"]
    for r in runs
        tr = r["trace"]; t = num(tr["t"])
        lines!(ax[1], t, num(tr["V_bus$host"]); label = r["name"])
        if haskey(tr, "perc2_bus$(host)_P_MW")
            lines!(ax[2], t, num(tr["perc2_bus$(host)_P_MW"]); label = r["name"])
            lines!(ax[3], t, num(tr["perc2_bus$(host)_FracOn"]); label = r["name"])
        end
        gk = [k for k in keys(tr) if startswith(k, "speed_dev_Hz_gen")]
        dev = maximum(reduce(hcat, [abs.(num(tr[k])) for k in gk]); dims = 2)
        lines!(ax[4], t, vec(dev); label = r["name"])
    end
    xlims!(ax[1], 0, 6); foreach(a -> linkxaxes!(ax[1], a), ax[2:4])
    Legend(fig[1:4, 2], ax[1]; labelsize = 9)
    footer!(fig, doc)
    return fig
end

function study7_fig(doc)
    runs = filter(ok, doc["runs"])
    fig = Figure(size = (1000, 900))
    a1 = Axis(fig[1, 1]; ylabel = "total PERC2 P (MW)",
              title = "Study 7: IEEE 9, PERC2 at buses $(doc["perc2_buses"]), fault at bus $(doc["fault_bus"])")
    a2 = Axis(fig[2, 1]; ylabel = "max |gen speed dev| (Hz)")
    a3 = Axis(fig[3, 1]; ylabel = "FracOn per bus (heterogeneous)", xlabel = "time (s)")
    for r in runs
        tr = r["trace"]; t = num(tr["t"])
        haskey(tr, "perc2_total_P_MW") && lines!(a1, t, num(tr["perc2_total_P_MW"]); label = r["name"])
        gk = [k for k in keys(tr) if startswith(k, "speed_dev_Hz_gen")]
        dev = maximum(reduce(hcat, [abs.(num(tr[k])) for k in gk]); dims = 2)
        lines!(a2, t, vec(dev); label = r["name"])
        if r["name"] == "heterogeneous"
            for k in sort(collect(keys(tr)))
                endswith(k, "_FracOn") && lines!(a3, t, num(tr[k]); label = replace(k, "perc2_" => "", "_FracOn" => ""))
            end
            axislegend(a3; position = :rb, labelsize = 9)
        end
    end
    xlims!(a1, 0, 12); linkxaxes!(a1, a2, a3)
    Legend(fig[1:2, 2], a1; labelsize = 9)
    footer!(fig, doc)
    return fig
end

function study8_fig(doc)
    sw = filter(r -> startswith(r["status"], "ok"), doc["sweep"])
    vlc = sort(unique(num([r["VLcease"] for r in sw]))); lds = sort(unique(num([r["Ldfact"] for r in sw])))
    fig = Figure(size = (1100, 800))
    for (j, (key, lab)) in enumerate((("trip_time_s", "trip time (s)"), ("reconnect_end_s", "reconnect end (s)"),
                                      ("min_V_host", "min V at PERC2 bus (pu)"), ("max_abs_speed_dev_Hz", "max |gen speed dev| (Hz)")))
        ax = Axis(fig[div(j - 1, 2) + 1, mod(j - 1, 2) + 1]; xlabel = "VLcease (pu)", ylabel = "Ldfact", title = lab,
                  xticks = (1:length(vlc), string.(vlc)), yticks = (1:length(lds), string.(lds)))
        Z = fill(NaN, length(vlc), length(lds))
        for r in sw
            Z[findfirst(==(r["VLcease"]), vlc), findfirst(==(r["Ldfact"]), lds)] = something(r["metrics"][key], NaN)
        end
        hm = heatmap!(ax, 1:length(vlc), 1:length(lds), Z); Colorbar(fig[div(j - 1, 2) + 1, mod(j - 1, 2) + 3], hm)
        for i in eachindex(vlc), k in eachindex(lds)
            text!(ax, i, k; text = isnan(Z[i, k]) ? "-" : string(round(Z[i, k]; sigdigits = 3)), align = (:center, :center), fontsize = 11)
        end
    end
    b = get(doc, "batched_check", Dict())
    if get(b, "status", "") == "ok"
        Label(fig[3, 1:4], "batched check (M=$(b["M"]), H scales $(b["H_scales"])): max |batched - single| speed = " *
              join(["$(c["max_abs_diff_speed_pu"])" for c in b["compare_with_single"]], ", "); fontsize = 11, tellwidth = false)
    end
    footer!(fig, doc)
    return fig
end

function study9_fig(doc)
    runs = filter(ok, doc["runs"])
    fig = Figure(size = (1000, 900))
    a1 = Axis(fig[1, 1]; ylabel = "estimated bus frequency (Hz)", title = "Study 9: generator trip at bus $(doc["generator_trip"]["bus"]), PERC2 at bus $(doc["perc2_bus"])")
    a2 = Axis(fig[2, 1]; ylabel = "PERC2 P (MW)")
    a3 = Axis(fig[3, 1]; ylabel = "FracOn", xlabel = "time (s)")
    a4 = Axis(fig[4, 1]; ylabel = "mean gen speed (Hz dev)", xlabel = "time (s)")
    b = doc["perc2_bus"]
    pal = Makie.wong_colors()
    col = Dict(r["name"] => pal[i] for (i, r) in enumerate(doc["runs"]))
    for r in runs
        tr = r["trace"]; t = num(tr["t"])
        if haskey(tr, "perc2_bus$(b)_f_Hz")
            lines!(a1, t, num(tr["perc2_bus$(b)_f_Hz"]); label = r["name"], color = col[r["name"]])
            lines!(a2, t, num(tr["perc2_bus$(b)_P_MW"]); label = r["name"], color = col[r["name"]])
            lines!(a3, t, num(tr["perc2_bus$(b)_FracOn"]); label = r["name"], color = col[r["name"]])
        end
        gk = [k for k in keys(tr) if startswith(k, "speed_dev_Hz_gen")]
        lines!(a4, t, vec(sum(reduce(hcat, [num(tr[k]) for k in gk]); dims = 2) ./ length(gk)); label = r["name"], color = col[r["name"]])
    end
    nd = doc["estimated_frequency_nadir_Hz"]
    nd === nothing || hlines!(a1, [nd]; color = :gray, linestyle = :dot)
    axislegend(a1; position = :rb, labelsize = 9); axislegend(a2; position = :rb, labelsize = 9)
    linkxaxes!(a1, a2, a3, a4)
    footer!(fig, doc)
    return fig
end

function main()
    made = String[]
    for (k, f) in ((1, d -> playback_fig(1, d; title = "Short dip, fast reconnect")),
                   (2, d -> playback_fig(2, d; title = "Long dip, slow reconnect")),
                   (3, d -> playback_fig(3, d; title = "Swell")),
                   (4, d -> playback_fig(4, d; title = "Current limiting")),
                   (5, study5_fig), (6, study6_fig), (7, study7_fig), (8, study8_fig), (9, study9_fig))
        doc = loadstudy(k)
        if doc === nothing
            println("study $k: no JSON, skipped"); continue
        end
        has = k == 8 ? !isempty(get(doc, "sweep", [])) : !isempty(filter(ok, doc["runs"]))
        fig = has ? f(doc) : skipped_fig(k, doc)
        path = joinpath(OUTDIR, "study$k.png")
        save(path, fig)
        push!(made, path)
        println("study $k: ", relpath(path, pwd()))
    end
    return made
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
