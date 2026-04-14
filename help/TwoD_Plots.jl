include("../help/MeanFlow_PostProc.jl")

"""
    sdf_slice(sim::Simulation; axis::Int, idx::Int, ε::Float32)

Returns (xvals, yvals) representing the locations on the selected slice
where the SDF crosses zero (i.e., body surface).
- axis = 1 → y-z plane at x = idx
- axis = 2 → x-z plane at y = idx
- axis = 3 → x-y plane at z = idx
- ε: Tolerance
"""
function sdf_slice(sim::Union{Simulation,BiotSimulation}, axis::Int, idx::Int, ε::Float32)
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    σ = sim.flow.σ |> Array
    measure_sdf!(σ, sim.body, sim_time(sim))
    sdf = σ |> Array
    Nx, Ny, Nz = size(sdf)
    coords = []

    # Loop over slice according to axis
    if axis == 1 # y-z plane at x=idx
        for j in 2:Ny-1, k in 2:Nz-1
            if abs(sdf[idx, j, k]) < ε
                push!(coords, (j, k)) # y, z
            end
        end
    elseif axis == 2 # x-z plane at y=idx
        for i in 2:Nx-1, k in 2:Nz-1
            if abs(sdf[i, idx, k]) < ε
                push!(coords, (i, k)) # x, z
            end
        end
    elseif axis == 3 # x-y plane at z=idx
        for i in 2:Nx-1, j in 2:Ny-1
            if abs(sdf[i, j, idx]) < ε
                push!(coords, (i, j)) # x, y
            end
        end
    else
        error("Invalid axis. Use 1 for (y-z) plane, 2 for (x-z) plane, or 3 for (x-y) plane")
    end

    x = [p[1] for p in coords]
    y = [p[2] for p in coords]
    return x, y
end

function centered_bluesreds(data, center_value)
    dmin, dmax = extrema(data)
    if dmin == dmax
        return cgrad(:vik100)
    end
    pos = clamp((center_value - dmin) / (dmax - dmin), 0.0, 1.0)
    base_colors = get_color_palette(:vik100, 100)
    return cgrad(base_colors, [0.0, pos, 1.0], scale=:linear)
end

function colorbar_ticks_with_center(clims::Tuple, center_value, nticks)
    ticks = collect(LinRange(clims[1], clims[2], nticks))
    if all(abs(t - center_value) > 1e-6 for t in ticks)
        push!(ticks, center_value)
    end
    sort!(ticks)
    labels = [abs(t - center_value) < 1e-6 ? @sprintf("%.2f (center)", t) : @sprintf("%.2f", t) for t in ticks]
    return ticks, labels
end

name_rotor(prefix, D, λ₁, λ₂, θ) = string(prefix, "_", D, "_", λ₁, "_", λ₂, "_", θ)

function list_psolver_logs(D, λ₁, λ₂, θ; dir=".", prefix="psolver_rotor_NonBiotFaces", order=:mtime, rev=false)
    base  = name_rotor(prefix, D, λ₁, λ₂, θ)  # "psolver_rotor_NonBiotFaces_16_3_3"
    files = filter(f -> startswith(f, base * "_") && endswith(f, ".log"), readdir(dir))
    paths = joinpath.(dir, files)

    if order == :mtime
        return sort(paths; by = p -> stat(p).mtime, rev=rev)

    elseif order == :name
        return sort(paths; by = basename, rev=rev)

    elseif order == :tagtime
        re  = r"_([0-9]{8}-[0-9]{6})\.log$"
        fmt = DateFormat("yyyymmdd-HHMMSS")
        keyfun = p -> begin
            m = match(re, basename(p))
            isnothing(m) ? DateTime(0) : DateTime(m.captures[1], fmt)
        end
        return sort(paths; by = keyfun, rev=rev)

    elseif order == :tagint
        # e.g., "..._3.log", "..._10.log" → 3, 10 (numeric, not lexicographic)
        re = r"_(\d+)\.log$"
        keyfun = p -> begin
            m = match(re, basename(p))
            isnothing(m) ? typemax(Int) : parse(Int, m.captures[1])
        end
        return sort(paths; by = keyfun, rev=rev)

    else
        error("Unknown order = $(order). Use :mtime, :name, :tagtime, or :tagint.")
    end
end

function merge_psolver_logs(outname::AbstractString, files::Vector{<:AbstractString})
    outfile = endswith(outname, ".log") ? outname : outname * ".log"
    open(outfile, "w") do outio
        # Header must match what WaterLily.logger writes.
        println(outio, "p/c, iter, r∞, r₂")
        for f in files
            infile = endswith(f, ".log") ? f : f * ".log"
            open(infile, "r") do io
                _ = readline(io)  # skip header
                for line in eachline(io)
                    startswith(line, "p/c") && continue
                    println(outio, line)
                end
            end
        end
    end
    return outfile
end

function plot_logger_merged(outname, files)
    outfile = merge_psolver_logs(outname, files)
    plot_logger(replace(outfile, ".log" => ""))
end

function plot_logger_fix(fname="WaterLily.log")
    predictor = []; corrector = []
    open(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log","r") do f
        readline(f) # header
        which = "p"
        while !eof(f)
            s = split(readline(f), ",")
            which = s[1] != "" ? s[1] : which
            push!(which == "p" ? predictor : corrector, parse.(Float64, s[2:end]))
        end
    end
    isempty(predictor) && error("No predictor entries parsed from log.")
    isempty(corrector) && error("No corrector entries parsed from log.")

    predictor = reduce(hcat, predictor)
    corrector = reduce(hcat, corrector)

    idxp = findall(==(0.0), @views predictor[1, :])
    idxc = findall(==(0.0), @views corrector[1, :])
    length(idxp) ≥ 1 || error("No predictor step markers (nᵖ==0) found.")
    length(idxc) ≥ 1 || error("No corrector step markers (nᵇ==0) found.")

 # Common LaTeX strings
    xlab_steps   = L"\mathrm{Time\ step}"
    ylab_Linf    = L"\mathrm{L}_{\infty}-\mathrm{norm}"
    ylab_L2      = L"\mathrm{L}_{2}-\mathrm{norm}"
    ylab_iters   = L"\mathrm{Iterations}"
    ttl_res      = L"\mathrm{Residuals}"
    ttl_mg       = L"\mathrm{MG\ Iterations}"
    ttl_biot     = L"\mathrm{Biot-Savart}"

    # Predictor: initial residuals
    p1 = plot(1:length(idxp), predictor[2, idxp],
              color=:1, ls=:dash, alpha=0.8,
              label=L"\mathrm{predictor\ initial}\ r_{\infty}",
              yaxis=:log, size=(800,400), dpi=600,
              xlabel=xlab_steps, ylabel=ylab_Linf, title=ttl_res,
              tickfontsize=6,labelfontsize=10,legendfontsize=8,titlefontsize=14,legend=:bottom,
              ylims=(1e-8,1e0), xlims=(0,length(idxp)))

    p2 = plot(1:length(idxp), predictor[3, idxp],
              color=:1, ls=:dash, alpha=0.8,
              label=L"\mathrm{predictor\ initial}\ r_{2}",
              yaxis=:log, size=(800,400), dpi=600,
              xlabel=xlab_steps, ylabel=ylab_L2, title=ttl_res,
              tickfontsize=6,labelfontsize=10,legendfontsize=8,titlefontsize=14,legend=:bottom,
              ylims=(1e-8,1e0), xlims=(0,length(idxp)))

    # Predictor: final residuals
    if length(idxp) > 1
        cols_p = vcat(predictor[2, idxp[2:end] .- 1], predictor[2, end])
        plot!(p1, 1:length(idxp), cols_p, color=:1, lw=2, alpha=0.8, label=L"\mathrm{predictor}\ r_{\infty}")
        cols_p = vcat(predictor[3, idxp[2:end] .- 1], predictor[3, end])
        plot!(p2, 1:length(idxp), cols_p, color=:1, lw=2, alpha=0.8, label=L"\mathrm{predictor}\ r_{2}")
    else
        plot!(p1, 1, predictor[2,end], seriestype=:scatter, color=:1, label=L"\mathrm{predictor}\ r_{\infty}")
        plot!(p2, 1, predictor[3,end], seriestype=:scatter, color=:1, label=L"\mathrm{predictor}\ r_{2}")
    end

    # Predictor: MG iterations
    pred_iters = clamp.(
        length(idxp) > 1 ? vcat(predictor[1, idxp[2:end] .- 1], predictor[1, end])
                         : [predictor[1,end]],
        √1/2, 32
    )
    p3 = plot(1:length(idxp), pred_iters,
              lw=2, alpha=0.8, label=L"\mathrm{predictor}",
              size=(800,400), dpi=600,
              xlabel=xlab_steps, ylabel=ylab_iters, title=ttl_mg,
              tickfontsize=6,labelfontsize=10,legendfontsize=8,titlefontsize=14,legend=:top,
              ylims=(√1/2,32), xlims=(0,length(idxp)), yaxis=:log2)
    yticks!([√1/2,1,2,4,8,16,32], [L"0",L"1",L"2",L"4",L"8",L"16",L"32"])

    # Corrector: initial residuals
    plot!(p1, 1:length(idxc), corrector[2, idxc],
          color=:2, ls=:dash, alpha=0.8,
          label=L"\mathrm{corrector\ initial}\ r_{\infty}", yaxis=:log)
    plot!(p2, 1:length(idxc), corrector[3, idxc],
          color=:2, ls=:dash, alpha=0.8,
          label=L"\mathrm{corrector\ initial}\ r_{2}", yaxis=:log)

    # Corrector: final residuals & iterations
    if length(idxc) > 1
        cols_c = vcat(corrector[2, idxc[2:end] .- 1], corrector[2, end])
        plot!(p1, 1:length(idxc), cols_c, color=:2, lw=2, alpha=0.8, label=L"\mathrm{corrector}\ r_{\infty}")
        cols_c = vcat(corrector[3, idxc[2:end] .- 1], corrector[3, end])
        plot!(p2, 1:length(idxc), cols_c, color=:2, lw=2, alpha=0.8, label=L"\mathrm{corrector}\ r_{2}")
        cols_c = clamp.(vcat(corrector[1, idxc[2:end] .- 1], corrector[1, end]), √1/2, 32)
        plot!(p3, 1:length(idxc), cols_c, lw=2, alpha=0.8, label=L"\mathrm{corrector}")
    else
        plot!(p3, 1, clamp(corrector[1,end], √1/2, 32), seriestype=:scatter, label=L"\mathrm{corrector}")
    end

    # Optional Biot–Savart coupling iterations (4th row)
    if size(predictor, 1) > 3 && size(corrector, 1) > 3
        p4 = plot(1:length(idxp),
                    clamp.(length(idxp) > 1 ? vcat(predictor[4, idxp[2:end] .- 1], predictor[4, end])
                                            : [predictor[4,end]], √1/2, 32),
                    lw=2, alpha=0.8, label=L"\mathrm{predictor}",
                    size=(800,400), dpi=600,
                    xlabel=xlab_steps, ylabel=ylab_iters, title=ttl_biot,
                    tickfontsize=6,labelfontsize=10,legendfontsize=8,titlefontsize=14,legend=:top,
                    ylims=(√1/2,32), xlims=(0,length(idxp)), yaxis=:log2)
        yticks!([√1/2,1,2,4,8,16,32], [L"0",L"1",L"2",L"4",L"8",L"16",L"32"])
        cols_c4 = clamp.(length(idxc) > 1 ? vcat(corrector[4, idxc[2:end] .- 1], corrector[4, end])
                                            : [corrector[4,end]], √1/2, 32)
        plot!(p4, 1:length(idxc), cols_c4, lw=2, alpha=0.8, label=L"\mathrm{corrector}")
        plot(p1, p2, p3, p4, layout=@layout [a b c d])
    else
        plot(p1, p2, p3, layout=@layout [a b c])
    end
end

"""
Draw total + x- and y- component arrows with small heads, and annotate tip with value.
- (x0,y0): origin
- Fx, Fy:  mean forces (already non-dimensional)
- name:    "1" or "2" for legend labels
- scale:   uniform scale for visual length
- head:    :simple or :closed
- headsz:  arrowsize (try 0.3–0.6)
"""
function draw_force_vectors!(x0, y0, Fx, Fy; name="1", scale=1.0, color=:green,
                             arrow=:simple, arrowsize=0.5, lw=2,
                             components=true, annotate_tip=true)

    # Main vector
    Plots.quiver!([x0], [y0];
        quiver=([scale*Fx], [scale*Fy]),
        seriestype=:quiver,
        arrow=arrow, arrowsize=arrowsize,
        color=color, linewidth=lw,
        label = L"\vec{F}_$name"
    )

    plot_Fy = false
    if components
        # x-component always drawn
        x1, y1 = x0 + scale*Fx, y0
        Plots.plot!([x0, x1], [y0, y1];
              lw=lw, color=color, alpha=0.5, linestyle=:dash, label=false)

        # y-component drawn only if significant
        if abs(Fy) ≥ 0.01
            x2, y2 = x0, y0 + scale*Fy
            Plots.plot!([x0, x2], [y0, y2];
                  lw=lw, color=color, alpha=0.5, linestyle=:dash, label=false)
            plot_Fy = true
        end
    end

    if annotate_tip
        
        Fx_lbl, Fy_lbl = round(Fx, digits=3), round(Fy, digits=3)
        if plot_Fy
            tx, ty = x0 + scale*Fx, y0 + scale*Fy
            offset = 20
            Plots.annotate!(tx + offset, ty + offset, Plots.text(latexstring("\\overline{C}=(\\overline{C}_{D}=$(Fx_lbl),\\overline{C}_{L}=$(Fy_lbl))"), 8, color=color))
        else
            tx, ty = x0, y0
            offset = 80
            Plots.annotate!(tx, ty + offset, Plots.text(latexstring("{\\overline{C}=(\\overline{C}_{D}=$(Fx_lbl))}"), 10, color=color))
        end
    end
end
