include("../help/MeanFlow_PostProc.jl")
include("../help/Forces_PostProc.jl")
include("../help/ThreeD_Plots.jl")
include("../help/TwoD_Plots.jl")
include("../setup/3D.jl")

Ds = [16] # diameter resolution
λs₁, λs₂ = [3], [3] # spin ratio 
backend = CuArray # Array for CPU
L = (8,5,8) # Domain dimensions (in D)
Re = 1000
T = Float32
U = 1 # Inflow velocity in x-direction
θ = 0 # Angle of incidence
U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
stats = true # false: do not calculate statustics, true: calculate statustics
stats_turb = true
time_max = 20 # Time to end sim
stats_init = 10 # Time to start calculating statistics
stats_trim = 10 # Time to trim the starting few timesteps for statistics
σ_init = 5 # Timestep to start calculatiing standart deviation
stats_interval = 0.1 # in CTU
dump_interval = 5 # in CTU
u_probe_loc=(3,2.5,3.5) # in D
u_probe_component = 1
z_vals = T.([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])
datadir = string(@__DIR__) * "../../sims/data/rotor_BiotSimulation/"
psolverdir = string(@__DIR__) * "../../"
pdf_file = "../sims/tex/img/rotor_BiotSimulation_validation.pdf"
fname_output = "meanflow"
verbose = true
run = true # false: postproc, true: run cases
_continue = false # false: start a new sim, true: continue from the last save (after restarting julia)
_plot = true
plot_log = false
clims_p = (-3,3) # Pressure plot limits
clims_ux = (-1,3) # Velocity plot limits
pstep = 0.25
ustep = 0.25
clims_c = (-5,5)
clength = 20
vectors = true # false: do not plot force vectors, true: plot force vectors
fscale = 15 # Force vector scale

function main(U=U)
    run && mkpath(datadir)
    for D in Ds
        for λ₁ in λs₁
            for λ₂ in λs₂
                x_vals = T.(2:1/D:4)
                println("Running D, λ₁, λ₂, θ = $(D), $(λ₁), $(λ₂), $(θ)")
                if run
                    sim, meanflow, p_force₁, p_force₂, v_force₁, v_force₂ = run_sim_rotor_NonBiotFaces(D, λ₁, λ₂, θ, U₊, backend;dir=datadir, L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=_continue, log_tag=2)
                end
                
                if plot_log
                    base    = name_rotor("psolver_rotor_NonBiotFaces", D, λ₁, λ₂, θ)             # e.g. "psolver_rotor_NonBiotFaces_16_3_3"
                    outbase = joinpath(psolverdir, base)                                       # final merged file path (no .log)
                    inputs  = list_psolver_logs(D, λ₁, λ₂, θ; dir=psolverdir, prefix="psolver_rotor_NonBiotFaces", order=:tagint)

                    in_bases = replace.(inputs, ".log" => "")
                    combined = merge_psolver_logs(outbase, in_bases)                           # writes "<outbase>.log"

                    plot_logger_fix(outbase) 
                    savefig(outbase)
                end

                p_force₁, p_force₂, v_force₁, v_force₂, u_probe_x_full, u_probe_y_full, u_probe_z_full, t_full = read_force_and_probe_vals("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; dir=datadir)
                

                forces = get_forces_and_coefficients(p_force₁, v_force₁, p_force₂, v_force₂, D)
                meanforces = get_mean_forces_and_coefficients(forces, t_full, stats_init)

                U_planes =  zeros(Float32, length(z_vals), length(x_vals))
                W_planes =  zeros(Float32, length(z_vals), length(x_vals))

                U_probe_x_mean = moving_average(stats_interval, u_probe_x_full, t_full)
                σ_U_probe_x =  moving_σ(u_probe_x_full,t_full,0)

                U_probe_y_mean = moving_average(stats_interval, u_probe_y_full, t_full)
                σ_U_probe_y =  moving_σ(u_probe_y_full,t_full,0)

                U_probe_z_mean = moving_average(stats_interval, u_probe_z_full, t_full)
                σ_U_probe_z =  moving_σ(u_probe_z_full,t_full,0)
                
                σ_U_probe_x_stats = moving_σ(u_probe_x_full,t_full,stats_init)
                σ_U_probe_y_stats = moving_σ(u_probe_y_full,t_full,stats_init)
                σ_U_probe_z_stats = moving_σ(u_probe_z_full,t_full,stats_init)
                idx = t_full .> stats_init
                u_probe_x_stats, u_probe_y_stats, u_probe_z_stats, t_stats = u_probe_x_full[idx], u_probe_y_full[idx], u_probe_z_full[idx], t_full[idx]
                U_probe_x_stats = moving_average(stats_interval, u_probe_x_stats, t_stats)
                U_probe_y_stats = moving_average(stats_interval, u_probe_y_stats, t_stats)
                U_probe_z_stats = moving_average(stats_interval, u_probe_z_stats, t_stats)

                σ_U_probe_x_trim = moving_σ(u_probe_x_full,t_full,stats_trim)
                σ_U_probe_y_trim = moving_σ(u_probe_y_full,t_full,stats_trim)
                σ_U_probe_z_trim = moving_σ(u_probe_z_full,t_full,stats_trim)
                idx_trim = t_full .> stats_trim
                u_probe_x_trim, u_probe_y_trim, u_probe_z_trim, t_trim = u_probe_x_full[idx_trim], u_probe_y_full[idx_trim], u_probe_z_full[idx_trim], t_full[idx_trim]
                U_probe_x_trim = moving_average(stats_interval, u_probe_x_trim, t_trim)
                U_probe_y_trim = moving_average(stats_interval, u_probe_y_trim, t_trim)
                U_probe_z_trim = moving_average(stats_interval, u_probe_z_trim, t_trim)

                if stats && stats_turb
                    P, U, UU, τ, t_meanflow = read_meanflow("meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; dir=datadir, stats, stats_turb)
                    Ux, Uy, Uz = U[:,:,:,1], U[:,:,:,2], U[:,:,:,3]
                else stats
                    P, U, t_meanflow = read_meanflow("meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; dir=datadir, stats, stats_turb)
                    Ux, Uy, Uz = U[:,:,:,1], U[:,:,:,2], U[:,:,:,3]
                end
                for (i, z) in enumerate(z_vals)
                    for (j, x) in enumerate(x_vals)
                        sample_loc_n = @. (x*D,(2.5*D)+1,z*D) |> ceil |> Int
                        val_U = view(Ux, sample_loc_n...) |> Array |> x -> x[]
                        val_W = view(Uz, sample_loc_n...) |> Array |> x -> x[]
                        U_planes[i,j] = val_U
                        W_planes[i,j] = val_W
                    end
                end

                p, u, Δt = read_flow("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2";dir=datadir)

                out_file = joinpath(datadir, "UW_planes_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2")
                jldsave(out_file; 
                    U_planes = U_planes,
                    W_planes = W_planes,
                    z_vals = z_vals,
                    x_vals = x_vals
                )
                
                curl = zeros(Float32, size(U,1), size(U,2), size(U,3), size(U,4))
                for i in 1:3
                    tmp = similar(U[:,:,:,1])  # scratch array for one component
                    @inside tmp[I] = WaterLily.curl(i, I, U)*D
                    curl[:,:,:,i] .= tmp
                end
                out_file = joinpath(datadir, "curl_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2")
                jldsave(out_file; curl = curl)
                curl_x, curl_y, curl_z = curl[:,:,:,1], curl[:,:,:,2], curl[:,:,:,3]

                if _plot 
                    plt_Ux = Plots.plot()
                    plt_Uz = Plots.plot()
                    for i in 1:size(U_planes, 1)
                        Plots.plot!(plt_Ux, x_vals, U_planes[i, :], label=@sprintf("z/D = %.2f", z_vals[i]), xlabel =L"x/D",
                        ylabel =L"u/U", framestyle=:box, grid=true, size(600,600), xlims=(x_vals[1], x_vals[end]),
                        tickfontsize=12, labelfontsize=12, legendfontsize=10, legend = :topright, left_margin=Plots.Measures.Length(:mm, 5), lw=2)
                        savefig(plt_Ux,string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_Ux.pdf")

                        Plots.plot!(plt_Uz,x_vals, W_planes[i, :], label = @sprintf("z/D = %.2f", z_vals[i]), xlabel =L"x/D",
                        ylabel =L"w/U", framestyle=:box, grid=true, size(600,600), xlims=(x_vals[1], x_vals[end]),
                        tickfontsize=12, labelfontsize=12, legendfontsize=10, legend = :topright, left_margin=Plots.Measures.Length(:mm, 5), lw=2)
                        savefig(plt_Uz,string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_Uz.pdf")
                    end
                    
                    P_planes = zeros(Float32, length(z_vals), size(P,1))
                    for (i, z) in enumerate(z_vals)
                        y0 = Int(2.5D+1)
                        z0 = clamp(round(Int, z*D) + 1, 1, size(P,3))
                        P_planes[i, :] .= P[:, y0, z0]
                    end
                    nx = size(P,1)
                    plt_P = Plots.plot()
                    x_vals = 0:1:nx-2
                    x_valsD = x_vals./D
                    for i in 1:size(P_planes, 1)
                        Plots.plot!(plt_P, x_valsD[2:end], P_planes[i, 2:end-1], label=@sprintf("P @ z/D = %.2f", z_vals[i]), xlabel =L"x/D",
                        ylabel =L"Pressure", framestyle=:box, grid=true, size(600,600), xlims=(x_valsD[1], x_valsD[end]),
                        tickfontsize=12, labelfontsize=12, legendfontsize=5, legend = :bottomleft, left_margin=Plots.Measures.Length(:mm, 5), lw=2)
                    end
                    savefig(plt_P,string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_P.pdf")

                    p_planes = zeros(Float32, length(z_vals), size(p,1))
                    for (i, z) in enumerate(z_vals)
                        y0 = Int(2.5D+1)
                        z0 = clamp(round(Int, z*D) + 1, 1, size(p,3))
                        p_planes[i, :] .= p[:, y0, z0]
                    end
                    nx = size(p,1)
                    plt_p = Plots.plot()
                    x_vals = 0:1:nx-2
                    x_valsD = x_vals./D
                    for i in 1:size(p_planes, 1)
                        Plots.plot!(plt_p, x_valsD[2:end], p_planes[i, 2:end-1], label=@sprintf("P @ z/D = %.2f", z_vals[i]), xlabel =L"x/D",
                        ylabel =L"Pressure", framestyle=:box, grid=true, size(600,600), xlims=(x_valsD[1], x_valsD[end]),
                        tickfontsize=12, labelfontsize=12, legendfontsize=5, legend = :bottomleft, left_margin=Plots.Measures.Length(:mm, 5), lw=2)
                    end
                    savefig(plt_p,string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_P_instantaneous.pdf")

                    # Rotor centers from 3D.jl definition
                    o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
                    o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
                    r  = (D-4)/2    # cylinder radius
                    for z in z_vals.*D
                        nx = size(Ux, 1)
                        ny = size(Ux, 2)
                        xtick_vals = 1:D:nx
                        ytick_vals = 1:D:ny
                        xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                        ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                        xticks = (xtick_vals, xtick_labels)
                        yticks = (ytick_vals, ytick_labels)
                        Ux_clims = clims_ux 
                        plt_Ux_flood = Plots.heatmap(
                            Ux[:,:,Int(z)]';
                            xlabel = L"x/D", ylabel = L"y/D",
                            xticks = xticks, yticks = yticks,
                            xlims = (-5,nx+5), ylims = (-5,ny+5),
                            color = :vik50,
                            clims = (Ux_clims),
                            colorbar = true, colorbar_title = L"\overline{u}/U_{\infty}",
                            levels = 50,                     
                            size = (600, 350),
                            aspect_ratio = :equal,  
                            tickfontsize = 10, labelfontsize = 10,
                            legendfontsize = 10, legend = :topright,
                            left_margin = Plots.Measures.Length(:mm, 5),
                        )
                        Plots.contour!(permutedims(Ux[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=ustep), color=:black, linewidth=1)
                        ϕ = range(0, 2π; length=100)
                        Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        if vectors
                            draw_force_vectors!(o₁[1], o₁[2], meanforces.C̄x₁, meanforces.C̄y₁; name="1", scale=fscale, arrow=:simple, arrowsize=0.1, lw=1, components=true, annotate_tip=true)
                            draw_force_vectors!(o₂[1], o₂[2], meanforces.C̄x₂, meanforces.C̄y₂; name="2", scale=fscale, arrow=:simple, arrowsize=0.1, lw=1, components=true, annotate_tip=true)
                        end
                        savefig(plt_Ux_flood, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_Ux_flood_at_$(z/D).pdf")
                    end

                    for z in z_vals.*D
                        nx = size(curl_z, 1)
                        ny = size(curl_z, 2)
                        xtick_vals = 1:D:nx
                        ytick_vals = 1:D:ny
                        xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                        ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                        xticks = (xtick_vals, xtick_labels)
                        yticks = (ytick_vals, ytick_labels)
                        Ux_clims = clims_c
                        plt_curl_z = Plots.heatmap(
                            curl_z[:,:,Int(z)]';
                            xlabel = L"x/D", ylabel = L"y/D",
                            xticks = xticks, yticks = yticks,
                            xlims = (-5,nx+5), ylims = (-5,ny+5),
                            color = :vik50,
                            clims = (Ux_clims),
                            colorbar = true, colorbar_title = L"\overline{\omega}_{z}",
                            colorbar_tickfont = font(10),
                            levels = 50,                     
                            size = (600, 350),
                            aspect_ratio = :equal,  
                            tickfontsize = 10, labelfontsize = 10,
                            legendfontsize = 10, legend = :topright,
                            left_margin = Plots.Measures.Length(:mm, 5),
                        )
                        Plots.contour!(permutedims(curl_z[:,:,Int(z)]), levels=range(clims_c[1],clims_c[2],length=clength), color=:black, linewidth=1)
                        ϕ = range(0, 2π; length=100)
                        Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        savefig(plt_curl_z, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_curl_z_at_$(z/D).pdf")
                    end

                    for z in z_vals.*D
                        nx = size(P, 1)
                        ny = size(P, 2)
                        xtick_vals = 1:D:nx
                        ytick_vals = 1:D:ny
                        xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                        ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                        xticks = (xtick_vals, xtick_labels)
                        yticks = (ytick_vals, ytick_labels)
                        plt_P_flood = Plots.heatmap(
                            P[:,:,Int(z)]';
                            xlabel = L"x/D", ylabel = L"y/D",
                            xticks = xticks, yticks = yticks,
                            xlims = (-5,nx+5), ylims = (-5,ny+5),
                            color = :vik25,
                            clims = clims_p,
                            colorbar = true, colorbar_title = L"\overline{P}",
                            levels = 25,                     
                            size = (600, 350),
                            aspect_ratio = :equal,  
                            tickfontsize = 10, labelfontsize = 10,
                            legendfontsize = 10, legend = :topright,
                            left_margin = Plots.Measures.Length(:mm, 5),
                        )
                        Plots.contour!(permutedims(P[:,:,Int(z)]), levels=range(clims_p[1],clims_p[2],step=pstep), color=:black, linewidth=1)
                        ϕ = range(0, 2π; length=100)
                        Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        if vectors
                            draw_force_vectors!(o₁[1], o₁[2], meanforces.C̄x₁, meanforces.C̄y₁; name="1", scale=fscale, arrow=:simple, arrowsize=0.1, lw=1, components=true, annotate_tip=true)
                            draw_force_vectors!(o₂[1], o₂[2], meanforces.C̄x₂, meanforces.C̄y₂; name="2", scale=fscale, arrow=:simple, arrowsize=0.1, lw=1, components=true, annotate_tip=true)
                        end
                        savefig(plt_P_flood, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_P_flood_at_$(z/D).pdf")
                    end

                    Mval = Printf.@sprintf("%.1f", prod(L .* D) / 1e6)
                    xD, yD, zD = u_probe_loc
                    main_lab = latexstring("u_{x}(t), D=$(D), λ_{1}=$(λ₁), λ_{2}=$(λ₂), \\theta=$(θ), Loc.=($(u_probe_loc[1])D, $(u_probe_loc[2])D, $(u_probe_loc[3])D)")
                    avg_lab  = L"u_{x},\ \mathrm{moving\ avg.}"
                    sig_lab  = L"u_{x},\ \mathrm{moving}\ \sigma" 
                    main_lab_x_y = latexstring("(u_{x}(t),u_{y}(t)), D=$(D), λ_{1}=$(λ₁), λ_{2}=$(λ₂), \\theta=$(θ), Loc.=($(u_probe_loc[1])D, $(u_probe_loc[2])D, $(u_probe_loc[3])D)")
                    avg_lab_x_y = L"\mathrm{moving\ avg.}\ (u_{x},u_{y})"
                    sig_lab_x_y = L"\mathrm{moving}\ \sigma\ (u_{x},u_{y})"
                    main_lab_force = latexstring("C_{L_{1}}(t), D=$(D), λ_{1}=$(λ₁), λ_{2}=$(λ₂), \theta=$(θ), Loc.=($(u_probe_loc[1])D, $(u_probe_loc[2])D, $(u_probe_loc[3])D)")
                    force_lab_1 = L"C_{L_{2}}(t)"
                    force_lab_2 = L"C_{D_{1}}(t)"
                    force_lab_3 = L"C_{D_{2}}(t)"
                    plt_u_probe_x = Plots.plot(
                        t_full, u_probe_x_full,
                        label = main_lab,
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"u/U_{\infty}",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (0, t_full[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :gray, alpha = 0.6
                    )
                    Plots.plot!(plt_u_probe_x, t_full, U_probe_x_mean; label = avg_lab, lw = 2, color="#e66101")
                    Plots.plot!(plt_u_probe_x, t_full, σ_U_probe_x; label = sig_lab, lw = 2, color=:green)
                    savefig(plt_u_probe_x, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_mean.pdf")

                    plt_u_probe_x_box = Plots.plot(
                        t_full, u_probe_x_full,
                        label = main_lab,
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"$u/U_{\infty}$",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (stats_init, t_stats[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :gray, alpha = 0.6
                    )
                    Plots.plot!(plt_u_probe_x_box, t_full, U_probe_x_mean, label=avg_lab, lw =2, color="#e66101")
                    Plots.plot!(plt_u_probe_x_box, t_full, σ_U_probe_x, label=sig_lab, lw=2, color=:green)
                    savefig(plt_u_probe_x_box, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_box.pdf")

                    plt_u_probe_x_stats = Plots.plot(
                        t_stats, u_probe_x_stats,
                        label = main_lab,
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"$u/U_{\infty}$",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (stats_init, t_stats[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :gray, alpha = 0.6
                    )
                    Plots.plot!(plt_u_probe_x_stats, t_stats, U_probe_x_stats, label=avg_lab, lw=2, color="#e66101")
                    Plots.plot!(plt_u_probe_x_stats, t_stats, σ_U_probe_x_stats, label=sig_lab, lw=2, color=:green)
                    savefig(plt_u_probe_x_stats, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_stats.pdf")

                    plt_u_probe_x_trim = Plots.plot(
                        t_trim, u_probe_x_trim,
                        label = main_lab,
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"$u/U_{\infty}$",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (stats_trim, t_trim[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :gray, alpha = 0.6
                    )
                    Plots.plot!(plt_u_probe_x_trim, t_trim, U_probe_x_trim, label=avg_lab, lw=2,color="#e66101")
                    Plots.plot!(plt_u_probe_x_trim, t_trim, σ_U_probe_x_trim, label=sig_lab, lw=2, color=:green)
                    savefig(plt_u_probe_x_trim, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_trim.pdf")

                    plt_u_probe_x_trim_box = Plots.plot(
                        t_trim, u_probe_x_trim,
                        label = main_lab,
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"$u/U_{\infty}$",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (stats_init, t_stats[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :gray, alpha = 0.6
                    )
                    Plots.plot!(plt_u_probe_x_trim_box, t_trim, U_probe_x_trim, label=avg_lab, lw=2,color="#e66101")
                    Plots.plot!(plt_u_probe_x_trim_box, t_trim, σ_U_probe_x_trim, label=sig_lab, lw=2, color=:green)
                    savefig(plt_u_probe_x_trim_box, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_trim_box.pdf")

                    plt_u_probe_x_y = Plots.scatter(
                        u_probe_x_trim, u_probe_y_trim;
                        label = main_lab_x_y,
                        xlabel = L"u_{x}/U_{\infty}", ylabel = L"u_{y}/U_{\infty}",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 8, legend = :top,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        markersize = 2, markerstrokewidth = 0.1, alpha = 0.5,   # scatter styling
                        aspect_ratio = :equal
                    )
                    Plots.scatter!(plt_u_probe_x_y, U_probe_x_mean[1:5:end], U_probe_y_mean[1:5:end];label = avg_lab_x_y,  markersize = 2, markerstrokewidth = 0.1, alpha = 1, color = "#e66101")
                    Plots.scatter!(plt_u_probe_x_y, σ_U_probe_x[1:5:end], σ_U_probe_y[1:5:end];label = sig_lab_x_y,  markersize = 2, markerstrokewidth = 0.1, alpha = 1, color = :green)
                    savefig(plt_u_probe_x_y, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_U_probe_x_y.pdf")

                    plt_forces = Plots.plot(
                        t_full, forces.Cx₁,
                        label = L"C_{D_{1}}(t)",
                        xlabel = L"$tU_{\infty}/D$", ylabel = L"$C_{D}-C_{L}$",
                        framestyle = :box, grid = true,
                        size = (600, 600),
                        xlims = (0, t_full[end]),
                        tickfontsize = 12, labelfontsize = 12, legendfontsize = 10, legend = :best,
                        left_margin = Plots.Measures.Length(:mm, 5),
                        lw = 2, color = :green, alpha = 1
                    )
                    Plots.plot!(plt_forces, t_full, forces.Cx₂, label=L"C_{D_{2}}(t)", lw=2)
                    Plots.plot!(plt_forces, t_full, forces.Cx₁+forces.Cx₂, label=L"C_{D_{tot.}}(t)", lw=2)
                    Plots.plot!(plt_forces, t_full, forces.Cy₁, label=L"C_{L_{1}}(t)", lw=2)
                    Plots.plot!(plt_forces, t_full, forces.Cy₂, label=L"C_{L_{2}}(t)", lw=2)
                    Plots.plot!(plt_forces, t_full, forces.Cy₂+forces.Cy₁, label=L"C_{L_{tot.}}(t)", lw=2)

                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄x₁, meanforces.C̄x₁], color=:orange, ls=:dash, lw=2, label=L"\overline{C}_{D_{1}}")
                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄x₂, meanforces.C̄x₂], color=:green, ls=:dash, lw=2, label=L"\overline{C}_{D_{2}}")
                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄x₁+meanforces.C̄x₂, meanforces.C̄x₁+meanforces.C̄x₂], color=:brown, ls=:dash, lw=2, label=L"\overline{C}_{D_{tot.}}")
                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄y₁, meanforces.C̄y₁], color=:blue, ls=:dash, lw=2, label=L"\overline{C}_{L_{1}}")
                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄y₂, meanforces.C̄y₂], color=:purple, ls=:dash, lw=2, label=L"\overline{C}_{L_{2}}")
                    Plots.plot!(plt_forces, [stats_init, t_full[end]], [meanforces.C̄y₁+meanforces.C̄y₂, meanforces.C̄y₁+meanforces.C̄y₂], color=:magenta, ls=:dash, lw=2, label=L"\overline{C}_{L_{tot.}}")
                    vline!(plt_forces, [stats_init], color=:black, ls=:solid, lw=2, label=false)
                    savefig(plt_forces, string(@__DIR__) * "../../tex/img/rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)_forces.pdf")
                end
                fig_path = joinpath(string(@__DIR__), pdf_file)
                println("Figure stored in $(fig_path)")
                savefig(fig_path) 
            end
        end
    end
end
main()






