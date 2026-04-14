using LaTeXStrings
include("POD.jl")

D = 72
λ1s = [3,4,5]
λ2s = [3,4,5]
θs = [0,15,30]
T=Float32
run_POD = false
_write = false
postprocess = true
datadir = "/sims/data/rotor_BiotSimulation/"
outdir = "/ml/data/rotor_BiotSimulation/POD/"
z_vals = T.([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])
ustep = 20
function main()
    mkpath(outdir)
    paths = nothing
    N     = nothing
    m     = nothing
    # 1) Optionally (re)write snapshots
    if _write
        println("🧮 Reading MeanFlow and creating snapshots")
        # NOTE: add stats_turb=true or false as appropriate for your case
        paths, N, m = write_snapshots_to_disk(D, λ1s, λ2s, θs; dir=datadir, outdir=outdir, T=T)
        println("✅ Snapshots written!")
    end

    if run_POD
        if paths === nothing
            println("📂 Using existing snapshot binaries on disk")
            paths, N, m = generate_paths(;dir=datadir, outdir=outdir)
        else
            println("📂 Using snapshots from this run (write_snapshots_to_disk)")
        end

        println("🧮 Running POD")
        Σ_U, Φ_U, A_U, Σ_P, Φ_P, A_P = run_POD_from_disk(paths, N, m)

        println("💾 Writing POD results")
        jldsave(joinpath(outdir, "POD.jld2");
                Σ_U=Σ_U, A_U=A_U, Σ_P=Σ_P, A_P=A_P
        )

        println("🧮 Recovering POD modes")
        recover_POD_modes(Φ_U, Φ_P, (λ1s[1], λ2s[1], θs[1]); D=D, dir=datadir, outdir=outdir, T=T)
        println("✅ POD done!")
    end

    if postprocess
        println("🧮 Postprocessing")
        Σ_U, A_U, Σ_P, A_P = read_POD("POD.jld2"; dir=outdir)
        Σ_Ux, Σ_Uy, Σ_Uz = Σ_U[:,:,1], Σ_U[:,:,2], Σ_U[:,:,3]
        A_Ux, A_Uy, A_Uz = A_U[:,:,1], A_U[:,:,2], A_U[:,:,3]
        nmode,_,_ = size(Σ_U)
        for k in 1:nmode
            Mode_U, Mode_P, idx = read_modes("POD_mode_$(k).jld2"; dir=outdir)  
            Mode_Ux, Mode_Uy, Mode_Uz = Mode_U[:,:,:,1], Mode_U[:,:,:,2], Mode_U[:,:,:,3] 
            o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
            o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
            r  = (D-8)/2
            for z in z_vals.*D
                nx = size(Mode_Ux, 1)
                ny = size(Mode_Ux, 2)
                xtick_vals = 1:D:nx
                ytick_vals = 1:D:ny
                xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                xticks = (xtick_vals, xtick_labels)
                yticks = (ytick_vals, ytick_labels)
                Ux_clims = (minimum(Mode_Ux[:,:,Int(z)]), maximum(Mode_Ux[:,:,Int(z)]))
                Mode_Ux_flood = Plots.heatmap(
                    Mode_Ux[:,:,Int(z)]';
                    xlabel = L"x/D", ylabel = L"y/D",
                    xticks = xticks, yticks = yticks,
                    xlims = (-5,nx+5), ylims = (-5,ny+5),
                    color = :vik50,
                    clims = (Ux_clims),
                    colorbar = false,
                    # colorbar = true, colorbar_title = latexstring("U_{x} POD mode $(k)"),
                    levels = 50,                     
                    size = (600, 350),
                    aspect_ratio = :equal,  
                    tickfontsize = 10, labelfontsize = 10,
                    legendfontsize = 10, legend = :topright,
                    left_margin = Plots.Measures.Length(:mm, 5),
                )
                ustep = (Ux_clims[2]-Ux_clims[1])/25
                Plots.contour!(Mode_Ux_flood, permutedims(Mode_Ux[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=ustep), color=:black, linewidth=1)
                ϕ = range(0, 2π; length=100)
                Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                savefig(Mode_Ux_flood, string(@__DIR__) * "/tex/Ux_modes/Ux_mode_$(k)_at_$(z/D).pdf")
            end

            for z in z_vals.*D
                nx = size(Mode_P, 1)
                ny = size(Mode_P, 2)
                xtick_vals = 1:D:nx
                ytick_vals = 1:D:ny
                xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                xticks = (xtick_vals, xtick_labels)
                yticks = (ytick_vals, ytick_labels)
                Ux_clims = (minimum(Mode_P[:,:,Int(z)]), maximum(Mode_P[:,:,Int(z)]))
                Mode_P_flood = Plots.heatmap(
                    Mode_P[:,:,Int(z)]';
                    xlabel = L"x/D", ylabel = L"y/D",
                    xticks = xticks, yticks = yticks,
                    xlims = (-5,nx+5), ylims = (-5,ny+5),
                    color = :vik50,
                    clims = (Ux_clims),
                    colorbar = false,
                    # colorbar = true, colorbar_title = latexstring("P POD mode $(k)"),
                    levels = 50,                     
                    size = (600, 350),
                    aspect_ratio = :equal,  
                    tickfontsize = 10, labelfontsize = 10,
                    legendfontsize = 10, legend = :topright,
                    left_margin = Plots.Measures.Length(:mm, 5),
                )
                ustep = (Ux_clims[2]-Ux_clims[1])/25
                Plots.contour!(Mode_P_flood, permutedims(Mode_Ux[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=ustep), color=:black, linewidth=1)
                ϕ = range(0, 2π; length=100)
                Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                savefig(Mode_P_flood, string(@__DIR__) * "/tex/P_modes/P_mode_$(k)_at_$(z/D).pdf")
            end

            # rows = modes, cols = snapshots
            nmode, nsnaps = size(A_P)
            j = 1:nsnaps
            ε = eps(Float64)
            r_keep = nmode
            p1 = Plots.plot(xlabel=L"x_{i}^{(\overline{u}_{x})}", ylabel=L"a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            for k in 1:r_keep
                Plots.plot!(p1, j, abs.(A_P[k, :]).+ε; lw=2, label=latexstring("\\phi_{M,$k}^{(\\overline{u}_{x})}"))
                Plots.xlims!(p1, 1, nmode)
                Plots.xticks!(p1, 1:nmode)
            end
            savefig(p1, string(@__DIR__) * "/tex/mode_coefficients/P_c_vs_s.pdf")

            p2 = Plots.plot(xlabel=L"x_{i}^{(\overline{u}_{x})}", ylabel=L"\sum_{i=1}^{N} a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            mode = 1:nmode
            row_sums = abs.(vec(sum(A_P, dims=2))).+ε     
            Plots.plot!(p2, mode, row_sums; lw=2, legend=false)
            Plots.xlims!(p2, 1, nmode)
            Plots.xticks!(p2, 1:nmode)
            savefig(p2, string(@__DIR__) * "/tex/mode_coefficients/P_c_vs_s_sum.pdf")

            p3 = Plots.plot(xlabel=L"\phi_{i}^{(\overline{u}_{x})}", ylabel=L"a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            j = 1:nmode
            for k in 1:nsnaps
                Plots.plot!(p3, j, abs.(A_P[:, k]).+ε; lw=2, label=latexstring("x_{M,$k}^{(\\overline{u}_{x})}"))
                Plots.xlims!(p3, 1, nmode)
                Plots.xticks!(p3, 1:nmode)
            end
            savefig(p3, string(@__DIR__) * "/tex/mode_coefficients/P_c_vs_m.pdf")

            p4 = Plots.plot(xlabel=L"\phi_{i}^{(\overline{u}_{x})}", ylabel=L"\sum_{i=1}^{N} a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            mode = 1:nmode
            column_sums = abs.(vec(sum(A_P, dims=1))).+ε     
            Plots.plot!(p4, mode, column_sums; lw=2, legend=false)
            Plots.xlims!(p4, 1, nmode)
            Plots.xticks!(p4, 1:nmode)
            savefig(p4, string(@__DIR__) * "/tex/mode_coefficients/P_c_vs_m_sum.pdf")

            p5 = Plots.plot(xlabel=L"x_{i}^{(\overline{P})}", ylabel=L"s_{i}^{(\overline{P})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            j = 1:nmode
            Plots.plot!(p5, j, Σ_P; lw=2, legend=false)
                Plots.xlims!(p5, 1, nmode)
                Plots.xticks!(p5, 1:nmode) 
            savefig(p5, string(@__DIR__) * "/tex/mode_coefficients/S_P.pdf")

            nmode, nsnaps = size(A_Ux)
            j = 1:nsnaps
            r_keep = nmode
            p6 = Plots.plot(xlabel=L"x_{i}^{(\overline{u}_{x})}", ylabel=L"a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            for k in 1:r_keep
                Plots.plot!(p6, j, abs.(A_Ux[k, :]).+ε; lw=2, label=latexstring("\\phi_{M,$k}^{(\\overline{u}_{x})}"))
                Plots.xlims!(p6, 1, nmode)
                Plots.xticks!(p6, 1:nmode) 
            end
            savefig(p6, string(@__DIR__) * "/tex/mode_coefficients/Ux_c_vs_s.pdf")

            p7 = Plots.plot(xlabel=L"x_{i}^{(\overline{u}_{x})}", ylabel=L"\sum_{i=1}^{N} a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            mode = 1:nmode
            row_sums = abs.(vec(sum(A_Ux, dims=2))).+ε     
            Plots.plot!(p7, mode, row_sums; lw=2, legend=false)
            Plots.xlims!(p7, 1, nmode)
            Plots.xticks!(p7, 1:nmode) 
            savefig(p7, string(@__DIR__) * "/tex/mode_coefficients/Ux_c_vs_s_sum.pdf")

            p8 = Plots.plot(xlabel=L"\phi_{i}^{(\overline{u}_{x})}", ylabel=L"a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            j = 1:nmode
            for k in 1:nsnaps
                Plots.plot!(p8, j, abs.(A_Ux[:, k]).+ε; lw=2, label=latexstring("x_{M,$k}^{(\\overline{u}_{x})}"))
                Plots.xlims!(p8, 1, nmode)
                Plots.xticks!(p8, 1:nmode)  
            end
            savefig(p8, string(@__DIR__) * "/tex/mode_coefficients/Ux_c_vs_m.pdf")

            p9 = Plots.plot(xlabel=L"\phi_{i}^{(\overline{u}_{x})}", ylabel=L"\sum_{i=1}^{N} a_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            mode = 1:nmode
            column_sums = abs.(vec(sum(A_Ux, dims=1))).+ε     
            Plots.plot!(p9, mode, column_sums; lw=2, legend=false)
            Plots.xlims!(p9, 1, nmode)
            Plots.xticks!(p9, 1:nmode)   
            savefig(p9, string(@__DIR__) * "/tex/mode_coefficients/Ux_c_vs_m_sum.pdf")

            p10 = Plots.plot(xlabel=L"\phi_{i}^{(\overline{u}_{x})}", ylabel=L"s_{i}^{(\overline{u}_{x})}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5)
            j = 1:nmode
            Plots.plot!(p10, j, Σ_Ux; lw=2, legend=false)
            Plots.xlims!(p10, 1, nmode)
            Plots.xticks!(p10, 1:nmode)    
            savefig(p10, string(@__DIR__) * "/tex/mode_coefficients/S_Ux.pdf")

            p11 = Plots.plot(xlabel=L"a_{i}", ylabel=L"e_{k}", size=(600,600), framestyle=:box, yscale=:log10,
            xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
            gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5, tickfontsize=10)
            j = 1:nmode
            eUx = (Σ_Ux.^2)./sum(Σ_Ux.^2) .+ ε
            eUy = (Σ_Uy.^2)./sum(Σ_Uy.^2) .+ ε
            eUz = (Σ_Uz.^2)./sum(Σ_Uz.^2) .+ ε
            eP  = (Σ_P.^2) ./sum(Σ_P.^2)  .+ ε
            Plots.plot!(p11, j, eUx; lw=2, label=latexstring("a_{i}^{(\\overline{u}_{x})}"))
            Plots.plot!(p11, j, eUy; lw=2, label=latexstring("a_{i}^{(\\overline{u}_{y})}"))
            Plots.plot!(p11, j, eUz; lw=2, label=latexstring("a_{i}^{(\\overline{u}_{z})}"))
            Plots.plot!(p11, j, eP;  lw=2, label=latexstring("a_{i}^{(\\overline{P})}"))
            Plots.xlims!(p11, 1, nmode)
            Plots.xticks!(p11, 1:nmode)
            savefig(p11, string(@__DIR__) * "/tex/mode_coefficients/energy_component.pdf")
        end
        s_Ux = vec(Array(Σ_Ux)); E_Ux = s_Ux.^2; totE_Ux = sum(E_Ux)
        s_Uy = vec(Array(Σ_Uy)); E_Uy = s_Uy.^2; totE_Uy = sum(E_Uy)
        s_Uz = vec(Array(Σ_Uz)); E_Uz = s_Uz.^2; totE_Uz = sum(E_Uz)
        s_P = vec(Array(Σ_P)); E_P = s_P.^2; totE_P = sum(E_P)
        runningE_Ux = zero(eltype(E_Ux)); runningE_Uy = zero(eltype(E_Uy)); runningE_Uz = zero(eltype(E_Uz)); runningE_P = zero(eltype(E_P)) 
        for i in 1:nmode
            runningE_Ux += E_Ux[i]
            percentE_Ux = (runningE_Ux/totE_Ux)*100
            println("Cumulative enrgy content Ux $(i): $(percentE_Ux)")
            runningE_Uy += E_Uy[i]
            percentE_Uy = (runningE_Uy/totE_Uy)*100
            println("Cumulative enrgy content Uy $(i): $(percentE_Uy)")
            runningE_Uz += E_Uz[i]
            percentE_Uz = (runningE_Uz/totE_Uz)*100
            println("Cumulative enrgy content Uz $(i): $(percentE_Uz)")
            runningE_P += E_P[i]
            percentE_P = (runningE_P/totE_P)*100
            println("Cumulative enrgy content P $(i): $(percentE_P)")
        end
        println("✅ Done!")
    end
end
main()


