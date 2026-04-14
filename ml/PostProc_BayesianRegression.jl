include("POD.jl")
include("ML.jl")
using HDF5, JLD2
using OutMacro

λ1s = [3,4,5]
λ2s = [3,4,5]
θs = [0,15,30]
λ1_recs = [3,4,5]
λ2_recs = [3,4,5]
θ_recs = [0,15,30]
z_vals = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5]
recover_fields = true
plot_fields = true
field_err = true 
clims_p = (-3,3)
ustep = 0.25
pstep = 0.25

recdir  = "/ml/data/rotor_BiotSimulation/recovered/"
datadir = "/ml/data/rotor_BiotSimulation/prediction/"
meanflowdir = "/sims/data/rotor_BiotSimulation/" 
PODdir  = "/ml/data/rotor_BiotSimulation/POD/"
h5 = h5open(datadir*"PredCoeffs.h5", "r")
A = read(h5["pred/aUx"]); aUx_bayes = permutedims(A, (2,1))
A = read(h5["pred/aUy"]); aUy_bayes = permutedims(A, (2,1))
A = read(h5["pred/aUz"]); aUz_bayes = permutedims(A, (2,1))
A = read(h5["pred/aP"]);  aP_bayes  = permutedims(A, (2,1))
A = read(h5["pred/params"]); params_bayes = permutedims(A, (2,1))
close(h5)
r = size(aUx_bayes,1)
@out aUx_bayes 
@out params_bayes
_, nsnap, C_Ux, C_Uy, C_Uz, C_P = truncate(r; dir=PODdir)

key = _keys(λ1s, λ2s, θs)
nsnap = length(key)

all_pred = Dict(:aUx=>Float32[], :aUy=>Float32[], :aUz=>Float32[], :aP=>Float32[])
all_true = Dict(:aUx=>Float32[], :aUy=>Float32[], :aUz=>Float32[], :aP=>Float32[])
for λ1_rec in λ1s 
    for λ2_rec in λ2s
        for θ_rec in θs
            j = snapshot_col_index(λ1s,λ2s,θs,λ1_rec,λ2_rec,θ_rec)
            @out j
            append!(all_pred[:aUx], vec(aUx_bayes[:,j])); append!(all_true[:aUx], vec(C_Ux[:, j]))
            append!(all_pred[:aUy], vec(aUy_bayes[:,j])); append!(all_true[:aUy], vec(C_Uy[:, j]))
            append!(all_pred[:aUz], vec(aUz_bayes[:,j])); append!(all_true[:aUz], vec(C_Uz[:, j]))
            append!(all_pred[:aP], vec(aP_bayes[:,j])); append!(all_true[:aP],  vec(C_P[:, j]))
        end
    end
end
return all_pred, all_true


paritydir="/ml/tex/errors/"
x  = all_pred[:aUx]               
y  = all_true[:aUx]               
xf = 1e4; yf = 1e4
param_list = key
cols = [
    :red, :blue, :green, :magenta, :orange, :cyan,
    :black, :purple, :brown, :olive, :navy, :teal,
    :pink, :gray, :gold, :indigo, :seagreen, :tomato,
    :coral, :darkorange, :darkgreen, :darkred, :dodgerblue,
    :chocolate, :slategray, :darkviolet, :turquoise, :darkkhaki,
    :deeppink
]
@assert length(cols) >= nsnap "Provide at least nsnaps colors in `cols`"
snap_labels = [latexstring("\\lambda_{1}=$(λ1), \\lambda_{2}=$(λ2), \\theta=$(θ)") for (λ1,λ2,θ) in key][1:nsnap]
lo = min(minimum(x), minimum(y))
hi = max(maximum(x), maximum(y))
Parity_Ux = Plots.plot(
    xlabel = L"a_{\star,i}^{(\overline{u}_{x})}\ \times 10^{4}",
    ylabel = L"a_{i}^{(\overline{u}_{x})}\ \times 10^{4}",
    size = (600, 600),
    framestyle = :box, grid = true, minorgrid = true,
    legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
    xformatter = v -> @sprintf("%.3f", v / xf),
    yformatter = v -> @sprintf("%.3f", v / yf),
)
for s in 1:nsnap
    idx = (s-1)*r + 1 : s*r
    Plots.scatter!(
        Parity_Ux, x[idx], y[idx];
        color = cols[s], ms = 3, markerstrokewidth = 0.5,
        label = snap_labels[s]
    )
end
Plots.plot!(Parity_Ux, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
savefig(Parity_Ux, joinpath(paritydir, "Parity_Ux_Bayesian_jl.pdf"))

x  = all_pred[:aUy]              
y  = all_true[:aUy]              
xf = 1e4; yf = 1e4
lo = min(minimum(x), minimum(y))
hi = max(maximum(x), maximum(y))
Parity_Uy = Plots.plot(
xlabel = L"a_{\star,i}^{(\overline{u}_{y})}\ \times 10^{4}",
ylabel = L"a_{i}^{(\overline{u}_{y})}\ \times 10^{4}",
size = (600, 600),
framestyle = :box, grid = true, minorgrid = true,
legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
xformatter = v -> @sprintf("%.3f", v / xf),
yformatter = v -> @sprintf("%.3f", v / yf),
)
for s in 1:nsnap
idx = (s-1)*r + 1 : s*r
Plots.scatter!(
Parity_Uy, x[idx], y[idx];
color = cols[s], ms = 3, markerstrokewidth = 0.5,
label = snap_labels[s]
)
end
Plots.plot!(Parity_Uy, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
savefig(Parity_Uy, joinpath(paritydir, "Parity_Uy_Bayesian_jl.pdf"))

x  = all_pred[:aUz]              
y  = all_true[:aUz]              
xf = 1e3; yf = 1e3
lo = min(minimum(x), minimum(y))
hi = max(maximum(x), maximum(y))
Parity_Uz = Plots.plot(
xlabel = L"a_{\star,i}^{(\overline{u}_{z})}\ \times 10^{3}",
ylabel = L"a_{i}^{(\overline{u}_{z})}\ \times 10^{3}",
size = (600, 600),
framestyle = :box, grid = true, minorgrid = true,
legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
xformatter = v -> @sprintf("%.3f", v / xf),
yformatter = v -> @sprintf("%.3f", v / yf),
)
for s in 1:nsnap
idx = (s-1)*r + 1 : s*r
Plots.scatter!(
Parity_Uz, x[idx], y[idx];
color = cols[s], ms = 3, markerstrokewidth = 0.5,
label = snap_labels[s]
)
end
Plots.plot!(Parity_Uz, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
savefig(Parity_Uz, joinpath(paritydir, "Parity_Uz_Bayesian_jl.pdf"))

x  = all_pred[:aP]              
y  = all_true[:aP]              
xf = 1e4; yf = 1e4
lo = min(minimum(x), minimum(y))
hi = max(maximum(x), maximum(y))
Parity_P = Plots.plot(
xlabel = L"a_{\star,i}^{(\overline{P})}\ \times 10^{4}",
ylabel = L"a_{i}^{(\overline{P})}\ \times 10^{4}",
size = (600, 600),
framestyle = :box, grid = true, minorgrid = true,
legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
xformatter = v -> @sprintf("%.3f", v / xf),
yformatter = v -> @sprintf("%.3f", v / yf),
)
for s in 1:nsnap
idx = (s-1)*r + 1 : s*r
Plots.scatter!(
Parity_P, x[idx], y[idx];
color = cols[s], ms = 3, markerstrokewidth = 0.5,
label = snap_labels[s]
)
end
Plots.plot!(Parity_P, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
savefig(Parity_P, joinpath(paritydir, "Parity_P_Bayesian_jl.pdf"))


for λ1_rec in λ1s 
    for λ2_rec in λ2s
        for θ_rec in θs
            if recover_fields
                println("🧮 Recovering new flow field data: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                j = snapshot_col_index(λ1s,λ2s,θs,λ1_rec,λ2_rec,θ_rec)
                aUx = aUx_bayes[:,j]; aUy = aUy_bayes[:,j]; aUz = aUz_bayes[:,j]; aP = aP_bayes[:,j]
                U_rec, P_rec = reconstruct_fields(r, aUx, aUy, aUz, aP; dir=PODdir)
                jldsave(recdir*"rec_fields_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2";
                    P_rec = Array(P_rec),
                    U_rec = Array(U_rec)
                )
            else
                P_rec, U_rec = read_recovered_fields("rec_fields_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2"; dir=recdir)
            end

            if plot_fields
                println("🧮 Plotting new flow fields: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                D = 72
                o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
                o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
                radii  = (D-8)/2  
                for z in z_vals.*D
                    nx = size(U_rec[:,:,:,1], 1)
                    ny = size(U_rec[:,:,:,1], 2)
                    xtick_vals = 1:D:nx
                    ytick_vals = 1:D:ny
                    xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                    ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                    xticks = (xtick_vals, xtick_labels)
                    yticks = (ytick_vals, ytick_labels)
                    Ux_clims = (-(max(λ1_rec,λ2_rec)-2),max(λ1_rec,λ2_rec)) 
                    plt_Ux_flood = Plots.heatmap(
                        U_rec[:,:,Int(z),1]';
                        xlabel = L"x/D", ylabel = L"y/D",
                        xticks = xticks, yticks = yticks,
                        xlims = (-5,nx+5), ylims = (-5,ny+5),
                        color = :vik50,
                        clims = (Ux_clims),
                        colorbar = true, colorbar_title = L"\overline{u}_{x}/U_{\infty}",
                        levels = 50,                     
                        size = (600, 350),
                        aspect_ratio = :equal,  
                        tickfontsize = 10, labelfontsize = 10,
                        legendfontsize = 10, legend = :topright,
                        left_margin = Plots.Measures.Length(:mm, 5),
                    )
                    Plots.contour!(permutedims(U_rec[:,:,Int(z),1]), levels=range(Ux_clims[1],Ux_clims[2],step=ustep), color=:black, linewidth=1)
                    ϕ = range(0, 2π; length=100)
                    Plots.plot!(o₁[1] .+ radii*cos.(ϕ), o₁[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    Plots.plot!(o₂[1] .+ radii*cos.(ϕ), o₂[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    savefig(plt_Ux_flood, string(@__DIR__) * "/tex/Ux_rec/Bayesian/Ux_rec_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                end

                for z in z_vals.*D
                    nx = size(P_rec, 1)
                    ny = size(P_rec, 2)
                    xtick_vals = 1:D:nx
                    ytick_vals = 1:D:ny
                    xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                    ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                    xticks = (xtick_vals, xtick_labels)
                    yticks = (ytick_vals, ytick_labels)
                    Ux_clims = (-3,3) #(minimum(Ux_rec[:,:,Int(z)]), maximum(Ux_rec[:,:,Int(z)]))
                    plt_P_flood = Plots.heatmap(
                        P_rec[:,:,Int(z)]';
                        xlabel = L"x/D", ylabel = L"y/D",
                        xticks = xticks, yticks = yticks,
                        xlims = (-5,nx+5), ylims = (-5,ny+5),
                        color = :vik50,
                        clims = (Ux_clims),
                        colorbar = true, colorbar_title = L"\overline{P}",
                        levels = 50,                     
                        size = (600, 350),
                        aspect_ratio = :equal,  
                        tickfontsize = 10, labelfontsize = 10,
                        legendfontsize = 10, legend = :topright,
                        left_margin = Plots.Measures.Length(:mm, 5),
                    )
                    Plots.contour!(permutedims(P_rec[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=pstep), color=:black, linewidth=1)
                    ϕ = range(0, 2π; length=100)
                    Plots.plot!(o₁[1] .+ radii*cos.(ϕ), o₁[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    Plots.plot!(o₂[1] .+ radii*cos.(ϕ), o₂[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    savefig(plt_P_flood, string(@__DIR__) * "/tex/P_rec/Bayesian/P_rec_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                end
            end

            if field_err
                println("🧮 Calculating field errors: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                P_tru, U_tru, _ = read_meanflow("meanflow_rotor_BiotSimulation_72_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2"; dir=meanflowdir, stats=true, stats_turb=false)
                Ux_err = ae(U_tru[:,:,:,1], U_rec[:,:,:,1])
                Ux_mse = mse(U_tru[:,:,:,1], U_rec[:,:,:,1])
                P_err = ae(P_tru, P_rec)
                P_mse = mse(P_tru, P_rec)
                @out size(U_tru)
                @out size(U_rec)
                @out size(Ux_err)
                @out Ux_mse
                @out size(P_tru)
                @out size(P_rec)
                @out size(P_err)
                @out P_mse

                println("🧮 Plotting field errors: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                D = 72
                o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
                o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
                radii  = (D-8)/2  
                for z in z_vals.*D
                    nx = size(Ux_err, 1)
                    ny = size(Ux_err, 2)
                    xtick_vals = 1:D:nx
                    ytick_vals = 1:D:ny
                    xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                    ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                    xticks = (xtick_vals, xtick_labels)
                    yticks = (ytick_vals, ytick_labels)
                    Ux_clims = (0,1)
                    plt_Ux_err_flood = Plots.heatmap(
                        Ux_err[:,:,Int(z)]';
                        xlabel = L"x/D", ylabel = L"y/D",
                        xticks = xticks, yticks = yticks,
                        xlims = (-5,nx+5), ylims = (-5,ny+5),
                        color = :vik50,
                        clims = (Ux_clims),
                        title = latexstring("MSE \$\\overline{u}_{x} = $(Ux_mse)\$"),
                        colorbar = true, colorbar_title = latexstring("Absolute Error \$\\overline{u}_{x}\$"),
                        levels = 50,                     
                        size = (600, 350),
                        aspect_ratio = :equal,  
                        titlefontsize = 10, tickfontsize = 10, labelfontsize = 10,
                        legendfontsize = 10, legend = :topright,
                        left_margin = Plots.Measures.Length(:mm, 5),
                    )
                    Plots.contour!(permutedims(Ux_err[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=0.1), color=:black, linewidth=1)
                    ϕ = range(0, 2π; length=100)
                    Plots.plot!(o₁[1] .+ radii*cos.(ϕ), o₁[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    Plots.plot!(o₂[1] .+ radii*cos.(ϕ), o₂[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    savefig(plt_Ux_err_flood, string(@__DIR__) * "/tex/Ux_err/Bayesian/Ux_err_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                end

                for z in z_vals.*D
                    nx = size(P_err, 1)
                    ny = size(P_err, 2)
                    xtick_vals = 1:D:nx
                    ytick_vals = 1:D:ny
                    xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                    ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                    xticks = (xtick_vals, xtick_labels)
                    yticks = (ytick_vals, ytick_labels)
                    Ux_clims = (0,1) #(minimum(P_err[:,:,Int(z)]), maximum(P_err[:,:,Int(z)]))
                    plt_P_err_flood = Plots.heatmap(
                        P_err[:,:,Int(z)]';
                        xlabel = L"x/D", ylabel = L"y/D",
                        xticks = xticks, yticks = yticks,
                        xlims = (-5,nx+5), ylims = (-5,ny+5),
                        color = :vik50,
                        clims = (Ux_clims),
                        title = latexstring("MSE \$\\overline{P} = $(P_mse)\$"),
                        colorbar = true, colorbar_title = latexstring("Absolute Error \$\\overline{P}\$"),
                        levels = 50,                     
                        size = (600, 350),
                        aspect_ratio = :equal,  
                        titlefontsize = 10, tickfontsize = 10, labelfontsize = 10,
                        legendfontsize = 10, legend = :topright,
                        left_margin = Plots.Measures.Length(:mm, 5),
                    )
                    Plots.contour!(permutedims(P_err[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=0.1), color=:black, linewidth=1)
                    ϕ = range(0, 2π; length=100)
                    Plots.plot!(o₁[1] .+ radii*cos.(ϕ), o₁[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    Plots.plot!(o₂[1] .+ radii*cos.(ϕ), o₂[2] .+ radii*sin.(ϕ), color=:red, lw=2, label="")
                    savefig(plt_P_err_flood, string(@__DIR__) * "/tex/P_err/Bayesian/P_err_B_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                end
            end
        end
    end
end
