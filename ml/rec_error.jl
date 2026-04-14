include("POD.jl")
include("ML.jl")
using HDF5, JLD2
using OutMacro

recdir = "/ml/data/rotor_BiotSimulation/recovered/"
trdir = "/ml/data/rotor_BiotSimulation/recovered/"
datadir = "/sims/data/rotor_BiotSimulation/"

λ1s = [3,4,5]
λ2s = [3,4,5]
θs = [0,15,30]

Ux_err_MLP = []; Ux_err_B = [] 
Uy_err_MLP = []; Uy_err_B = []
Uz_err_MLP = []; Uz_err_B = []
P_err_MLP = []; P_err_B = []

for λ1 in λ1s
    for λ2 in λ2s
        for θ in θs

            P_MLP, U_MLP = read_recovered_fields("rec_fields_$(λ1)_$(λ2)_$(θ).jld2";dir=recdir)
            P_B, U_B = read_recovered_fields("rec_fields_B_$(λ1)_$(λ2)_$(θ).jld2";dir=recdir)

            println("🧮 Calculating field errors: λ₁ = $(λ1), λ₂ = $(λ2), θ = $(θ)")
            P_tru, U_tru, _ = read_meanflow("meanflow_rotor_BiotSimulation_72_$(λ1)_$(λ2)_$(θ).jld2"; dir=datadir, stats=true, stats_turb=false)
            Ux_tru, Uy_tru, Uz_tru, = U_tru[:,:,:,1], U_tru[:,:,:,2], U_tru[:,:,:,3]
            Ux_mse_MLP = mse(Ux_tru, U_MLP[:,:,:,1])
            Uy_mse_MLP = mse(Uy_tru, U_MLP[:,:,:,2])
            Uz_mse_MLP = mse(Uz_tru, U_MLP[:,:,:,3])
            P_mse_MLP = mse(P_tru, P_MLP)
            push!(Ux_err_MLP, Ux_mse_MLP); push!(Uy_err_MLP, Uy_mse_MLP); push!(Uz_err_MLP, Uz_mse_MLP); push!(P_err_MLP, P_mse_MLP)

            Ux_mse_B = mse(Ux_tru, U_B[:,:,:,1])
            Uy_mse_B = mse(Uy_tru, U_B[:,:,:,2])
            Uz_mse_B = mse(Uz_tru, U_B[:,:,:,3])
            P_mse_B = mse(P_tru, P_B)
            push!(Ux_err_B, Ux_mse_B); push!(Uy_err_B, Uy_mse_B); push!(Uz_err_B, Uz_mse_B); push!(P_err_B, P_mse_B)
        end
    end
end

using Plots

# ----------------------------
# 1) Convert error lists to arrays
# ----------------------------
Ux_err_MLP = Float64.(Ux_err_MLP)
Uy_err_MLP = Float64.(Uy_err_MLP)
Uz_err_MLP = Float64.(Uz_err_MLP)
P_err_MLP  = Float64.(P_err_MLP)

Ux_err_B = Float64.(Ux_err_B)
Uy_err_B = Float64.(Uy_err_B)
Uz_err_B = Float64.(Uz_err_B)
P_err_B  = Float64.(P_err_B)

# ----------------------------
# 2) Snapshot index
# ----------------------------
nsnap = length(Ux_err_MLP)
snapshots = 1:nsnap

function plot_field_errors(snapshots, err_MLP, err_B, lab_MLP, lab_BR, color1, color2, filename)

    plt = Plots.plot(
        snapshots, err_MLP;
        label      = lab_MLP,
        xlabel     = "Snapshot",
        ylabel     = "MSE",
        size       = (600, 600),
        lw         = 2,
        marker     = :circle,
        linestyle = :solid,
        markersize = 4,
        color      = color1,
        legend     = :topright,
        legendfontsize = 8,
        tickfontsize   = 8,
        labelfontsize  = 10,
        markerstrokewidth = 0.8,
        xticks = (1:nsnap, string.(1:nsnap)),   # major ticks at all snapshots
    )

    Plots.plot!(
        plt, snapshots, err_B;
        label      = lab_BR,
        lw         = 2,
        marker     = :diamond,
        markersize = 4,
        linestyle = :dash,
        color      = color2,
        markerstrokewidth = 0.8,
    )

    savefig(plt, filename)
    println("Saved → $filename")
end

# ========== 3) Create separate scatter plots ==========
plot_field_errors(
    snapshots, Ux_err_MLP, Ux_err_B,
    latexstring("\$U_{x}\$ — MLP"), latexstring("\$U_{x}\$ — BR"),
    :red, :pink,
    "MSE_Ux_log.pdf"
)

plot_field_errors(
    snapshots, Uy_err_MLP, Uy_err_B,
    latexstring("\$U_{y}\$ — MLP"), latexstring("\$U_{y}\$ — BR"),
    :blue, :skyblue,
    "MSE_Uy_log.pdf"
)

plot_field_errors(
    snapshots, Uz_err_MLP, Uz_err_B,
    latexstring("\$U_{z}\$ — MLP"), latexstring("\$U_{z}\$ — BR"),
    :green, :lightgreen,
    "MSE_Uz_log.pdf"
)

plot_field_errors(
    snapshots, P_err_MLP, P_err_B,
    latexstring("\$P\$ — MLP"), latexstring("\$P\$ — BR"),
    :purple, :violet,
    "MSE_P_log.pdf"
)