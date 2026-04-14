using Plots
include("E:/Users/borau/Julia/Rotor_BiotSavartBCs/help/Forces_PostProc.jl")
include("E:/Users/borau/Julia/Rotor_BiotSavartBCs/help/MeanFlow_PostProc.jl")
include("E:/Users/borau/Julia/Rotor_BiotSavartBCs/setup/3D.jl")

Ds = [72] # diameter resolution
D = 72
λs₁, λs₂ = [3], [3]
backend = Array
L = (8,5,8)
Re = 1000
T = Float32
U = 1
θ = 0
U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
stats = true
stats_turb = true
time_max = 800 # in CTU
stats_init = 600 # in CTU
stats_trim = 100
σ_init = 600
stats_interval = 0.1 # in CTU
dump_interval = 5 # in CTU
u_probe_loc=(3,2.5,3.5) # in D
u_probe_component = 1
z_vals = T.([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])
datadir = "E:/Users/borau/Julia/Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation/"
outdir = "E:/Users/borau/Julia/Rotor_BiotSavartBCs/sims/postproc/tex/"
psolverdir = "E:/Users/borau/Julia/Rotor_BiotSavartBCs/"
pdf_file = "../../Rotor_BiotSavartBCs/sims/tex/img/rotor_BiotSimulation_validation.pdf"
fname_output = "meanflow"
verbose = true
run = false # false: postproc, true: run cases
_plot = true
plot_log = false
clims_p = (-5,5)
clims_u = (-1,3)
vectors = true

# Gap ratios
Darvishyadegari_Hassanzadeh_GD = [2.0, 3.0]
Rastan_GD = [2.5, 4]
Siddiqui_GD = [2, 4]

# Rotation rates
# Darvishyadegari_Hassanzadeh_RS = [0, 1, 2, 3, 4]
Darvishyadegari_Hassanzadeh_RS = [3, 4]
Rastan_RS = [3, 4, 5]
Siddiqui_RS = [4, 6]

# Each entry: rows = G/D, cols = R.S
# Darvishyadegari_Hassanzadeh_CD1 = [
#     1.0980  0.7990  0.0005  -3.4570 -8.7240;   # G/D = 1.5
#     # 1.0620  0.6930  -0.6300 -3.1410 -7.9690;   # G/D = 2.0
#     1.0290  0.9240  -0.2540 -2.2990 -6.2740    # G/D = 3.0
# ]

Darvishyadegari_Hassanzadeh_CD1 = [
    -3.4570 -8.7240;   # G/D = 1.5
    # -3.1410 -7.9690;   # G/D = 2.0
    -2.2990 -6.2740    # G/D = 3.0
]

Rastan_CD1 = [
    -2.3999 -7.0465 -14.7200;       # G/D = 2.5
    -1.7489 -5.0952 -10.9902;     # G/D = 4.0
    # -1.1766 -3.7940 -8.42720        # G/D = 6.0
]

# Siddiqui_CD1 = [
#     # -0.4232 -8.5767 -29.3190;       # G/D = 1.5
#     -0.4605 -7.9383 -28.1698 ;     # G/D = 2.0
#     -0.1852 -5.2160 -19.5355        # G/D = 4.0
# ]

Siddiqui_CD1 = [
    # -8.5767 -29.3190;       # G/D = 1.5
    -7.9383 -28.1698 ;     # G/D = 2.0
    -5.2160 -19.5355        # G/D = 4.0
]


# Darvishyadegari_Hassanzadeh_CD2 = [
# -0.194  -0.128   0.178   3.623   9.269;    # G/D = 1.5
# # -0.206   0.003   0.961   3.383   8.578;    # G/D = 2.0
# -0.117   0.506   0.963   2.577   6.892     # G/D = 3.0
# ]

Darvishyadegari_Hassanzadeh_CD2 = [
3.623   9.269;    # G/D = 1.5
# 3.383   8.578;    # G/D = 2.0
2.577   6.892     # G/D = 3.0
]

Rastan_CD2 = [
    2.4 7.05 14.8;    # G/D = 2.5
    1.2 5.1 10.1;     # G/D = 4.0
    # 1.2 4.0 8.5       # G/D = 6.0
]

# Siddiqui_CD2 = [
#     # 0.6670 8.6566 30.5578;      # G/D = 1.5
#     0.9960 8.0313 29.3524;      # G/D = 2.0
#     0.8844 5.3417 20.6714       # G/D = 4.0
# ]

Siddiqui_CD2 = [
    # 8.6566 30.5578;      # G/D = 1.5
    8.0313 29.3524;      # G/D = 2.0
    5.3417 20.6714       # G/D = 4.0
]

x_force1_3_3_2, y_force1_3_3_2, x_force2_3_3_2, y_force2_3_3_2 = sectional_forces_rotor(D, 3, 3, 0, [2.0]) 
Cx1_3_3_2 = x_force1_3_3_2./(0.5*D); Cy1_3_3_2 = y_force1_3_3_2./(0.5*D); Cx2_3_3_2 = x_force2_3_3_2./(0.5*D); Cy2_3_3_2 = y_force2_3_3_2./(0.5*D)

x_force1_4_4_2, y_force1_4_4_2, x_force2_4_4_2, y_force2_4_4_2 = sectional_forces_rotor(D, 4, 4, 0, [2.0]) 
Cx1_4_4_2 = x_force1_4_4_2./(0.5*D); Cy1_4_4_2 = y_force1_4_4_2./(0.5*D); Cx2_4_4_2 = x_force2_4_4_2./(0.5*D); Cy2_4_4_2 = y_force2_4_4_2./(0.5*D)

x_force1_5_5_2, y_force1_5_5_2, x_force2_5_5_2, y_force2_5_5_2 = sectional_forces_rotor(D, 5, 5, 0, [2.0]) 
Cx1_5_5_2 = x_force1_5_5_2./(0.5*D); Cy1_5_5_2 = y_force1_5_5_2./(0.5*D); Cx2_5_5_2 = x_force2_5_5_2./(0.5*D); Cy2_5_5_2 = y_force2_5_5_2./(0.5*D)

D=72
x_force1_3_3_3, y_force1_3_3_3, x_force2_3_3_3, y_force2_3_3_3 = sectional_forces_rotor(D, 3, 3, 0, [3.0]) 
Cx1_3_3_3 = x_force1_3_3_3./(0.5*D); Cy1_3_3_3 = y_force1_3_3_3./(0.5*D); Cx2_3_3_3 = x_force2_3_3_3./(0.5*D); Cy2_3_3_3 = y_force2_3_3_3./(0.5*D)

x_force1_4_4_3, y_force1_4_4_3, x_force2_4_4_3, y_force2_4_4_3 = sectional_forces_rotor(D, 4, 4, 0, [3.0]) 
Cx1_4_4_3 = x_force1_4_4_3./(0.5*D); Cy1_4_4_3 = y_force1_4_4_3./(0.5*D); Cx2_4_4_3 = x_force2_4_4_3./(0.5*D); Cy2_4_4_3 = y_force2_4_4_3./(0.5*D)

x_force1_5_5_3, y_force1_5_5_3, x_force2_5_5_3, y_force2_5_5_3 = sectional_forces_rotor(D, 5, 5, 0, [3.0]) 
Cx1_5_5_3 = x_force1_5_5_3./(0.5*D); Cy1_5_5_3 = y_force1_5_5_3./(0.5*D); Cx2_5_5_3 = x_force2_5_5_3./(0.5*D); Cy2_5_5_3 = y_force2_5_5_3./(0.5*D)

x_force1_3_3_4, y_force1_3_3_4, x_force2_3_3_4, y_force2_3_3_4 = sectional_forces_rotor(D, 3, 3, 0, [4.0]) 
Cx1_3_3_4 = x_force1_3_3_4./(0.5*D); Cy1_3_3_4 = y_force1_3_3_4./(0.5*D); Cx2_3_3_4 = x_force2_3_3_4./(0.5*D); Cy2_3_3_4 = y_force2_3_3_4./(0.5*D)

x_force1_4_4_4, y_force1_4_4_4, x_force2_4_4_4, y_force2_4_4_4 = sectional_forces_rotor(D, 4, 4, 0, [4.0]) 
Cx1_4_4_4 = x_force1_4_4_4./(0.5*D); Cy1_4_4_4 = y_force1_4_4_4./(0.5*D); Cx2_4_4_4 = x_force2_4_4_4./(0.5*D); Cy2_4_4_4 = y_force2_4_4_4./(0.5*D)

x_force1_5_5_4, y_force1_5_5_4, x_force2_5_5_4, y_force2_5_5_4 = sectional_forces_rotor(D, 5, 5, 0, [4.0]) 
Cx1_5_5_4 = x_force1_5_5_4./(0.5*D); Cy1_5_5_4 = y_force1_5_5_4./(0.5*D); Cx2_5_5_4 = x_force2_5_5_4./(0.5*D); Cy2_5_5_4 = y_force2_5_5_4./(0.5*D)


function potential_force(λ)
    D = 72
    a = D/2
    Γ = 2*π*a*λ
    ΔC = ((Γ^2)/(12*π*a))/(0.5*π*D)
    return ΔC
end
λs = 2.9:0.1:8
ΔCD = [potential_force(λ) for λ in λs]

plt = Plots.plot(
    # yscale = :log10,                # <-- Add this line
    xlabel = L"λ",
    ylabel = L"\overline{C}_{D}",
    xlims = (2.9, 6.1),
    # ylims = (1e-1, 1e2),            # <-- log scale requires positive y-values
    grid = true,
    size = (800, 600),
    framestyle = :box,
    legend = :bottomleft,
    legend_columns = 2,
    labelfontsize = 16,
    tickfontsize = 14,
    legendfontsize = 12,
    xgrid=true, ygrid=true, xminorgrid=false, yminorgrid=true, gridalpha=0.2, minorgridalpha=0.2,
    gridcolor=:black, minorgridcolor=:black, gridlinewidth=0.5, minorgridlinewidth=0.5
)
Plots.plot!(plt, λs, ΔCD, label="Potential theory", color=:red, linewidth=3)
Plots.plot!(plt, λs, -ΔCD, label=false, color=:red, linewidth=3)

for (i, GD) in enumerate(Darvishyadegari_Hassanzadeh_GD)
    lbl1 = i == 1 ? L"C_{D_{1}}" * ", 2D Studies" : false
    lbl2 = i == 1 ? L"C_{D_{2}}" * ", 2D Studies" : false
    Plots.scatter!(
        plt, Darvishyadegari_Hassanzadeh_RS, Darvishyadegari_Hassanzadeh_CD1[i, :],
        label = lbl1,
        xlabel=L"λ",
        ylabel=L"\overline{C}_{x}",
        # xlims=(2.9,6.1),
        ylims=(-30,30),
        marker=:square,
        markersize=5, alpha=0.5,
        color=:green
    )
    Plots.scatter!(plt, Darvishyadegari_Hassanzadeh_RS, Darvishyadegari_Hassanzadeh_CD2[i, :], label=lbl2, marker=:square, markersize=5, alpha=0.5, markerstrokewidth=1.5, color=:red)
end
for (i, GD) in enumerate(Rastan_GD)
    Plots.scatter!(plt, Rastan_RS, Rastan_CD1[i, :], marker=:square, markersize=5, alpha=0.5, markerstrokewidth=1.5, color=:green, label=false)
    Plots.scatter!(plt, Rastan_RS, Rastan_CD2[i, :], marker=:square, markersize=5, alpha=0.5, markerstrokewidth=1.5, color=:red, label=false)
end

for (i, GD) in enumerate(Siddiqui_GD)
    Plots.scatter!(plt, Siddiqui_RS, Siddiqui_CD1[i, :], marker=:square, markersize=5, alpha=0.5, markerstrokewidth=1.5, color=:green, label=false)
    Plots.scatter!(plt, Siddiqui_RS, Siddiqui_CD2[i, :], marker=:square, markersize=5, alpha=0.5, markerstrokewidth=1.5, color=:red, label=false)
end

lbl7 = L"C_{D_{1}}" * ", Present"
lbl8 = L"C_{D_{2}}" * ", Present"
Plots.scatter!(plt, [3.0], Cx1_3_3_2, label=lbl7, color=:magenta, marker=:circle, markersize=7, markerstrokewidth=1.5)
Plots.scatter!(plt, [3.0], Cx2_3_3_2, label=lbl8, color=:cyan, marker=:circle, markersize=7, markerstrokewidth=1.5)
Plots.scatter!(plt, [4.0], Cx1_4_4_2, label=false, color=:magenta, marker=:circle, markersize=7, markerstrokewidth=1.5)
Plots.scatter!(plt, [4.0], Cx2_4_4_2, label=false, color=:cyan, marker=:circle, markersize=7, markerstrokewidth=1.5)
Plots.scatter!(plt, [5.0], Cx1_5_5_2, label=false, color=:magenta, marker=:circle, markersize=7, markerstrokewidth=1.5)
Plots.scatter!(plt, [5.0], Cx2_5_5_2, label=false, color=:cyan, marker=:circle, markersize=7, markerstrokewidth=1.5)


savefig(plt, joinpath(outdir, "forces.pdf"))

# view_forces!(72,3,3,0;stats_init=600,dir=datadir)
# view_forces!(72,3,4,0;stats_init=625,dir=datadir)
# view_forces!(72,3,5,0;stats_init=600,dir=datadir)
# view_forces!(72,4,3,0;stats_init=600,dir=datadir)
# view_forces!(72,4,4,0;stats_init=600,dir=datadir)
# view_forces!(72,4,5,0;stats_init=425,dir=datadir)
# view_forces!(72,5,3,0;stats_init=400,dir=datadir)
# view_forces!(72,5,4,0;stats_init=460,dir=datadir)
# view_forces!(72,5,5,0;stats_init=400,dir=datadir)

# view_forces!(72,3,3,15;stats_init=300,dir=datadir)
# view_forces!(72,3,4,15;stats_init=300,dir=datadir)
# view_forces!(72,3,5,15;stats_init=300,dir=datadir)
# view_forces!(72,4,3,15;stats_init=300,dir=datadir)
# view_forces!(72,4,4,15;stats_init=300,dir=datadir)
# view_forces!(72,4,5,15;stats_init=300,dir=datadir)
# view_forces!(72,5,3,15;stats_init=300,dir=datadir)
# view_forces!(72,5,4,15;stats_init=300,dir=datadir)
# view_forces!(72,5,5,15;stats_init=300,dir=datadir)

# view_forces!(72,3,3,30;stats_init=300,dir=datadir)
# view_forces!(72,3,4,30;stats_init=300,dir=datadir)
# view_forces!(72,3,5,30;stats_init=300,dir=datadir)
# view_forces!(72,4,3,30;stats_init=300,dir=datadir)
# view_forces!(72,4,4,30;stats_init=300,dir=datadir)
# view_forces!(72,4,5,30;stats_init=300,dir=datadir)
# view_forces!(72,5,3,30;stats_init=300,dir=datadir)
# view_forces!(72,5,4,30;stats_init=300,dir=datadir)
# view_forces!(72,5,5,30;stats_init=300,dir=datadir)

# # choose your (λ1, λ2, θ) cases once
# cases = [(3,3,0),(3,4,0),(3,5,0),
#          (4,3,0),(4,4,0),(4,5,0),
#          (5,3,0),(5,4,0),(5,5,0)]

# # Gather data for each cylinder
# x1 = Float64[]; y1 = Float64[]; lab1 = LaTeXString[]
# x2 = Float64[]; y2 = Float64[]; lab2 = LaTeXString[]

# for (λ1,λ2,θ) in cases
#     m = meanforces[(72,λ1,λ2,θ)]
#     # First cylinder plotted in magenta (you negated in your example)
#     push!(x1, m.C̄x₁); push!(y1, m.C̄y₁)
#     push!(lab1, latexstring("\\qquad\\mathbf{($(λ1),$(λ2))}"))
#     # Second cylinder plotted in cyan
#     push!(x2, abs(m.C̄x₂)); push!(y2, m.C̄y₂)
#     push!(lab2, latexstring("\\mathbf{($(λ1),$(λ2))}\\qquad"))
# end

# plt2 = Plots.plot(
#     xlabel = L"\overline{C}_{x}",
#     ylabel = L"\overline{C}_{y}",
#     title = L"\theta=0\degree",
#     grid = true, size = (800,600), framestyle = :box,
#     legend = :bottomright, legend_columns = 2,
#     labelfontsize = 12, tickfontsize = 12, legendfontsize = 12, titlefontsize=12,
#     xgrid=true, ygrid=true, xminorgrid=true, yminorgrid=true,
#     gridalpha=0.2, minorgridalpha=0.2,
#     gridcolor=:black, minorgridcolor=:black,
#     gridlinewidth=0.5, minorgridlinewidth=0.5
# )

# # Magenta: cylinder 1, with per-point annotations placed on top
# Plots.scatter!(plt2, x1, y1;
#     label = L"C_{F_{1}}", color=:magenta, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab1, 10, :left, :left, :black) # bottom of text sits on the point → text above marker
# )

# # Cyan: cylinder 2, with per-point annotations placed on top
# Plots.scatter!(plt2, x2, y2;
#     label = L"C_{F_{2}}", color=:cyan, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab2, 10, :left, :right, :black)
# )
# savefig(plt2, joinpath(outdir, "force_pairs_0.pdf"))

# # choose your (λ1, λ2, θ) cases once
# cases = [(3,3,15),(3,4,15),(3,5,15),
#          (4,3,15),(4,4,15),(4,5,15),
#          (5,3,15),(5,4,15),(5,5,15)]

# # Gather data for each cylinder
# x1 = Float64[]; y1 = Float64[]; lab1 = LaTeXString[]
# x2 = Float64[]; y2 = Float64[]; lab2 = LaTeXString[]

# for (λ1,λ2,θ) in cases
#     m = meanforces[(72,λ1,λ2,θ)]
#     # First cylinder plotted in magenta (you negated in your example)
#     push!(x1, m.C̄x₁); push!(y1, m.C̄y₁)
#     push!(lab1, latexstring("\\quad\\mathbf{($(λ1),$(λ2))}"))
#     # Second cylinder plotted in cyan
#     push!(x2, abs(m.C̄x₂)); push!(y2, m.C̄y₂)
#     push!(lab2, latexstring("\\mathbf{($(λ1),$(λ2))}\\quad"))
# end

# plt3 = Plots.plot(
#     xlabel = L"\overline{C}_{x}",
#     ylabel = L"\overline{C}_{y}",
#     title = L"\theta=15\degree",
#     grid = true, size = (800,600), framestyle = :box,
#     legend = :bottomright, legend_columns = 2,
#     labelfontsize = 12, tickfontsize = 12, legendfontsize = 12, titlefontsize=12,
#     xgrid=true, ygrid=true, xminorgrid=true, yminorgrid=true,
#     gridalpha=0.2, minorgridalpha=0.2,
#     gridcolor=:black, minorgridcolor=:black,
#     gridlinewidth=0.5, minorgridlinewidth=0.5
# )

# # Magenta: cylinder 1, with per-point annotations placed on top
# Plots.scatter!(plt3, x1, y1;
#     label = L"C_{F_{1}}", color=:magenta, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab1, 10, :bottom, :left, :black) # bottom of text sits on the point → text above marker
# )

# # Cyan: cylinder 2, with per-point annotations placed on top
# Plots.scatter!(plt3, x2, y2;
#     label = L"C_{F_{2}}", color=:cyan, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab2, 10, :bottom, :right, :black)
# )
# savefig(plt3, joinpath(outdir, "force_pairs_15.pdf"))

# # choose your (λ1, λ2, θ) cases once
# cases = [(3,3,30),(3,4,30),(3,5,30),
#          (4,3,30),(4,4,30),(4,5,30),
#          (5,3,30),(5,4,30),(5,5,30)]

# # Gather data for each cylinder
# x1 = Float64[]; y1 = Float64[]; lab1 = LaTeXString[]
# x2 = Float64[]; y2 = Float64[]; lab2 = LaTeXString[]

# for (λ1,λ2,θ) in cases
#     m = meanforces[(72,λ1,λ2,θ)]
#     # First cylinder plotted in magenta (you negated in your example)
#     push!(x1, m.C̄x₁); push!(y1, m.C̄y₁)
#     push!(lab1, latexstring("\\quad\\mathbf{($(λ1),$(λ2))}"))
#     # Second cylinder plotted in cyan
#     push!(x2, abs(m.C̄x₂)); push!(y2, m.C̄y₂)
#     push!(lab2, latexstring("\\mathbf{($(λ1),$(λ2))}\\quad"))
# end

# plt4 = Plots.plot(
#     xlabel = L"\overline{C}_{x}",
#     ylabel = L"\overline{C}_{y}",
#     title = L"\theta=30\degree",
#     grid = true, size = (800,600), framestyle = :box,
#     legend = :bottomright, legend_columns = 2,
#     labelfontsize = 12, tickfontsize = 12, legendfontsize = 12, titlefontsize=12,
#     xgrid=true, ygrid=true, xminorgrid=true, yminorgrid=true,
#     gridalpha=0.2, minorgridalpha=0.2,
#     gridcolor=:black, minorgridcolor=:black,
#     gridlinewidth=0.5, minorgridlinewidth=0.5
# )

# # Magenta: cylinder 1, with per-point annotations placed on top
# Plots.scatter!(plt4, x1, y1;
#     label = L"C_{F_{1}}", color=:magenta, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab1, 10, :bottom, :left, :black) # bottom of text sits on the point → text above marker
# )

# # Cyan: cylinder 2, with per-point annotations placed on top
# Plots.scatter!(plt4, x2, y2;
#     label = L"C_{F_{2}}", color=:cyan, marker=:circle, markersize=5, markerstrokewidth=1,
#     series_annotations = Plots.text.(lab2, 10, :bottom, :right, :black)
# )
# savefig(plt4, joinpath(outdir, "force_pairs_30.pdf"))

# function efficiency_surface_rotor1(meanforces, D::Int,
#                                    λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector;
#                                    outdir::AbstractString, fname::AbstractString)
#     mkpath(outdir)

#     # helper: build Z(λ1, λ2) = |Cy1/Cx1| for a given θ
#     function build_Z(θv)
#         Z = fill(Float64(NaN), length(λ1s), length(λ2s))  # rows = λ1, cols = λ2
#         for (i, λ1) in enumerate(λ1s), (j, λ2) in enumerate(λ2s)
#             key = (D, λ1, λ2, θv)
#             if haskey(meanforces, key)
#                 m = meanforces[key]
#                 cx = m.C̄x₁; cy = m.C̄y₁
#                 if isfinite(cx) && isfinite(cy) && cx != 0
#                     Z[i, j] = abs(cy / cx)
#                 end
#             end
#         end
#         return Z
#     end

#     for θv in θs
#         Z = build_Z(θv)

#         fig = Figure(size = (1000, 700))
#         ax = Axis3(fig[1, 1];
#             title  = latexstring("\$\\theta = $θv\\degree\$"),
#             titlesize = 22,
#             xlabel = L"\lambda_{1}",  ylabel = L"\lambda_{2}",
#             zlabel = L"\eta_{1}=\left|\overline{C}_{y_{1}}/\overline{C}_{x_{1}}\right|",
#             xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
#             xticklabelsize = 14, yticklabelsize = 14, zticklabelsize = 14,
#             azimuth = π/4
#         )

#         c_eff = cgrad(:algae, alpha = 1)
#         surf = CairoMakie.surface!(ax, λ1s, λ2s, Z; colormap = c_eff, shading = true, transparency = false)
#         CairoMakie.wireframe!(ax, λ1s, λ2s, Z; color = :green, linewidth = 0.5)

#         # overlay the actual data points (grid points) in your usual style
#         x_pts = repeat(λ1s, outer = length(λ2s))
#         y_pts = repeat(λ2s, inner = length(λ1s))
#         z_pts = vec(Z)
#         CairoMakie.scatter!(ax, x_pts, y_pts, z_pts;
#             marker = :circle, markersize = 10, color = :green,
#             strokecolor = :black, strokewidth = 0.5)

#         ax.xticks = (λ1s, string.(λ1s))
#         ax.yticks = (λ2s, string.(λ2s))
#         hidespines!(ax)

#         # optional colorbar (remove if you want to match exactly without a bar)
#         Colorbar(fig[1, 2], surf, label = L"\eta_{1}", labelsize = 16, ticklabelsize = 12)

#         save(joinpath(outdir, string(fname, θv, ".pdf")), fig)
#     end
# end

# function efficiency_surface_rotor2(meanforces, D::Int,
#                                    λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector;
#                                    outdir::AbstractString, fname::AbstractString)
#     mkpath(outdir)

#     # helper: build Z(λ1, λ2) = |Cy1/Cx1| for a given θ
#     function build_Z(θv)
#         Z = fill(Float64(NaN), length(λ1s), length(λ2s))  # rows = λ1, cols = λ2
#         for (i, λ1) in enumerate(λ1s), (j, λ2) in enumerate(λ2s)
#             key = (D, λ1, λ2, θv)
#             if haskey(meanforces, key)
#                 m = meanforces[key]
#                 cx = m.C̄x₂; cy = m.C̄y₂
#                 if isfinite(cx) && isfinite(cy) && cx != 0
#                     Z[i, j] = abs(cy / cx)
#                 end
#             end
#         end
#         return Z
#     end

#     for θv in θs
#         Z = build_Z(θv)

#         fig = Figure(size = (1000, 700))
#         ax = Axis3(fig[1, 1];
#             title  = latexstring("\$\\theta = $θv\\degree\$"),
#             titlesize = 22,
#             xlabel = L"\lambda_{1}",  ylabel = L"\lambda_{2}",
#             zlabel = L"\eta_{2}=\left|\overline{C}_{y_{2}}/\overline{C}_{x_{2}}\right|",
#             xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
#             xticklabelsize = 14, yticklabelsize = 14, zticklabelsize = 14,
#             azimuth = π/4
#         )

#         c_eff = cgrad(:algae, alpha = 1)
#         surf = CairoMakie.surface!(ax, λ1s, λ2s, Z; colormap = c_eff, shading = true, transparency = false)
#         CairoMakie.wireframe!(ax, λ1s, λ2s, Z; color = :green, linewidth = 0.5)

#         # overlay the actual data points (grid points) in your usual style
#         x_pts = repeat(λ1s, outer = length(λ2s))
#         y_pts = repeat(λ2s, inner = length(λ1s))
#         z_pts = vec(Z)
#         CairoMakie.scatter!(ax, x_pts, y_pts, z_pts;
#             marker = :circle, markersize = 10, color = :green,
#             strokecolor = :black, strokewidth = 0.5)

#         ax.xticks = (λ1s, string.(λ1s))
#         ax.yticks = (λ2s, string.(λ2s))
#         hidespines!(ax)

#         # optional colorbar (remove if you want to match exactly without a bar)
#         Colorbar(fig[1, 2], surf, label = L"\eta_{2}", labelsize = 16, ticklabelsize = 12)

#         save(joinpath(outdir, string(fname, θv, ".pdf")), fig)
#     end
# end

# function efficiency_surface_tot(meanforces, D::Int,
#                                    λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector;
#                                    outdir::AbstractString, fname::AbstractString)
#     mkpath(outdir)

#     # helper: build Z(λ1, λ2) = |Cy1/Cx1| for a given θ
#     function build_Z(θv)
#         Z = fill(Float64(NaN), length(λ1s), length(λ2s))  # rows = λ1, cols = λ2
#         for (i, λ1) in enumerate(λ1s), (j, λ2) in enumerate(λ2s)
#             key = (D, λ1, λ2, θv)
#             if haskey(meanforces, key)
#                 m = meanforces[key]
#                 cx1 = m.C̄x₁; cy1 = m.C̄y₁; cx2 = m.C̄x₂; cy2 = m.C̄y₂
#                 if isfinite(cx1) && isfinite(cy1) && isfinite(cx2) && isfinite(cy2) && (cx1+cx2) != 0
#                     Z[i, j] = abs((cy1+cy2) / (cx1+cx2))
#                 end
#             end
#         end
#         return Z
#     end

#     for θv in θs
#         Z = build_Z(θv)

#         fig = Figure(size = (1000, 700))
#         ax = Axis3(fig[1, 1];
#             title  = latexstring("\$\\theta = $θv\\degree\$"),
#             titlesize = 22,
#             xlabel = L"\lambda_{1}",  ylabel = L"\lambda_{2}",
#             zlabel = L"\eta=\left|\overline{C}_{y_{tot.}}/\overline{C}_{x_{tot.}}\right|",
#             xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
#             xticklabelsize = 14, yticklabelsize = 14, zticklabelsize = 14,
#             azimuth = π/4
#         )

#         c_eff = cgrad(:algae, alpha = 1)
#         surf = CairoMakie.surface!(ax, λ1s, λ2s, Z; colormap = c_eff, shading = true, transparency = false)
#         CairoMakie.wireframe!(ax, λ1s, λ2s, Z; color = :green, linewidth = 0.5)

#         # overlay the actual data points (grid points) in your usual style
#         x_pts = repeat(λ1s, outer = length(λ2s))
#         y_pts = repeat(λ2s, inner = length(λ1s))
#         z_pts = vec(Z)
#         CairoMakie.scatter!(ax, x_pts, y_pts, z_pts;
#             marker = :circle, markersize = 10, color = :green,
#             strokecolor = :black, strokewidth = 0.5)

#         ax.xticks = (λ1s, string.(λ1s))
#         ax.yticks = (λ2s, string.(λ2s))
#         hidespines!(ax)

#         # optional colorbar (remove if you want to match exactly without a bar)
#         Colorbar(fig[1, 2], surf, label = L"\eta", labelsize = 16, ticklabelsize = 12)

#         save(joinpath(outdir, string(fname, θv, ".pdf")), fig)
#     end
# end

# efficiency_surface_rotor1(meanforces, 72, [3,4,5], [3,4,5], [0];
#     outdir = outdir, fname = "efficiency_rotor1_")
# efficiency_surface_rotor1(meanforces, 72, [3,4,5], [3,4,5], [15];
#     outdir = outdir, fname = "efficiency_rotor1_")
# efficiency_surface_rotor1(meanforces, 72, [3,4,5], [3,4,5], [30];
#     outdir = outdir, fname = "efficiency_rotor1_")
# efficiency_surface_rotor2(meanforces, 72, [3,4,5], [3,4,5], [0];
#     outdir = outdir, fname = "efficiency_rotor2_")
# efficiency_surface_rotor2(meanforces, 72, [3,4,5], [3,4,5], [15];
#     outdir = outdir, fname = "efficiency_rotor2_")
# efficiency_surface_rotor2(meanforces, 72, [3,4,5], [3,4,5], [30];
#     outdir = outdir, fname = "efficiency_rotor2_")
# efficiency_surface_tot(meanforces, 72, [3,4,5], [3,4,5], [0];
#     outdir = outdir, fname = "efficiency_rotor_tot_")
# efficiency_surface_tot(meanforces, 72, [3,4,5], [3,4,5], [15];
#     outdir = outdir, fname = "efficiency_rotor_tot_")
# efficiency_surface_tot(meanforces, 72, [3,4,5], [3,4,5], [30];
#     outdir = outdir, fname = "efficiency_rotor_tot_")