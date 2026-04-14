import BiotSavartBCs: interaction, symmetry, image
include("../../help/Forces_PostProc.jl")

T = Float32
D = 72
dz = 1/D
savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/"
z_vals_full = 0:dz:12
z_vals_half = 0:dz:6
x_force_full, y_force_full = sectional_forces_validation(z_vals_full; _full=true)  
Cx_full = x_force_full./(0.5*(D))
Cy_full = y_force_full./(0.5*(D))

x_force_free_slip, y_force_free_slip = sectional_forces_validation(z_vals_half; _free_slip=true)  
Cx_free_slip = x_force_free_slip./(0.5*(D))
Cy_free_slip = y_force_free_slip./(0.5*(D))

x_force_sym, y_force_sym = sectional_forces_validation(z_vals_half; _symmetry=true)  
Cx_sym = x_force_sym./(0.5*(D))
Cy_sym = y_force_sym./(0.5*(D))

z_rotor=0:dz:8
λ1 = 5; λ2 = 5; θ = 0
x_force1, y_force1, x_force2, y_force2 = sectional_forces_rotor(D, λ1, λ2, θ, z_rotor) 
Cx1 = x_force1./(0.5*D); Cy1 = y_force1./(0.5*D); Cx2 = x_force2./(0.5*D); Cy2 = y_force2./(0.5*D)

plt_full = Plots.plot(
    xlabel = L"z/D",
    ylabel = L"\overline{C}_{D_{s},L_{s}}",
    xlims = (1,11),
    ylims = (0,10),
    yticks = 0:0.5:10,
    legend = :topright,
    framestyle = :box,
    tickfontsize = 12,
    legendfontsize = 12,
    grid = true,
    size = (600, 600)
)
Plots.plot!(plt_full, z_vals_full, Cx_full; label = L"\overline{C}_{D_{s}}", lw=2, color=:red)
Plots.plot!(plt_full, z_vals_full, Cy_full; label = L"\overline{C}_{L_{s}}", lw=2, color=:blue)
savefig(plt_full, joinpath(savedir, "Cxy_s_validation_full.pdf"))

plt_free_slip_symmetry = Plots.plot(
    xlabel = L"z/D",
    ylabel = L"\overline{C}_{D_{s},L_{s}}",
    xlims = (0.05,5),
    ylims = (0,10),
    yticks = 0:0.5:10,
    legend = :topright,
    framestyle = :box,
    tickfontsize = 12,
    legendfontsize = 12,
    grid = true,
    size = (600, 600)
)
Plots.plot!(plt_free_slip_symmetry, z_vals_half, Cx_free_slip; label = latexstring("\$\\overline{C}_{D_{s}}\$ (free slip)"), lw=2, color=:red)
Plots.plot!(plt_free_slip_symmetry, z_vals_half, Cy_free_slip; label = latexstring("\$\\overline{C}_{L_{s}}\$ (free slip)"), lw=2, color=:blue)
Plots.plot!(plt_free_slip_symmetry, z_vals_half, Cx_sym; label = latexstring("\$\\overline{C}_{D_{s}}\$ (symmetry)"), lw=2, color=:red, linestyle=:dash)
Plots.plot!(plt_free_slip_symmetry, z_vals_half, Cy_sym; label = latexstring("\$\\overline{C}_{L_{s}}\$ (symmetry)"), lw=2, color=:blue, linestyle=:dash)
savefig(plt_free_slip_symmetry, joinpath(savedir, "Cxy_s_validation_slip_sym.pdf"))

