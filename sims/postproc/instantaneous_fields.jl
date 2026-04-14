include("../../setup/3D.jl")
include("../../help/MeanFlow_PostProc.jl")
include("../../help/ThreeD_Plots.jl")
using CairoMakie
GLMakie.activate!()
_symmetry = false
_free_slip = false
_full = true
_validation_rot = false
_rotor = false
_rec = false
_mean = false
λ1r = 3; λ2r = 3; θr = 15

function read_recovered_fields(fname::AbstractString; dir::AbstractString="")
    jldopen(joinpath(dir, fname)) do f
        return f["P_rec"], f["U_rec"]
    end
end

if _symmetry
    import BiotSavartBCs: interaction, symmetry, image
    @inline function symmetry(ω, T, args...)
        T₃, sgn₃ = image(T, size(ω), -3)
        return interaction(ω, T, args...) + sgn₃ * interaction(ω, T₃, args...)
    end
    D = 72; L = (12,5,6); Re=500; λ₁ = 3; θ = 0; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁ = ThreeD_Rotor_Validation_half(D, λ₁, U₊, Array; L, Re, T)

    datadir = "../Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation_Validation_Symmetry/"
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_symmetry.jld2"

    Qiso = 1f-3
    s = (1100, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "validation_symmetry_Q_$(Qiso).png")

    WaterLily.load!(sim.flow; fname=flow_file, dir=datadir)

elseif _free_slip
    D = 72; L = (12,5,6); Re=500; λ₁ = 3; θ = 0; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁ = ThreeD_Rotor_Validation_half(D, λ₁, U₊, Array; L, Re, T)
    
    datadir = "../Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation_Validation_Free_Slip/"
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_free_slip.jld2"
    
    Qiso = 1f-3
    s = (1100, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "validation_free_slip_Q_$(Qiso).png")

    WaterLily.load!(sim.flow; fname=flow_file, dir=datadir)
elseif _full
    D = 72; L = (12,5,12); Re=500; λ₁ = 3; θ = 0; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁ = ThreeD_Rotor_Validation(D, λ₁, U₊, Array; L, Re, T)
    
    datadir = "../Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation_Validation_Full/"
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_full.jld2"

    Qiso = 1f-3
    s = (800, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "validation_full_Q_$(Qiso).png")

    WaterLily.load!(sim.flow; fname=flow_file, dir=datadir)

elseif _validation_rot
    D = 72; L = (10,10,3); Re=1000; λ₁ = 3; T=Float32
    sim = ThreeD_Cylinder_wPlate_wRot_wBL(D, λ₁, Array; L, Re, T)

    datadir = "../Validation/sims/data/rotating_cylinder/"
    flow_file = "flow_rotating_cylinder_$(D)_$(λ₁).jld2"

    Qiso = 1f0
    s = (1600, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "validation_rot_$(Qiso).png")

    WaterLily.load!(sim.flow; fname=flow_file, dir=datadir)

elseif _rotor
    D = 72; L = (8,5,8); Re=1000; λ₁=λ1r; λ₂=λ2r; θ=θr ; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, Array; L, Re, T)

    datadir = "../Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation/"
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
    
    Qiso = 1f-2
    s = (1100, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "rotor_free_slip_Q_$(Qiso)_$(λ₁)_$(θ).png")

    WaterLily.load!(sim.flow; fname=flow_file, dir=datadir)

elseif _rec
    D = 72; L = (8,5,8); Re=1000; λ₁=λ1r; λ₂=λ2r; θ=θr ; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, Array; L, Re, T)

    datadir = "../Rotor_BiotSavartBCs/ml/data/rotor_BiotSimulation/recovered/"
    flow_file = "rec_fields_B_$(λ₁)_$(λ₂)_$(θ).jld2"
    P, U = read_recovered_fields(flow_file; dir=datadir)

    Qiso = 1f-2
    s = (1100, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "rotor_free_slip_Q_rec_$(Qiso)_$(λ₁)_$(λ₂)_$(θ).png")

    sim.flow.u .= U

elseif _mean
    D = 72; L = (8,5,8); Re=1000; λ₁=λ1r; λ₂=λ2r; θ=θr ; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, Array; L, Re, T)

    datadir = "../Rotor_BiotSavartBCs/sims/data/rotor_BiotSimulation/"
    flow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
    P, U, _ = read_meanflow(flow_file; dir=datadir, stats=true)

    Qiso = 1f-2
    s = (1100, 800)
    savedir = "../Rotor_BiotSavartBCs/sims/postproc/tex/instantaneous_fields/"
    file = joinpath(savedir, "rotor_free_slip_Q_avg_$(Qiso)_$(λ₁)_$(λ₂)_$(θ).png")

    sim.flow.u .= U
else
    error("No simulation mode selected!")
end

vertsM, facesM, colors = Q_iso_vertices_ω₃(sim;Qiso=Qiso)

@show size(vertsM) size(facesM) size(colors)

fig = Figure(size = s)
ax  = Axis3(fig[1, 1],
            xlabel = "x",
            ylabel = "y",
            zlabel = "z",
            aspect = :data,
            xlabelsize = 26,
            ylabelsize = 26,
            zlabelsize = 26,
            xticklabelsize = 18,
            yticklabelsize = 18,
            zticklabelsize = 18,)

crange = (-1,1)   # (min, max)


plt = mesh!(ax, vertsM, facesM;
            color      = colors,
            colormap   = :jet1,   # or :vik, :coolwarm, etc.
            colorrange = crange,
            shading    = false)


Colorbar(fig[1, 2], plt; label = L"\omega_z", labelsize = 26, ticklabelsize = 18)
CairoMakie.save(file, fig)
fig


# function viz_λ₂(sim; cmap = :plasma, crange = nothing)
#     λ2 = flow_λ₂(sim)            # 3D array on inside region
#     plt = CairoMakie.contour(λ2, levels=[-5,5],alpha=0.5)
#     return plt
# end
# plt = viz_λ₂(sim)
# save("lambda2_volume_$(D)_$(λ₁)_$(λ₂)_$(θ).png", plt)











# function phys_crds(sim,vts,fcs) 
#     a = sim.flow.σ; R = inside(a); D = sim.L
#     s = size(a)
#     r = size(R)
#     Lx, Ly, Lz = s[1], s[2], s[3]
#     nx, ny, nz = r[1], r[2], r[3]                  
#     dx = Lx / nx
#     dy = Ly / ny
#     dz = Lz / nz

#     nv = length(vts)
#     vertsM = Array{Float32}(undef, nv, 3)
#     for (n, (i, j, k)) in enumerate(vts)
#         # cell-center mapping: (i-0.5)*Δx etc.
#         vertsM[n, 1] = (i - 0.5f0) * dx
#         vertsM[n, 2] = (j - 0.5f0) * dy
#         vertsM[n, 3] = (k - 0.5f0) * dz
#     end

#     # --- Faces as (M,3) matrix for Makie ---
#     nf = length(fcs)
#     facesM = Array{Int}(undef, nf, 3)
#     for (m, (a₁, a₂, a₃)) in enumerate(fcs)
#         facesM[m, 1] = a₁
#         facesM[m, 2] = a₂
#         facesM[m, 3] = a₃
#     end
#     return vertsM, facesM
# end


