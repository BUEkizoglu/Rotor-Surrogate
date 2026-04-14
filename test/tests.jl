using CUDA
include("../setup/3D.jl")
include("../test/forces_pressure.jl")
include("../test/sdf.jl")
include("../help/MeanFlow_PostProc.jl")
include("../help/TwoD_Plots.jl")

# Comment in for symmetry BC 
# import BiotSavartBCs: interaction, symmetry, image
# @inline function symmetry(ω, T, args...)
#     T₃, sgn₃ = image(T, size(ω), -3)
#     return interaction(ω, T, args...) + sgn₃ * interaction(ω, T₃, args...)
# end

D = 48
backend = CUDA.CuArray
L = (8,5,8)
Re = 1000
T = Float32
θ = 0
U = 1
U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
ν = U*D/Re
λ₁ = 0
x,y,z = Int[2D],Int[2.5D],Int[3.0D]
slice_z_idx = Int(3.0D)
duration = 1
tstep = 0.1
sim,Rotor₁ = ThreeD_Rotor_Validation_Free_Slip(D, λ₁, U₊, backend; L, Re, T)
check_sdf_field!(sim, D, x, y, z)
log_pressure_sim_single(sim,Rotor₁;D, λ=λ₁, θ, slice_z_idx, duration, tstep, vmin=-5, vmax=5)

# U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
# println("Running single rotor: D=$(D), λ=$(λ₁), θ=$(θ)")
# vmin, vmax = -5, 5
# sim,Rotor₁ = ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, U₊, backend; L, Re, T)
# log_pressure_sim_single(sim,Rotor₁;D, λ=λ₁, θ, slice_z_idx, duration, tstep, vmin, vmax)

# function main()
#     for λ₁ in λs
#         for λ₂ in λs
#             for θ in θs
#                 U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
#                 println("Running single rotor: D=$(D), λ=$(λ₁), θ=$(θ)")
#                 vmin, vmax = -5, 5
#                 sim,Rotor₁ = ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, U₊, backend; L, Re, T)
#                 log_pressure_sim_single(sim,Rotor₁;D, λ=λ₁, θ, slice_z_idx, duration, tstep, vmin, vmax)

#                 # println("Running double rotor: D=$(D), λ₁=$(λ₁), λ₂=$(λ₂) θ=$(θ)")
#                 # vmin, vmax = -5, 5
#                 # sim,Rotor₁,Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, backend; L, Re, T)
#                 # log_pressure_sim_double(sim,Rotor₁,Rotor₂;D, λ₁, λ₂, θ, slice_z_idx, duration, tstep, vmin, vmax)
#             end
#         end
#     end
# end
# main()

# sim,_ = ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, λ₂, U₊, backend; L, Re, T)
# sim_step!(sim)
# println("Ran one step!")
# check_sdf_field!(sim,D,x,y,z)

# Create a video using Makie
# dat = sim.flow.σ[inside(sim.flow.σ)] |> Array; # CPU buffer array
# function λ₂!(dat,sim)                          # compute log10(-λ₂)
#     a = sim.flow.σ
#     @inside a[I] = log10(max(1e-6,-WaterLily.λ₂(I,sim.flow.u)*sim.L/sim.U))
#     copyto!(dat,a[inside(a)])                  # copy to CPU
# end
# @time makie_video!(sim,dat,λ₂!,name="../test/Rotor.mp4",duration=10) do obs
#     GLMakie.contour(obs,levels=[-3,-2,-1,0],alpha=0.1)
# end
