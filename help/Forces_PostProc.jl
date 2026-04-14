using StaticArrays
include("../setup/2D.jl")
include("../help/MeanFlow_PostProc.jl")

function decompose_forces(f)
    fd = mapreduce(permutedims,vcat,f)
    fx, fy, fz = -fd[:,1], -fd[:,2], -fd[:,3]
    return fx,fy,fz
end

function decompose_forces_2d(f)
    fd = mapreduce(permutedims,vcat,f)
    fx, fy = -fd[:,1], -fd[:,2]
    return fx,fy
end

function get_forces_and_coefficients(p_force₁, v_force₁, p_force₂, v_force₂, D)
    t_force₁ = p_force₁ .+ v_force₁
    t_force₂ = p_force₂ .+ v_force₂
    Ft_x₁, Ft_y₁, Ft_z₁ = decompose_forces(t_force₁)
    Ft_x₂, Ft_y₂, Ft_z₂ = decompose_forces(t_force₂)
    Fx₁ = Ft_x₁.*(0.5*(D^2))
    Fy₁ = Ft_y₁.*(0.5*(D^2))
    Fz₁ = Ft_z₁.*(0.5*(D^2))
    Fx₂ = Ft_x₂.*(0.5*(D^2))
    Fy₂ = Ft_y₂.*(0.5*(D^2))
    Fz₂ = Ft_z₂.*(0.5*(D^2))
    A = ((35*π*D^2)/4) + (2*π*(2+√2))
    Cx₁ = (Fx₁)/((0.5*A))
    Cy₁ = (Fy₁)/((0.5*A))
    Cz₁ = (Fz₁)/((0.5*A))
    Cx₂ = (Fx₂)/((0.5*A))
    Cy₂ = (Fy₂)/((0.5*A))
    Cz₂ = (Fz₂)/((0.5*A))
    return (; Fx₁, Fy₁, Fz₁, Fx₂, Fy₂, Fz₂, Cx₁, Cy₁, Cz₁, Cx₂, Cy₂, Cz₂)
end

function get_forces_and_coefficients_2d(p_force₁, v_force₁, p_force₂, v_force₂, D)
    t_force₁ = p_force₁ .+ v_force₁
    t_force₂ = p_force₂ .+ v_force₂
    Ft_x₁, Ft_y₁, Ft_z₁ = decompose_forces(t_force₁)
    Ft_x₂, Ft_y₂, Ft_z₂ = decompose_forces(t_force₂)
    Fx₁ = Ft_x₁.*(0.5*(D^2))
    Fy₁ = Ft_y₁.*(0.5*(D^2))
    Fz₁ = Ft_z₁.*(0.5*(D^2))
    Fx₂ = Ft_x₂.*(0.5*(D^2))
    Fy₂ = Ft_y₂.*(0.5*(D^2))
    Fz₂ = Ft_z₂.*(0.5*(D^2))
    Cx₁ = (Fx₁)/((0.5*7*D^2))
    Cy₁ = (Fy₁)/((0.5*7*D^2))
    Cz₁ = (Fz₁)/((0.5*7*D^2))
    Cx₂ = (Fx₂)/((0.5*7*D^2))
    Cy₂ = (Fy₂)/((0.5*7*D^2))
    Cz₂ = (Fz₂)/((0.5*7*D^2))
    return (; Fx₁, Fy₁, Fz₁, Fx₂, Fy₂, Fz₂, Cx₁, Cy₁, Cz₁, Cx₂, Cy₂, Cz₂)
end

"""
Return mean forces (Fx̄,Fȳ) for each cylinder computed from stats_init onward.
Inputs:
  t_full         :: AbstractVector       # time in CTU
  p_force₁, v_force₁ :: AbstractMatrix   # size (Nt, 3) or (3, Nt); see getcol
  p_force₂, v_force₂ :: AbstractMatrix
  stats_init     :: Real                 # CTU
Keyword:
  dims = 1  # if your data is (Nt,3), keep dims=1; if (3,Nt), set dims=2
"""

function get_mean_forces_and_coefficients(forces::NamedTuple, t_full, stats_init)
    idx = t_full .>= stats_init
    @assert any(idx) "No samples at/after stats_init=$(stats_init)."
    F̄x₁ = mean(forces.Fx₁[idx]);  F̄x₂ = mean(forces.Fx₂[idx])
    F̄y₁ = mean(forces.Fy₁[idx]);  F̄y₂ = mean(forces.Fy₂[idx])
    F̄z₁ = mean(forces.Fz₁[idx]);  F̄z₂ = mean(forces.Fz₂[idx])
    C̄x₁ = mean(forces.Cx₁[idx]);  C̄x₂ = mean(forces.Cx₂[idx])
    C̄y₁ = mean(forces.Cy₁[idx]);  C̄y₂ = mean(forces.Cy₂[idx])
    C̄z₁ = mean(forces.Cz₁[idx]);  C̄z₂ = mean(forces.Cz₂[idx])
    return (; F̄x₁, F̄y₁, F̄z₁, F̄x₂, F̄y₂, F̄z₂, C̄x₁, C̄y₁, C̄z₁, C̄x₂, C̄y₂, C̄z₂)
end

function get_forces_and_coefficients_single(p_force₁, v_force₁, D)
    t_force₁ = p_force₁ .+ v_force₁
    Ft_x₁, Ft_y₁, Ft_z₁ = decompose_forces(t_force₁)
    Fx₁ = Ft_x₁.*(0.5*(D^2))
    Fy₁ = Ft_y₁.*(0.5*(D^2))
    Fz₁ = Ft_z₁.*(0.5*(D^2))
    A = ((35*π*D^2)/4) + (2*π*(2+√2))
    Cx₁ = (Fx₁)/((0.5*A))
    Cy₁ = (Fy₁)/((0.5*A))
    Cz₁ = (Fz₁)/((0.5*A))
    return (; Fx₁, Fy₁, Fz₁, Cx₁, Cy₁, Cz₁)
end

function get_forces_and_coefficients_single_2d(p_force₁, v_force₁, D)
    t_force₁ = p_force₁ .+ v_force₁
    Ft_x₁, Ft_y₁, Ft_z₁ = decompose_forces(t_force₁)
    Fx₁ = Ft_x₁.*(0.5*(D^2))
    Fy₁ = Ft_y₁.*(0.5*(D^2))
    Fz₁ = Ft_z₁.*(0.5*(D^2))
    Cx₁ = (Fx₁)/((0.5*7*D^2))
    Cy₁ = (Fy₁)/((0.5*7*D^2))
    Cz₁ = (Fz₁)/((0.5*7*D^2))
    return (; Fx₁, Fy₁, Fz₁, Cx₁, Cy₁, Cz₁)
end

function get_mean_forces_and_coefficients_single(forces::NamedTuple, t_full, stats_init)
    idx = t_full .>= stats_init
    @assert any(idx) "No samples at/after stats_init=$(stats_init)."
    F̄x₁ = mean(forces.Fx₁[idx])
    F̄y₁ = mean(forces.Fy₁[idx])
    F̄z₁ = mean(forces.Fz₁[idx])  
    C̄x₁ = mean(forces.Cx₁[idx])
    C̄y₁ = mean(forces.Cy₁[idx])
    C̄z₁ = mean(forces.Cz₁[idx])
    return (; F̄x₁, F̄y₁, F̄z₁, C̄x₁, C̄y₁, C̄z₁)
end

forces = Dict()
meanforces = Dict()
function view_forces!(D,λ₁,λ₂,θ;stats_init::Int,dir::String)
    p_force₁, p_force₂, v_force₁, v_force₂, u_probe_x_full, u_probe_y_full, u_probe_z_full, t_full = read_force_and_probe_vals("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; dir=dir)
    forces[(D,λ₁,λ₂,θ)] = get_forces_and_coefficients_2d(p_force₁, v_force₁, p_force₂, v_force₂, D)
    meanforces[(D,λ₁,λ₂,θ)] = get_mean_forces_and_coefficients(forces[(D,λ₁,λ₂,θ)], t_full, stats_init)
    return forces, meanforces 
end

function sectional_forces_validation(z_vals; _free_slip=false, _symmetry=false, _full=false)    
    if _symmetry
        @inline function symmetry(ω, T, args...)
            T₃, sgn₃ = image(T, size(ω), -3)
            return interaction(ω, T, args...) + sgn₃ * interaction(ω, T₃, args...)
        end
        D = 72; L = (12,5); Re=500; λ₁ = 3; θ = 0; T=Float32
        U₊ = (T(cosd(θ)), T(-sind(θ)))
        sim, Rotor₁ = Rotor_Validation_2D(D, λ₁, U₊, Array; L, Re, T)

        datadir = "/sims/data/rotor_BiotSimulation_Validation_Symmetry/"
        meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_symmetry.jld2"

        P, U, t = read_meanflow(meanflow_file;dir=datadir, stats = true, stats_turb = false)

    elseif _free_slip
        D = 72; L = (12,5); Re=500; λ₁ = 3; θ = 0; T=Float32
        U₊ = (T(cosd(θ)), T(-sind(θ)))
        sim, Rotor₁ = Rotor_Validation_2D(D, λ₁, U₊, Array; L, Re, T)

        datadir = "/sims/data/rotor_BiotSimulation_Validation_Free_Slip/"
        meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_free_slip.jld2"

        P, U, t = read_meanflow(meanflow_file;dir=datadir, stats = true, stats_turb = false)

    elseif _full
        D = 72; L = (12,5); Re=500; λ₁ = 3; θ = 0; T=Float32
        U₊ = (T(cosd(θ)), T(-sind(θ)))
        sim, Rotor₁ = Rotor_Validation_2D(D, λ₁, U₊, Array; L, Re, T)

        datadir = "/sims/data/rotor_BiotSimulation_Validation_Full/"
        meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_validation_full.jld2"

        P, U, t = read_meanflow(meanflow_file;dir=datadir, stats = true, stats_turb = false)
    else
        error("No simulation mode selected!")
    end
      
    x_force₁ = T[]
    y_force₁ = T[]
    for (i, z) in enumerate(z_vals)
        z_idx = (z*D)+1 |> ceil |> Int
        P_plane = P[:,:,z_idx]
        U_plane = U[:,:,z_idx,1:2]

        @assert size(sim.flow.u) == size(U_plane)
        @assert size(sim.flow.p) == size(P_plane)
        sim.flow.u .= U_plane
        sim.flow.p .= P_plane

        p_force₁ = WaterLily.pressure_force(sim.flow,Rotor₁)
        v_force₁ = WaterLily.viscous_force(sim.flow,Rotor₁)
        t_force₁ = p_force₁ .+ v_force₁
        fx₁, fy₁ = -t_force₁[1], -t_force₁[2]
        push!(x_force₁, fx₁)
        push!(y_force₁, fy₁)
    end
    return x_force₁, y_force₁
end

function sectional_forces_rotor(D, λ₁, λ₂, θ, z_vals)    
    D = D; L = (8,5); Re=1000; T=Float32
    U₊ = (T(cosd(θ)), T(-sind(θ)))
    sim, Rotor₁, Rotor₂ = Rotor_2D(D, λ₁, λ₂, U₊, Array; L, Re, T)

    datadir = "/sims/data/rotor_BiotSimulation/"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"

    P, U, t = read_meanflow(meanflow_file;dir=datadir, stats = true, stats_turb = false)

    x_force₁ = T[]
    y_force₁ = T[]
    x_force₂ = T[]
    y_force₂ = T[]
    for (i, z) in enumerate(z_vals)
        z_idx = (z*D)+1 |> ceil |> Int
        P_plane = P[:,:,z_idx]
        U_plane = U[:,:,z_idx,1:2]

        @assert size(sim.flow.u) == size(U_plane)
        @assert size(sim.flow.p) == size(P_plane)
        sim.flow.u .= U_plane
        sim.flow.p .= P_plane

        p_force₁ = WaterLily.pressure_force(sim.flow,Rotor₁)
        v_force₁ = WaterLily.viscous_force(sim.flow,Rotor₁)
        t_force₁ = p_force₁ .+ v_force₁
        fx₁, fy₁ = -t_force₁[1], -t_force₁[2]
        push!(x_force₁, fx₁)
        push!(y_force₁, fy₁)

        p_force₂ = WaterLily.pressure_force(sim.flow,Rotor₂)
        v_force₂ = WaterLily.viscous_force(sim.flow,Rotor₂)
        t_force₂ = p_force₂ .+ v_force₂
        fx₂, fy₂ = -t_force₂[1], -t_force₂[2]
        push!(x_force₂, fx₂)
        push!(y_force₂, fy₂)
    end
    return x_force₁, y_force₁, x_force₂, y_force₂
end



