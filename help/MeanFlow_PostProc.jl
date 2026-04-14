using OutMacro
using WaterLily,BiotSavartBCs
using CUDA
using ReadVTK, WriteVTK
using FFTW, Interpolations, JLD2, DelimitedFiles, Plots, LaTeXStrings, Printf, DSP, Statistics
using Dates
import Base: time
using ColorSchemes

Plots.default(
    fontfamily = "Computer Modern",
    linewidth = 1,
    framestyle = :box,
    grid = false,
    left_margin = Plots.Measures.Length(:mm, 5),
    right_margin = Plots.Measures.Length(:mm, 5),
    bottom_margin = Plots.Measures.Length(:mm, 5),
    top_margin = Plots.Measures.Length(:mm, 5),
    titlefontsize = 14,
    legendfontsize = 14,
    tickfontsize = 14,
    labelfontsize = 14,
)

struct MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf}
    P :: Sf # pressure
    U :: Vf # velocity
    UU :: Union{Mf, Nothing} # squared velocity u⊗u
    τ  :: Union{Mf, Nothing} # Reynolds stress
    t :: Vector{T}           # time

    function MeanFlow(flow::Flow{D,T}; t_init=0.0, stats_turb=false) where {D,T}
        f = typeof(flow.u).name.wrapper
        P  = zeros(T, size(flow.p)) |> f
        U  = zeros(T, size(flow.u)) |> f
        UU = stats_turb ? zeros(T, size(flow.p)..., D, D) |> f : nothing
        τ  = stats_turb ? zeros(T, size(flow.p)..., D, D) |> f : nothing
        new{T, typeof(P), typeof(U), typeof(UU)}(P, U, UU, τ, T[t_init])
    end
end
time(meanflow::MeanFlow) = meanflow.t[end]-meanflow.t[1]

function reset!(meanflow::MeanFlow; t_init=0.0)
    fill!(meanflow.P, 0)
    fill!(meanflow.U, 0)
    !isnothing(meanflow.UU) && fill!(meanflow.UU, 0)
    !isnothing(meanflow.τ) && fill!(meanflow.τ, 0)
    empty!(meanflow.t)
    push!(meanflow.t, t_init)
end

function load!(meanflow::MeanFlow, fname::String; dir="data/")
    obj = jldopen(dir*fname)
    @assert size(meanflow.P) == size(obj["P"]) "Simulation size does not match the size of the JLD2-stored simulation."
    f = typeof(meanflow.U).name.wrapper
    meanflow.P  .= obj["P"] |> f
    meanflow.U  .= obj["U"] |> f
    !isnothing(meanflow.UU) && (meanflow.UU .= obj["UU"] |> f)
    !isnothing(meanflow.τ)  && (meanflow.τ  .= obj["τ"]  |> f)
    empty!(meanflow.t)
    push!(meanflow.t, obj["t"]...)
    close(obj)
end

function write!(fname, meanflow::MeanFlow; dir="data/", vtk=false, sim=nothing)
    jldsave(dir*fname*".jld2";
        P = Array(meanflow.P),
        U = Array(meanflow.U),
        UU = isnothing(meanflow.UU) ? nothing : Array(meanflow.UU),
        τ  = isnothing(meanflow.τ)  ? nothing : Array(meanflow.τ),
        t = meanflow.t
    )
    if vtk && sim isa Simulation
        copy!(sim.flow, meanflow)
        wr = vtkWriter(fname; dir=dir)
        WaterLily.save!(wr, sim)
        close(wr)
    end
end

function update!(meanflow::MeanFlow, flow::Flow; stats_turb=false)
    dt = WaterLily.time(flow) - meanflow.t[end]
    ε = dt / (dt + (meanflow.t[end] - meanflow.t[1]) + eps(eltype(flow.p)))
    WaterLily.@loop meanflow.P[I] = ε * flow.p[I] + (1.0 - ε) * meanflow.P[I] over I in CartesianIndices(flow.p)
    WaterLily.@loop meanflow.U[Ii] = ε * flow.u[Ii] + (1.0 - ε) * meanflow.U[Ii] over Ii in CartesianIndices(flow.u)

    if stats_turb && !isnothing(meanflow.UU) && !isnothing(meanflow.τ)
        for i in 1:ndims(flow.p), j in 1:ndims(flow.p)
            WaterLily.@loop meanflow.UU[I,i,j] = ε * (flow.u[I,i] * flow.u[I,j]) + (1.0 - ε) * meanflow.UU[I,i,j] over I in CartesianIndices(flow.p)
            WaterLily.@loop meanflow.τ[I,i,j]  = meanflow.UU[I,i,j] - meanflow.U[I,i] * meanflow.U[I,j] over I in CartesianIndices(flow.p)
        end
    end
    push!(meanflow.t, meanflow.t[end] + dt)
end

function copy!(a::Flow, b::MeanFlow)
    a.u .= b.U
    a.p .= b.P
end

function read_forces(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["p_force₁"], f["p_force₂"], f["v_force₁"], f["v_force₂"], f["u_probe"], f["time"]
    end
end

function read_force_and_probe_vals(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["p_force₁"], f["p_force₂"], f["v_force₁"], f["v_force₂"],
               f["u_probe_x"], f["u_probe_y"], f["u_probe_z"], f["time"]
    end
end

function read_force_and_probe_vals_single(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["p_force₁"], f["v_force₁"],
               f["u_probe_x"], f["u_probe_y"], f["u_probe_z"], f["time"]
    end
end

function read_meanflow(fname::String; dir="data/", stats=false, stats_turb=false)
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        if stats && stats_turb
            return f["P"], f["U"], f["UU"], f["τ"], f["t"]
        elseif stats
            return f["P"], f["U"], f["t"]
        else
            error("Invalid combination of 'stats' and 'stats_turb'")
        end
    end
end

function read_flow(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["p"], f["u"], f["Δt"]
    end
end

function read_probe(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["u_probe_x"], f["u_probe_y"], f["u_probe_z"], f["time"]
    end
end

function run_sim_rotor_NonBiotFaces(D, λ₁, λ₂, θ, U₊, backend;dir=dir, L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=false, log_tag=nothing) # Changed
    tag = isnothing(log_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : string(log_tag)
    psolver_name = "psolver_rotor_NonBiotFaces_$(D)_$(λ₁)_$(λ₂)_$(θ)_$(tag)"
    WaterLily.logger(psolver_name)
    if cont
        println("⏯️ Resuming simulation from saved files")
        sim,Rotor₁,Rotor₂,meanflow = sim_cont_NonBiotFaces(D, λ₁, λ₂, θ, U₊, backend; L, Re, T, dir=dir)
        p_force₁,p_force₂,v_force₁,v_force₂,u_probe_x,u_probe_y,u_probe_z,time = read_force_and_probe_vals("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2", dir=dir)
    else
        sim,Rotor₁,Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, backend; L, Re, T)
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        p_force₁ = Vector{T}[]  
        p_force₂ = Vector{T}[]  
        v_force₁ = Vector{T}[]  
        v_force₂ = Vector{T}[]  
        u_probe_x = T[]
        u_probe_y = T[]
        u_probe_z = T[]
        time = T[]
    end

    ts_fname = "timestep_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).log"
    ts_mode = cont ? "a" : "w"
    ts_logger = open(ts_fname, ts_mode)
    if !cont
        println(ts_logger, "tU/D, Δt")
    end

    u_probe_loc_n = @. (u_probe_loc * sim.L) |> ceil |> Int
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim)+stats_interval; remeasure=true, verbose=false)
        push!(p_force₁, WaterLily.pressure_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(p_force₂, WaterLily.pressure_force(sim.flow,Rotor₂)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₁, WaterLily.viscous_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₂, WaterLily.viscous_force(sim.flow,Rotor₂)/(0.5*sim.U^2*sim.L^2))
        push!(u_probe_x, view(sim.flow.u,u_probe_loc_n...,1) |> Array |> x->x[]) 
        push!(u_probe_y, view(sim.flow.u,u_probe_loc_n...,2) |> Array |> x->x[]) 
        push!(u_probe_z, view(sim.flow.u,u_probe_loc_n...,3) |> Array |> x->x[]) 
        push!(time, sim_time(sim))
        println(ts_logger, "$(time[end]), $(sim.flow.Δt[end])")
        ct₁ = round(p_force₁[end][1] + v_force₁[end][1], digits=4)
        ct₂ = round(p_force₂[end][1] + v_force₂[end][1], digits=4)
        verbose && println("tU/D = $(time[end]); Δt = $(sim.flow.Δt[end]); Ct₁ = $ct₁; Ct₂ = $ct₂; U²=$(sim.U^2); L²=$(sim.L^2)")
        if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
            verbose && println("💾 Writing force and probe values")
            jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; p_force₁=p_force₁, p_force₂=p_force₂, v_force₁=v_force₁, v_force₂=v_force₂, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            verbose && println("💾 Writing Flow data")
            WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2",sim.flow,dir=dir)
        end
        if stats && sim_time(sim) > stats_init
            length(meanflow.t) == 1 && reset!(meanflow; t_init=WaterLily.time(sim))
            verbose && println("🧮 Computing stats")
            update!(meanflow, sim.flow; stats_turb=stats_turb)
            if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
                verbose && println("💾 Writing MeanFlow and Flow data")
                write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)", meanflow; dir=dir)
                WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2",sim.flow,dir=dir)
                verbose && println("💾 Writing force and probe values")
                jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; p_force₁=p_force₁, p_force₂=p_force₂, v_force₁=v_force₁, v_force₂=v_force₂, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            end
        end
    end
    verbose && println("💾 Writing final force and probe values")
    jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"; p_force₁=p_force₁, p_force₂=p_force₂, v_force₁=v_force₁, v_force₂=v_force₂, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
    verbose && println("💾 Writing final Flow and MeanFlow data")
    write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)", meanflow; dir=dir)
    WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2",sim.flow,dir=dir)
    wr = vtkWriter("rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ)"; dir=dir)
    WaterLily.save!(wr, sim)
    close(wr)
    close(ts_logger)
    println("✅ Done!")
    return sim, meanflow, p_force₁, p_force₂, v_force₁, v_force₂ 
end

function run_sim_rotor_NonBiotFaces_Single(D, λ₁, θ, U₊, backend;dir=dir, L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=false, log_tag=nothing) # Changed
    tag = isnothing(log_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : string(log_tag)
    psolver_name = "psolver_rotor_NonBiotFaces_$(D)_$(λ₁)_$(θ)_Single_$(tag)"
    WaterLily.logger(psolver_name)
    if cont
        println("⏯️ Resuming simulation from saved files")
        sim,Rotor₁,meanflow = sim_cont_NonBiotFaces_Single(D, λ₁, θ, U₊, backend; L, Re, T, dir=dir)
        p_force₁,v_force₁,u_probe_x,u_probe_y,u_probe_z,time = read_force_and_probe_vals_single("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2", dir=dir)
    else
        sim,Rotor₁ = ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, U₊, backend; L, Re, T)
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        p_force₁ = Vector{T}[]  
        v_force₁ = Vector{T}[]  
        u_probe_x = T[]
        u_probe_y = T[]
        u_probe_z = T[]
        time = T[]
    end

    ts_fname = "timestep_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.log"
    ts_mode = cont ? "a" : "w"
    ts_logger = open(ts_fname, ts_mode)
    if !cont
        println(ts_logger, "tU/D, Δt")
    end

    u_probe_loc_n = @. (u_probe_loc * sim.L) |> ceil |> Int
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim)+stats_interval; remeasure=true, verbose=false)
        push!(p_force₁, WaterLily.pressure_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₁, WaterLily.viscous_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(u_probe_x, view(sim.flow.u,u_probe_loc_n...,1) |> Array |> x->x[]) 
        push!(u_probe_y, view(sim.flow.u,u_probe_loc_n...,2) |> Array |> x->x[]) 
        push!(u_probe_z, view(sim.flow.u,u_probe_loc_n...,3) |> Array |> x->x[]) 
        push!(time, sim_time(sim))
        println(ts_logger, "$(time[end]), $(sim.flow.Δt[end])")
        ct₁ = round(p_force₁[end][1] + v_force₁[end][1], digits=4)
        verbose && println("tU/D = $(time[end]); Δt = $(sim.flow.Δt[end]); Ct₁ = $ct₁")
        if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
            verbose && println("💾 Writing force and probe values")
            jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            verbose && println("💾 Writing Flow data")
            WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
        end
        if stats && sim_time(sim) > stats_init
            length(meanflow.t) == 1 && reset!(meanflow; t_init=WaterLily.time(sim))
            verbose && println("🧮 Computing stats")
            update!(meanflow, sim.flow; stats_turb=stats_turb)
            if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
                verbose && println("💾 Writing MeanFlow and Flow data")
                write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
                WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
                verbose && println("💾 Writing force and probe values")
                jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            end
        end
    end
    verbose && println("💾 Writing final force and probe values")
    jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
    verbose && println("💾 Writing final Flow and MeanFlow data")
    write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
    WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
    wr = vtkWriter("rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single"; dir=dir)
    WaterLily.save!(wr, sim)
    close(wr)
    close(ts_logger)
    println("✅ Done!")
    return sim, meanflow, p_force₁, v_force₁
end

function run_sim_rotor_Validation_Free_Slip(D, λ₁, θ, U₊, backend;dir=dir, L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=false, log_tag=nothing) # Changed
    tag = isnothing(log_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : string(log_tag)
    psolver_name = "psolver_rotor_NonBiotFaces_$(D)_$(λ₁)_$(θ)_Single_$(tag)"
    WaterLily.logger(psolver_name)
    if cont
        println("⏯️ Resuming simulation from saved files")
        sim,Rotor₁,meanflow = sim_cont_Validation_Free_Slip(D, λ₁, θ, U₊, backend; L, Re, T, dir=dir)
        p_force₁,v_force₁,u_probe_x,u_probe_y,u_probe_z,time = read_force_and_probe_vals_single("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2", dir=dir)
    else
        sim,Rotor₁ = ThreeD_Rotor_Validation_Free_Slip(D, λ₁, U₊, backend; L, Re, T)
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        p_force₁ = Vector{T}[]  
        v_force₁ = Vector{T}[]  
        u_probe_x = T[]
        u_probe_y = T[]
        u_probe_z = T[]
        time = T[]
    end

    ts_fname = "timestep_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.log"
    ts_mode = cont ? "a" : "w"
    ts_logger = open(ts_fname, ts_mode)
    if !cont
        println(ts_logger, "tU/D, Δt")
    end

    u_probe_loc_n = @. (u_probe_loc * sim.L) |> ceil |> Int
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim)+stats_interval; remeasure=true, verbose=false)
        push!(p_force₁, WaterLily.pressure_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₁, WaterLily.viscous_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(u_probe_x, view(sim.flow.u,u_probe_loc_n...,1) |> Array |> x->x[]) 
        push!(u_probe_y, view(sim.flow.u,u_probe_loc_n...,2) |> Array |> x->x[]) 
        push!(u_probe_z, view(sim.flow.u,u_probe_loc_n...,3) |> Array |> x->x[]) 
        push!(time, sim_time(sim))
        println(ts_logger, "$(time[end]), $(sim.flow.Δt[end])")
        ct₁ = round(p_force₁[end][1] + v_force₁[end][1], digits=4)
        verbose && println("tU/D = $(time[end]); Δt = $(sim.flow.Δt[end]); Ct₁ = $ct₁")
        if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
            verbose && println("💾 Writing force and probe values")
            jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            verbose && println("💾 Writing Flow data")
            WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
        end
        if stats && sim_time(sim) > stats_init
            length(meanflow.t) == 1 && reset!(meanflow; t_init=WaterLily.time(sim))
            verbose && println("🧮 Computing stats")
            update!(meanflow, sim.flow; stats_turb=stats_turb)
            if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
                verbose && println("💾 Writing MeanFlow and Flow data")
                write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
                WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
                verbose && println("💾 Writing force and probe values")
                jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            end
        end
    end
    verbose && println("💾 Writing final force and probe values")
    jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
    verbose && println("💾 Writing final Flow and MeanFlow data")
    write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
    WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
    wr = vtkWriter("rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single"; dir=dir)
    WaterLily.save!(wr, sim)
    close(wr)
    close(ts_logger)
    println("✅ Done!")
    return sim, meanflow, p_force₁, v_force₁
end

function run_sim_rotor_Validation(D, λ₁, θ, U₊, backend;dir=dir, L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=false, log_tag=nothing) # Changed
    tag = isnothing(log_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : string(log_tag)
    psolver_name = "psolver_rotor_NonBiotFaces_$(D)_$(λ₁)_$(θ)_Single_$(tag)"
    WaterLily.logger(psolver_name)
    if cont
        println("⏯️ Resuming simulation from saved files")
        sim,Rotor₁,meanflow = sim_cont_Validation(D, λ₁, θ, U₊, backend; L, Re, T, dir=dir)
        p_force₁,v_force₁,u_probe_x,u_probe_y,u_probe_z,time = read_force_and_probe_vals_single("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2", dir=dir)
    else
        sim,Rotor₁ = ThreeD_Rotor_Validation(D, λ₁, U₊, backend; L, Re, T)
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        p_force₁ = Vector{T}[]  
        v_force₁ = Vector{T}[]  
        u_probe_x = T[]
        u_probe_y = T[]
        u_probe_z = T[]
        time = T[]
    end

    ts_fname = "timestep_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.log"
    ts_mode = cont ? "a" : "w"
    ts_logger = open(ts_fname, ts_mode)
    if !cont
        println(ts_logger, "tU/D, Δt")
    end

    u_probe_loc_n = @. (u_probe_loc * sim.L) |> ceil |> Int
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim)+stats_interval; remeasure=true, verbose=false)
        push!(p_force₁, WaterLily.pressure_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₁, WaterLily.viscous_force(sim.flow,Rotor₁)/(0.5*sim.U^2*sim.L^2))
        push!(u_probe_x, view(sim.flow.u,u_probe_loc_n...,1) |> Array |> x->x[]) 
        push!(u_probe_y, view(sim.flow.u,u_probe_loc_n...,2) |> Array |> x->x[]) 
        push!(u_probe_z, view(sim.flow.u,u_probe_loc_n...,3) |> Array |> x->x[]) 
        push!(time, sim_time(sim))
        println(ts_logger, "$(time[end]), $(sim.flow.Δt[end])")
        ct₁ = round(p_force₁[end][1] + v_force₁[end][1], digits=4)
        verbose && println("tU/D = $(time[end]); Δt = $(sim.flow.Δt[end]); Ct₁ = $ct₁")
        if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
            verbose && println("💾 Writing force and probe values")
            jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            verbose && println("💾 Writing Flow data")
            WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
        end
        if stats && sim_time(sim) > stats_init
            length(meanflow.t) == 1 && reset!(meanflow; t_init=WaterLily.time(sim))
            verbose && println("🧮 Computing stats")
            update!(meanflow, sim.flow; stats_turb=stats_turb)
            if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
                verbose && println("💾 Writing MeanFlow and Flow data")
                write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
                WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
                verbose && println("💾 Writing force and probe values")
                jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
            end
        end
    end
    verbose && println("💾 Writing final force and probe values")
    jldsave(dir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"; p_force₁=p_force₁, v_force₁=v_force₁, time=time, u_probe_x=u_probe_x, u_probe_y=u_probe_y, u_probe_z=u_probe_z)
    verbose && println("💾 Writing final Flow and MeanFlow data")
    write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single", meanflow; dir=dir)
    WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2",sim.flow,dir=dir)
    wr = vtkWriter("rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single"; dir=dir)
    WaterLily.save!(wr, sim)
    close(wr)
    close(ts_logger)
    println("✅ Done!")
    return sim, meanflow, p_force₁, v_force₁
end

function run_sim_rotor_BiotSimualtion(D, λ₁, λ₂, backend; L, u_probe_loc, u_probe_component, Re, T, restart=false, cont=false) # Changed
    if cont
        println("⏯️ Resuming simulation from saved files")
        sim,Rotor₁,Rotor₂,meanflow = sim_cont(D, λ₁, λ₂, backend; L, Re, T, dir=datadir)
        p_force₁,p_force₂,v_force₁,v_force₂,u_probe,time = read_force_and_probe_vals("force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2", dir=datadir)
    else
        sim,Rotor₁,Rotor₂ = ThreeD_Rotor_BiotSimulation(D, λ₁, λ₂, backend; L, Re, T)
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        p_force₁ = Vector{SVector{3, T}}()
        p_force₂ = Vector{SVector{3, T}}()
        v_force₁ = Vector{SVector{3, T}}()
        v_force₂ = Vector{SVector{3, T}}()
        u_probe_x = Vector{T}()
        u_probe_y = Vector{T}()
        u_probe_z = Vector{T}()
        time = Vector{T}()
    end
    ts_logger = open("timestep_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).log", cont ? "a" : "w")
    if !cont
        println(ts_logger, "tU/D, Δt")
    end
    
    u_probe_loc_n = @. (u_probe_loc * sim.L) |> ceil |> Int
    while sim_time(sim) < time_max
        sim_step!(sim, sim_time(sim)+stats_interval; remeasure=true, verbose=false)
        push!(p_force₁, WaterLily.pressure_force(sim.flow,sim.body.a)/(0.5*sim.U^2*sim.L^2))
        push!(p_force₂, WaterLily.pressure_force(sim.flow,sim.body.b)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₁, WaterLily.viscous_force(sim.flow,sim.body.a)/(0.5*sim.U^2*sim.L^2))
        push!(v_force₂, WaterLily.viscous_force(sim.flow,sim.body.b)/(0.5*sim.U^2*sim.L^2))
        push!(u_probe_x, view(sim.flow.u,u_probe_loc_n...,1) |> Array |> x->x[]) 
        push!(u_probe_y, view(sim.flow.u,u_probe_loc_n...,2) |> Array |> x->x[]) 
        push!(u_probe_z, view(sim.flow.u,u_probe_loc_n...,3) |> Array |> x->x[]) 
        push!(time, sim_time(sim))
        println(ts_logger, "$(time[end]), $(sim.flow.Δt[end])")
        ct₁ = round(p_force₁[end][1], digits=4) #+ v_force₁[end][1], digits=4)
        ct₂ = round(p_force₂[end][1], digits=4) #+ v_force₂[end][1], digits=4)
        verbose && println("tU/D = $(time[end]); Δt = $(sim.flow.Δt[end]); Ct₁ = $ct₁; Ct₂ = $ct₂")
        if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
            verbose && println("💾 Writing force and probe values")
            jldsave(datadir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2"; p_force₁=p_force₁, p_force₂=p_force₂, v_force₁=v_force₁, v_force₂=v_force₂, time=time, u_probe=u_probe)
            WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2",sim.flow,dir=datadir)
        end
        if stats && sim_time(sim) > stats_init
            length(meanflow.t) == 1 && reset!(meanflow; t_init=WaterLily.time(sim))
            verbose && println("🧮 Computing stats")
            update!(meanflow, sim.flow; stats_turb=stats_turb)
            if WaterLily.sim_time(sim)%dump_interval < sim.flow.Δt[end]*sim.U/sim.L + 0.1
                verbose && println("💾 Writing MeanFlow and Flow data")
                write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)", meanflow; dir=datadir)
                WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2",sim.flow,dir=datadir)
            end
        end
    end
    verbose && println("💾 Writing final force and probe values")
    jldsave(datadir*"force_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2"; p_force₁=p_force₁, p_force₂=p_force₂, v_force₁=v_force₁, v_force₂=v_force₂, time=time, u_probe=u_probe)
    verbose && println("💾 Writing final Flow and MeanFlow data")
    write!(fname_output*"_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)", meanflow; dir=datadir)
    WaterLily.save!("flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2",sim.flow,dir=datadir)
    wr = vtkWriter("rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)"; dir=datadir)
    WaterLily.save!(wr, sim)
    close(wr)
    close(ts_logger)
    println("✅ Done!")
    return sim, meanflow, p_force₁, p_force₂, v_force₁, v_force₂ 
end

function moving_average(Δt, data, t)
    n = length(data)
    meanforce = zeros(n)
    ε = zeros(n)
    meanforce[1] = data[1]
    for i in 2:n
        ε[i-1] = Δt / (Δt + (t[i] - t[1]))   
        meanforce[i] = ε[i-1]*data[i] + (1 - ε[i-1])*meanforce[i-1]
    end
    return meanforce
end

function σ(data, t_σ, σ_init)
    idx = t_σ .> σ_init
    data, t_σ = data[idx], t_σ[idx]
    return std(data)
end

function moving_σ(data, t, σ_i)
    σ_init = convert(Float32,σ_i)
    idx = t .> σ_init
    data, t = data[idx], t[idx]
    n = length(data)
    σ_array = zeros(n)

    if n == 0
        return σ_array  # Return empty array if no data after cutoff
    end

    mean = data[1]
    M2 = 0.0
    σ_array[1] = 0.0  # std dev is zero for single sample

    for i in 2:n
        delta = data[i] - mean
        mean += delta / i
        M2 += delta * (data[i] - mean)
        σ_array[i] = sqrt(M2 / (i - 1))
    end

    return σ_array
end

function sim_cont(D, λ₁, λ₂, backend; L, Re, T, dir="data/rotating_cylinder/")
    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_BiotSimulation(D, λ₁, λ₂, backend; L, Re, T)
    meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂).jld2"
    WaterLily.load!(sim.flow; fname=flow_file, dir=dir)
    if WaterLily.sim_time(sim) < stats_init
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        println("Initialized empty MeanFlow (tU/D = $(round(sim_time(sim), digits=2)) < stats_init = $stats_init)")
    else
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        load!(meanflow, meanflow_file; dir=dir)
        println("Loaded MeanFlow from $meanflow_file (tU/D = $(round(sim_time(sim), digits=2)))")
    end
    return sim, Rotor₁, Rotor₂, meanflow
end

function sim_cont_NonBiotFaces(D, λ₁, λ₂, θ, U₊, backend; L, Re, T, dir="data/rotating_cylinder/")
    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, backend; L, Re, T)
    meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
    WaterLily.load!(sim.flow; fname=flow_file, dir=dir)
    if WaterLily.sim_time(sim)-0.09 <= stats_init
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        println("Initialized empty MeanFlow (tU/D = $(round(sim_time(sim), digits=2)) < stats_init = $stats_init)")
    else
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        load!(meanflow, meanflow_file; dir=dir)
        println("Loaded MeanFlow from $meanflow_file (tU/D = $(round(sim_time(sim), digits=2)))")
    end
    return sim, Rotor₁, Rotor₂, meanflow
end

function sim_cont_NonBiotFaces_Single(D, λ₁, θ, U₊, backend; L, Re, T, dir="data/rotating_cylinder/")
    sim, Rotor₁ = ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, U₊, backend; L, Re, T)
    meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    WaterLily.load!(sim.flow; fname=flow_file, dir=dir)
    if WaterLily.sim_time(sim)-0.09 <= stats_init
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        println("Initialized empty MeanFlow (tU/D = $(round(sim_time(sim), digits=2)) < stats_init = $stats_init)")
    else
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        load!(meanflow, meanflow_file; dir=dir)
        println("Loaded MeanFlow from $meanflow_file (tU/D = $(round(sim_time(sim), digits=2)))")
    end
    return sim, Rotor₁, meanflow
end

function sim_cont_Validation_Free_Slip(D, λ₁, θ, U₊, backend; L, Re, T, dir="data/rotating_cylinder/")
    sim, Rotor₁ = ThreeD_Rotor_Validation_Free_Slip(D, λ₁, U₊, backend; L, Re, T)
    meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    WaterLily.load!(sim.flow; fname=flow_file, dir=dir)
    if WaterLily.sim_time(sim)-0.09 <= stats_init
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        println("Initialized empty MeanFlow (tU/D = $(round(sim_time(sim), digits=2)) < stats_init = $stats_init)")
    else
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        load!(meanflow, meanflow_file; dir=dir)
        println("Loaded MeanFlow from $meanflow_file (tU/D = $(round(sim_time(sim), digits=2)))")
    end
    return sim, Rotor₁, meanflow
end

function sim_cont_Validation(D, λ₁, θ, U₊, backend; L, Re, T, dir="data/rotating_cylinder/")
    sim, Rotor₁ = ThreeD_Rotor_Validation(D, λ₁, U₊, backend; L, Re, T)
    meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
    flow_file = "flow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    meanflow_file = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(θ)_Single.jld2"
    WaterLily.load!(sim.flow; fname=flow_file, dir=dir)
    if WaterLily.sim_time(sim)-0.09 <= stats_init
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        println("Initialized empty MeanFlow (tU/D = $(round(sim_time(sim), digits=2)) < stats_init = $stats_init)")
    else
        meanflow = MeanFlow(sim.flow; stats_turb=stats_turb)
        load!(meanflow, meanflow_file; dir=dir)
        println("Loaded MeanFlow from $meanflow_file (tU/D = $(round(sim_time(sim), digits=2)))")
    end
    return sim, Rotor₁, meanflow
end




