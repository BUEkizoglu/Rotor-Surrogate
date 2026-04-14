using WaterLily,BiotSavartBCs
using Test
using GPUArrays
using CUDA
using OutMacro
using LinearAlgebra
using Plots, LaTeXStrings
include("../help/MeanFlow_PostProc.jl")

function plot_body_slice!(sim; axis=3, idx=1, levels=[0], color=:green)
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    σ = Array(sim.flow.σ)

    if axis == 1
        body = σ[idx, :, :]
    elseif axis == 2
        body = σ[:, idx, :]
    elseif axis == 3
        body = σ[:, :, idx]
    else
        error("Invalid axis: must be 1, 2, or 3")
    end
    contour!(permutedims(body); levels=levels, color=color, linewidth=2)
end


function get_global_clims(sim; slice_z_idx, duration, tstep)
    t₀ = sim_time(sim)
    t = sum(sim.flow.Δt[1:end-1])
    vmin = Inf
    vmax = -Inf
    for tᵢ in range(t₀, t₀ + duration; step = tstep)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            biot_mom_step!(sim.flow,sim.pois)
            t += sim.flow.Δt[end]
        end
        println("tU/L=",round(tᵢ,digits=4),",  Δt=",round(sim.flow.Δt[end],digits=3))
        p = sim.flow.p
        if ndims(p) == 3
            s = Array(p[:, :, slice_z_idx])
        elseif ndims(p) == 2
            s = Array(p)
        else
            error("Pressure field has unexpected dimensions: ", size(p))
        end
        vmin = min(vmin, minimum(s))
        vmax = max(vmax, maximum(s))
    end
    return vmin, vmax
end

function log_pressure_sim_double(sim,Rotor₁,Rotor₂;D, λ₁, λ₂, θ, slice_z_idx, duration, tstep, vmin, vmax)
    output_dir="../test/tex/"
    logger_name="test_psolver_$(D)_$(λ₁)_$(λ₂)_$(θ)"
    pressure_gif_name="pressure_monitor_z$(slice_z_idx)_$(D)_$(λ₁)_$(λ₂)_$(θ).gif"
    curl_gif_name="curl_monitor_z$(slice_z_idx)_$(D)_$(λ₁)_$(λ₂)_$(θ).gif" 
    log_plot_name="psolver_$(D)_$(λ₁)_$(λ₂)_$(θ).png"
    force_plot_name="forces_$(D)_$(λ₁)_$(λ₂)_$(θ).pdf"
    gif_file_pressure = joinpath(output_dir, pressure_gif_name)
    gif_file_curl = joinpath(output_dir, curl_gif_name)
    force_plot_file = joinpath(output_dir, force_plot_name)

    WaterLily.logger(joinpath(output_dir, logger_name))

    t₀ = sim_time(sim)
    forces_p1 = Vector{Float32}[]
    forces_v1= Vector{Float32}[]
    forces_p2 = Vector{Float32}[]
    forces_v2 = Vector{Float32}[]

    anim_pressure = Animation()
    anim_curl = Animation()
    for tᵢ in range(t₀, t₀ + duration; step = tstep)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            sim_step!(sim)
            pforce1 = WaterLily.pressure_force(sim.flow,Rotor₁)
            vforce1 = WaterLily.viscous_force(sim.flow,Rotor₁)
            pforce2 = WaterLily.pressure_force(sim.flow,Rotor₂)
            vforce2 = WaterLily.viscous_force(sim.flow,Rotor₂)
            push!(forces_p1, pforce1)
            push!(forces_v1, vforce1)
            push!(forces_p2, pforce2)
            push!(forces_v2, vforce2)
            t += sim.flow.Δt[end]
        end
        println("tU/L=", round(tᵢ, digits=4), ",  Δt=", round(sim.flow.Δt[end], digits=3))

        a = sim.flow.σ
        @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        curl = a[:, :, slice_z_idx] |> Array
        p = sim.flow.p[:, :, slice_z_idx] |> Array

        # Curl plot
        Plots.plot()
        flood(curl; shift=(0,0), cfill=:vik, clims=(-10,10), levels=10) 
        Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=15), color=:black, linewidth=1, legend=false)
        plot_body_slice!(sim; axis=3, idx=slice_z_idx)
        frame(anim_curl)

        # Pressure plot
        Plots.plot()
        flood(p; shift=(0,0), cfill=:vik, clims=(vmin,vmax), levels=15)
        Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=15), color=:black, linewidth=1, legend=false)
        plot_body_slice!(sim; axis=3, idx=slice_z_idx)
        frame(anim_pressure)
    end
    gif(anim_pressure, gif_file_pressure)
    gif(anim_curl, gif_file_curl)
    plot_logger_fix(joinpath(output_dir, logger_name))
    savefig(joinpath(output_dir, log_plot_name))
    time = cumsum(sim.flow.Δt[5:end-1])

    P_force1 = mapreduce(permutedims,vcat,forces_p1)
    Fp_x1, Fp_y1, Fp_z1 = P_force1[:,1], P_force1[:,2], P_force1[:,3]
    V_force1 = mapreduce(permutedims,vcat,forces_v1)
    Fv_x1, Fv_y1, Fv_z1 = V_force1[:,1], V_force1[:,2], V_force1[:,3]
    t_force1 = forces_p1 .+ forces_v1
    t_force2 = forces_p2 .+ forces_v2
    T_force1 = mapreduce(permutedims,vcat,t_force1)
    T_force2 = mapreduce(permutedims,vcat,t_force2)
    Ft_x1, Ft_y1, Ft_z1 = -T_force1[:,1], -T_force1[:,2], -T_force1[:,3]
    Ft_x2, Ft_y2, Ft_z2 = -T_force2[:,1], -T_force2[:,2], -T_force2[:,3]

    Plots.plot(time / sim.L, Ft_x1[5:end], label = L"F_{D_{1}}",labelfontsize=8,legendfontsize=8,tickfontsize=8,linewidth=2)
    Plots.plot!(time / sim.L, Ft_x2[5:end], label = L"F_{D_{2}}",linewidth=2)
    Plots.plot!(time / sim.L, Ft_y1[5:end], label = L"F_{L_{1}}",linewidth=2)
    Plots.plot!(time / sim.L, Ft_y2[5:end], label = L"F_{L_{2}}",linewidth=2)
    Plots.plot!(time / sim.L, Ft_x1[5:end]+Ft_x2[5:end], label = L"F_{D_{total}}", linestyle=:dashdot,linewidth=2)
    Plots.plot!(time / sim.L, Ft_y1[5:end]+Ft_y2[5:end], label = L"F_{L_{total}}", linestyle=:dashdot,linewidth=2)
    Plots.xlabel!("tU/L")
    Plots.ylabel!("Force")
    savefig(force_plot_file)
end

function log_pressure_sim_single(sim,Rotor₁;D, λ, θ, slice_z_idx, duration, tstep, vmin, vmax)
    output_dir="../test/tex/"
    logger_name="test_psolver"
    pressure_gif_name="pressure_monitor_z$(slice_z_idx)_$(D)_$(λ)_$(θ).gif"
    curl_gif_name="curl_monitor_z$(slice_z_idx)_$(D)_$(λ)_$(θ).gif" 
    log_plot_name="psolver_$(D)_$(λ)_$(θ).png"
    force_plot_name="forces_$(D)_$(λ)_$(θ).pdf"
    gif_file_pressure = joinpath(output_dir, pressure_gif_name)
    gif_file_curl = joinpath(output_dir, curl_gif_name)
    force_plot_file = joinpath(output_dir, force_plot_name)

    WaterLily.logger(joinpath(output_dir, logger_name))

    t₀ = sim_time(sim)
    forces_p1 = Vector{Float32}[]
    forces_v1= Vector{Float32}[]

    anim_pressure = Animation()
    anim_curl = Animation()
    for tᵢ in range(t₀, t₀ + duration; step = tstep)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            sim_step!(sim)
            pforce1 = WaterLily.pressure_force(sim.flow,Rotor₁)
            vforce1 = WaterLily.viscous_force(sim.flow,Rotor₁)
            push!(forces_p1, pforce1)
            push!(forces_v1, vforce1)
            t += sim.flow.Δt[end]
        end
        println("tU/L=", round(tᵢ, digits=4), ",  Δt=", round(sim.flow.Δt[end], digits=3))

        a = sim.flow.σ
        @inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        curl = a[:, :, slice_z_idx] |> Array
        p = sim.flow.p[:, :, slice_z_idx] |> Array

        # Curl plot
        Plots.plot()
        flood(curl; shift=(0,0), cfill=:vik, clims=(-10,10), levels=10) 
        Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=15), color=:black, linewidth=1, legend=false)
        plot_body_slice!(sim; axis=3, idx=slice_z_idx)
        frame(anim_curl)

        # Pressure plot
        Plots.plot()
        flood(p; shift=(0,0), cfill=:vik, clims=(vmin,vmax), levels=15)
        Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=15), color=:black, linewidth=1, legend=false)
        plot_body_slice!(sim; axis=3, idx=slice_z_idx)
        frame(anim_pressure)
    end
    gif(anim_pressure, gif_file_pressure)
    gif(anim_curl, gif_file_curl)
    plot_logger_fix(joinpath(output_dir, logger_name))
    savefig(joinpath(output_dir, log_plot_name))
    time = cumsum(sim.flow.Δt[5:end-1])

    P_force1 = mapreduce(permutedims,vcat,forces_p1)
    Fp_x1, Fp_y1, Fp_z1 = P_force1[:,1], P_force1[:,2], P_force1[:,3]
    V_force1 = mapreduce(permutedims,vcat,forces_v1)
    Fv_x1, Fv_y1, Fv_z1 = V_force1[:,1], V_force1[:,2], V_force1[:,3]
    t_force1 = forces_p1 .+ forces_v1
    T_force1 = mapreduce(permutedims,vcat,t_force1)
    Ft_x1, Ft_y1, Ft_z1 = -T_force1[:,1], -T_force1[:,2], -T_force1[:,3]

    Plots.plot(time / sim.L, Ft_x1[5:end], label = L"F_{D}",labelfontsize=8,legendfontsize=8,tickfontsize=8,linewidth=2)
    Plots.plot!(time / sim.L, Ft_y1[5:end], label = L"F_{L}",linewidth=2)
    Plots.xlabel!("tU/L")
    Plots.ylabel!("Force")
    savefig(force_plot_file)
end

function log_ω_mag_sim(sim; probe_xy_idx, slice_z_idx, duration, tstep, vmin, vmax)
    output_dir="/test/"
    logger_name="test_psolver"
    pressure_gif_name="pressure_monitor_z$(slice_z_idx).gif"
    curl_gif_name="curl_monitor_z$(slice_z_idx).gif" 
    log_plot_name="psolver.png"
    force_plot_name="forces.png"
    gif_file_pressure = joinpath(output_dir, pressure_gif_name)
    gif_file_curl = joinpath(output_dir, curl_gif_name)
    force_plot_file = joinpath(output_dir, force_plot_name)

    WaterLily.logger(joinpath(output_dir, logger_name))

    t₀ = sim_time(sim)
    forces_p = Float32[]
    forces_ν = Float32[]

    anim_pressure = Animation()
    anim_curl = Animation()
    for tᵢ in range(t₀, t₀ + duration; step = tstep)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            biot_mom_step!(sim.flow,sim.pois)
            if !(sim.body isa WaterLily.NoBody)
                force = -WaterLily.pressure_force(sim)[1]
                vforce = -WaterLily.viscous_force(sim)[1]
            else
                p = sim.flow.p
                if CUDA.has_cuda() && isa(p, CUDA.CuArray)
                    p_val = Array(p)[probe_xy_idx]
                else
                    p_val = p[probe_xy_idx]
                end
                force = p_val
                vforce = NaN
            end
            push!(forces_p, force)
            push!(forces_ν, vforce)
            t += sim.flow.Δt[end]
        end

        println("tU/L=", round(tᵢ, digits=4), ",  Δt=", round(sim.flow.Δt[end], digits=3))

        if !(sim.body isa WaterLily.NoBody)
            a = sim.flow.σ
            @inside a[I] = WaterLily.λ₂(I,sim.flow.u) #*sim.L/sim.U
            curl = a[:, :, slice_z_idx] |> Array
            p = sim.flow.p[:, :, slice_z_idx] |> Array

            
            Plots.plot()
            flood(curl; shift=(0,0), cfill=:vik, clims=(-10,10), levels=10) 
            Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=10), color=:black, linewidth=0.5, legend=false)
            #Plots.contour!(permutedims(curl), levels=range(-10,10,length=10), color=:black, linewidth=0.5, legend=false)
            # plot_body_slice!(sim; axis=3, idx=slice_z_idx)
            frame(anim_curl)

            # Pressure plot
            Plots.plot()
            flood(p; shift=(0,0), cfill=:vik, clims=(vmin,vmax), levels=10)
            Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=10), color=:black, linewidth=0.5, legend=false)
            # plot_body_slice!(sim; axis=3, idx=slice_z_idx)
            frame(anim_pressure)

        else
            a = sim.flow.σ
            @inside a[I] = WaterLily.λ₂(I,sim.flow.u) #*sim.L/sim.U
            curl = a[:, :, slice_z_idx] |> Array
            p = sim.flow.p[:, :, slice_z_idx] |> Array

            Plots.plot()
            flood(curl; shift=(0,0), cfill=:vik, clims=(-10,10), levels=10) 
            Plots.contour!(permutedims(p), levels=range(vmin,vmax,length=10), color=:black, linewidth=0.5)
            scatter!([probe_xy_idx[1]], [probe_xy_idx[2]], label="Probe Location", color=:red)
            #Plots.contour!(permutedims(curl), levels=range(-10,10,length=10), color=:black, linewidth=0.5, legend=false)
            frame(anim_curl)

            Plots.plot()
            flood(p; shift=(0,0), cfill=:vik, clims=(vmin,vmax), levels=10)
            scatter!([probe_xy_idx[1]], [probe_xy_idx[2]], label="Probe Location", color=:red)
            frame(anim_pressure)
        end
    end
    gif(anim_pressure, gif_file_pressure)
    gif(anim_curl, gif_file_curl)
    plot_logger_fix(joinpath(output_dir, logger_name))
    savefig(joinpath(output_dir, log_plot_name))

    time = cumsum(sim.flow.Δt[4:end-1])
    label_p = !(sim.body isa WaterLily.NoBody) ? "pressure force" : "pressure at probe"
    Plots.plot(time / sim.L, forces_p[4:end], label = label_p)
    if !(sim.body isa WaterLily.NoBody)
        Plots.plot!(time / sim.L, forces_ν[4:end], label = "viscous force")
    end
    Plots.xlabel!("tU/L")
    Plots.ylabel!(label_p)
    savefig(force_plot_file)
end










