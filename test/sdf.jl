using WaterLily
using Test
using GPUArrays
using CUDA
using OutMacro
using LinearAlgebra
using Plots

function check_sdf_field!(sim::Union{Simulation,BiotSimulation}, D, xvals, yvals, zvals)
    N, n = WaterLily.size_u(sim.flow.u)
    Nx, Ny, Nz = N[1], N[2], N[3]
    σ = sim.flow.σ |> Array
    measure_sdf!(σ, sim.body, sim_time(sim))
    sdf = σ |> Array
    x_range = 1:Nx 
    y_range = 1:Ny 
    z_range = 1:Nz 

    for z_val in zvals
        sdf_vals_z_slice = zeros(eltype(sdf),Nx, Ny)
        for I in WaterLily.slice(N, z_val, 3)
            ix, iy, iz = Tuple(I)
            sdf_vals_z_slice[ix, iy] = sdf[I]
        end
        plt1 = flood(
            sdf_vals_z_slice;
            clims = (minimum(sdf_vals_z_slice), maximum(sdf_vals_z_slice)),
            cfill = :viridis,
            colorbar_title = "SDF",
            aspect_ratio = 1,
            xlims = (x_range[1], x_range[end]),
            ylims = (y_range[1], y_range[end]),
            xticks = x_range[1]:D:x_range[end],
            yticks = y_range[1]:D:y_range[end],
            title = "Signed Distance Field at z = $(z_val) plane",
            xlabel = "x", ylabel = "y"
        )
        Plots.contour!(
            plt1,
            x_range, y_range, sdf_vals_z_slice'; 
            levels=[0.0],                         
            linewidth=2,                            
            color=:red,                            
            legend=false
        )
        savefig(plt1, string(@__DIR__) * "/tex/sdf_z_slice_$(z_val).pdf")
    end

    for y_val in yvals
        sdf_vals_y_slice = zeros(eltype(sdf),Nx, Nz)
        for I in WaterLily.slice(N, y_val, 2)
            ix, iy, iz = Tuple(I)
            sdf_vals_y_slice[ix, iz] = sdf[I]
        end
        plt1 = flood(
            sdf_vals_y_slice;
            clims = (minimum(sdf_vals_y_slice), maximum(sdf_vals_y_slice)),
            cfill = :viridis,
            colorbar_title = "SDF",
            aspect_ratio = 1,
            xlims = (x_range[1], x_range[end]),
            ylims = (z_range[1], z_range[end]),
            xticks = x_range[1]:D:x_range[end],
            yticks = z_range[1]:D:z_range[end],
            title = "Signed Distance Field at y = $(y_val) plane",
            xlabel = "x", ylabel = "z"
        )
        Plots.contour!(
            plt1,
            x_range, z_range, sdf_vals_y_slice';      
            levels=[0.0],                          
            linewidth=2,                           
            color=:red,                              
            legend=false
        )
        savefig(plt1, string(@__DIR__) * "/tex/sdf_y_slice_$(y_val).pdf")
    end

    for x_val in xvals
        sdf_vals_x_slice = zeros(eltype(sdf),Ny, Nz)
        for I in WaterLily.slice(N, x_val, 1)
            ix, iy, iz = Tuple(I)
            sdf_vals_x_slice[iy, iz] = sdf[I]
        end
        plt1 = flood(
            sdf_vals_x_slice;
            clims = (minimum(sdf_vals_x_slice), maximum(sdf_vals_x_slice)),
            cfill = :viridis,
            colorbar_title = "SDF",
            aspect_ratio = 1,
            xlims = (y_range[1], y_range[end]),
            ylims = (z_range[1], z_range[end]),
            xticks = y_range[1]:D:y_range[end],
            yticks = y_range[1]:D:y_range[end],
            title = "Signed Distance Field at x = $(x_val) plane",
            xlabel = "y", ylabel = "z"
        )
        Plots.contour!(
            plt1,
            y_range, z_range, sdf_vals_x_slice';      
            levels=[0.0],                          
            linewidth=2,                           
            color=:red,                              
            legend=false
        )
        savefig(plt1, string(@__DIR__) * "/tex/sdf_x_slice_$(x_val).pdf")
    end
end

"""
    sdf_slice(sim, axis::Int, idx::Int, ε::Real)

Return `(xlist, ylist)` indices of surface voxels where |σ| < ε
on a 2D slice of the domain defined by `axis` and `idx`.
"""
function sdf_slice(sim, axis::Int, idx::Int, ε::Real)
    Nx, Ny, Nz = size(sim.flow.p)
    σ = sim.flow.σ |> Array
    measure_sdf!(σ, sim.body,t=zero(T))
    sdf = σ |> Array
    if axis == 1
        I = [(j,k) for j in 1:Ny, k in 1:Nz if abs(sdf[idx,j,k]) <  ε]
        xlist, ylist = first.(I), last.(I)
    elseif axis == 2
        I = [(i,k) for i in 1:Nx, k in 1:Nz if abs(sdf[i,idx,k]) < ε]
        xlist, ylist = first.(I), last.(I)
    elseif axis == 3
        I = [(i,j) for i in 1:Nx, j in 1:Ny if abs(sdf[i,j,idx]) < ε]
        xlist, ylist = first.(I), last.(I)
    else
        error("axis must be 1, 2, or 3")
    end

    return xlist, ylist
end