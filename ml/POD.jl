using ProperOrthogonalDecomposition
using OutMacro
using DataFrames
using JLD2
using Mmap
using StaticArrays
include("../help/MeanFlow_PostProc.jl")

_asvec(x::AbstractVector) = x
_asvec(x::Int) = [x]
_keys(λ1s, λ2s, θs) = [(λ₁,λ₂,θ) for λ₁ in _asvec(λ1s) for λ₂ in _asvec(λ2s) for θ in _asvec(θs)]

using LinearAlgebra, Mmap

function write_snapshots_to_disk(D::Int, λ1s, λ2s, θs; dir::AbstractString, outdir::AbstractString, T=Float32)
    mkpath(outdir)
    keys = _keys(λ1s, λ2s, θs)
    println("Printing snapshot keys:"); @out keys
    m = length(keys)
    @assert m > 0 "No snapshots to write."

    # Probe grid size once
    λ₁₀, λ₂₀, θ₀ = keys[1]
    mf0 = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁₀)_$(λ₂₀)_$(θ₀).jld2"
    P0, U0, _ = read_meanflow(mf0; dir=dir, stats=true, stats_turb=false)
    Nx, Ny, Nz, d = size(U0)
    N = Nx*Ny*Nz

    # Prepare memory-mapped files
    paths = (
        Ux = joinpath(outdir, "snap_Ux.bin"),
        Uy = joinpath(outdir, "snap_Uy.bin"),
        Uz = joinpath(outdir, "snap_Uz.bin"),
        P  = joinpath(outdir, "snap_P.bin"),
    )

    ios  = Dict{Symbol,IO}()
    mats = Dict{Symbol,Matrix{T}}()
    for (k, p) in pairs(paths)
        io = open(p, "w+")
        ios[k]  = io
        mats[k] = Mmap.mmap(io, Matrix{T}, (N, m))  # file-backed matrix
    end

    buf = Vector{T}(undef, N)  # one reusable column buffer

    # Fill one column at a time
    for (j,(λ₁,λ₂,θ)) in enumerate(keys)
        println("Creating snapshot $(j): (λ₁,λ₂,θ)=($λ₁,$λ₂,$θ)")
        mf = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
        P, U, _ = read_meanflow(mf; dir=dir, stats=true, stats_turb=false)
        @views begin
            # Ux
            buf .= vec(Array(U[:,:,:,1]));  mats[:Ux][:,j] .= buf
            # Uy
            buf .= vec(Array(U[:,:,:,2]));  mats[:Uy][:,j] .= buf
            # Uz
            buf .= vec(Array(U[:,:,:,3]));  mats[:Uz][:,j] .= buf
            # P
            buf .= vec(Array(P));           mats[:P][:,j]  .= buf
        end
        GC.gc()   # keep memory pressure down
    end

    # Flush/close
    for io in values(ios)
        flush(io); close(io)
    end
    return paths, N, m
end

function generate_paths(;dir, outdir)
    keys = _keys(λ1s, λ2s, θs)
    println("Printing snapshot keys:"); @out keys
    m = length(keys)
    @assert m > 0 "No snapshots to write."
    # Probe grid size once
    λ₁₀, λ₂₀, θ₀ = keys[1]
    mf0 = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁₀)_$(λ₂₀)_$(θ₀).jld2"
    P0, U0, _ = read_meanflow(mf0; dir=dir, stats=true, stats_turb=false)
    Nx, Ny, Nz, d = size(U0)
    N = Nx*Ny*Nz
    # Prepare memory-mapped files
    paths = (
        Ux = joinpath(outdir, "snap_Ux.bin"),
        Uy = joinpath(outdir, "snap_Uy.bin"),
        Uz = joinpath(outdir, "snap_Uz.bin"),
        P  = joinpath(outdir, "snap_P.bin"),
    )
    return paths, N, m
end


"""
    run_POD_from_disk(paths, N, m)

Memory-maps the snapshot matrices and runs PODeigen component-wise.
Returns (Φ_U, A_U, Σ_U, Φ_P, A_P, Σ_P) shaped like your original code.
"""
function run_POD_from_disk(paths::NamedTuple, N::Int, m::Int)
    # Map read-only
    XUx = Mmap.mmap(open(paths.Ux, "r"), Matrix{Float32}, (N, m))
    XUy = Mmap.mmap(open(paths.Uy, "r"), Matrix{Float32}, (N, m))
    XUz = Mmap.mmap(open(paths.Uz, "r"), Matrix{Float32}, (N, m))
    XP  = Mmap.mmap(open(paths.P,  "r"), Matrix{Float32}, (N, m))

    # POD per field
    podUx, sUx = ProperOrthogonalDecomposition.PODeigen!(XUx)
    podUy, sUy = ProperOrthogonalDecomposition.PODeigen!(XUy)
    podUz, sUz = ProperOrthogonalDecomposition.PODeigen!(XUz)
    podP,  sP  = ProperOrthogonalDecomposition.PODeigen!(XP)

    Φ_Ux, A_Ux = podUx.modes, podUx.coefficients
    
    Φ_Uy, A_Uy = podUy.modes, podUy.coefficients
    Φ_Uz, A_Uz = podUz.modes, podUz.coefficients
    Φ_P,  A_P  = podP.modes,  podP.coefficients

    Σ_U = cat(reshape(sUx,: ,1), reshape(sUy,: ,1), reshape(sUz,: ,1); dims=3)
    Σ_P = reshape(sP, :, 1)
    Φ_U = (x = Φ_Ux, y = Φ_Uy, z = Φ_Uz)
    A_U = cat(A_Ux, A_Uy, A_Uz; dims=3)

    return Σ_U, Φ_U, A_U, Σ_P, Φ_P, A_P
end

function read_POD(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["Σ_U"], f["A_U"], f["Σ_P"], f["A_P"]
    end
end

function read_modes(fname::String; dir="data/")
    JLD2.jldopen(joinpath(dir, fname), "r") do f
        return f["Mode_U"], f["Mode_P"], f["mode_index"]
    end
end

# Get grid dims once from any existing meanflow file you used to build POD
function meanflow_dims(D, λ₁, λ₂, θ; dir::AbstractString)
    mf = "meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2"
    # Use your read_meanflow; stats_turb choice doesn't matter for dims
    P, U, _ = read_meanflow(mf; dir=dir, stats=true, stats_turb=false)
    Nx, Ny, Nz, _ = size(U)
    return Nx, Ny, Nz
end

"""
    save_pod_modes_as_meanflow(Φ_U, Φ_P; D, dims_ref, dir, outdir, r_keep, T=Float32)

Writes each POD mode k to `outdir/POD_mode_k.jld2` with:
  - U :: Array{T,4} of size (Nx,Ny,Nz,3)
  - P :: Array{T,3} of size (Nx,Ny,Nz)
  - metadata: (:mode_index, :D, :dims, :component_order)

Arguments:
  Φ_U :: (N, m, 3)   # velocity modes stacked along 3rd dim (x,y,z)
  Φ_P :: (N, m)      # pressure modes
  D, dims_ref        # D and a reference (λ₁,λ₂,θ) triple to read dims from
  dir                # directory of your meanflow files (to read dims)
  outdir             # where to save per-mode files
  r_keep             # how many leading modes to write (default: all)
  T                  # storage eltype (Float32 recommended)
"""
function recover_POD_modes(Φ_U::NamedTuple{(:x,:y,:z)}, Φ_P::AbstractArray, dims_ref::NTuple{3,Int}; D::Int, dir::AbstractString, outdir::AbstractString, T=Float32)
    mkpath(outdir)
    λ₁ref, λ₂ref, θref = dims_ref
    Nx, Ny, Nz = meanflow_dims(D, λ₁ref, λ₂ref, θref; dir=dir)

    N, m = size(Φ_U.x)
    @assert size(Φ_P,1) == N "Φ_P length ≠ Φ_U length"
    @assert size(Φ_P,2) == m "Φ_P snapshot count ≠ Φ_U snapshot count"

    # Reusable buffers to keep RAM small
    Ux_mode = Array{T}(undef, Nx,Ny,Nz)
    Uy_mode = Array{T}(undef, Nx,Ny,Nz)
    Uz_mode = Array{T}(undef, Nx,Ny,Nz)
    P_mode = Array{T}(undef, Nx,Ny,Nz)

    for k in 1:m
        # reshape each component back to 3-D
        Ux_mode .= reshape(@view(Φ_U.x[:,k]),Nx,Ny,Nz)
        Uy_mode .= reshape(@view(Φ_U.y[:,k]),Nx,Ny,Nz)
        Uz_mode .= reshape(@view(Φ_U.z[:,k]),Nx,Ny,Nz)
        P_mode .= reshape(@view(Φ_P[:,k]),Nx,Ny,Nz)

        # pack into a single U(x,y,z,3) like your sim output
        U_mode = Array{T}(undef, Nx,Ny,Nz,3)
        @views begin
            U_mode[:,:,:,1] .= Ux_mode
            U_mode[:,:,:,2] .= Uy_mode
            U_mode[:,:,:,3] .= Uz_mode
        end

        # write one file per mode (easy to load and plot later)
        fpath = joinpath(outdir, "POD_mode_$(k).jld2")
        jldsave(fpath;
            Mode_U = U_mode,
            Mode_P = P_mode,
            mode_index = k,
        )
        GC.gc()  # keep memory pressure low across modes
    end
end