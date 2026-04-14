using Flux,CUDA,cuDNN,Random
using JLD2, HDF5
using Statistics
using OutMacro
using LinearAlgebra
using Plots, CairoMakie
using Printf, LaTeXStrings
using ColorSchemes
using Random
include("POD.jl")

Plots.default(
    fontfamily = "Computer Modern",
    linewidth = 1,
    framestyle = :box,
    grid = true,
    left_margin = Plots.Measures.Length(:mm, 5),
    right_margin = Plots.Measures.Length(:mm, 5),
    bottom_margin = Plots.Measures.Length(:mm, 5),
    top_margin = Plots.Measures.Length(:mm, 5),
    titlefontsize = 12,
    legendfontsize = 10,
    tickfontsize = 10,
    labelfontsize = 12,
)
"""
pod_surface_plots(; mode_cutoff, λ1s, λ2s, θs, pod_dir, outdir, backend=:cairo)

For each θ ∈ θs and each mode k=1:mode_cutoff, plot the *raw* POD coefficient a_k
as a 3D surface over (λ₁, λ₂) for Ux, Uy, Uz, and P, and save PDFs.

Assumptions:
- read_POD("POD.jld2"; dir=pod_dir) returns:
    Σ_U::AbstractVector, A_U::Array{<:Real,3}  (nmode × nsnap × 3)
    Σ_P::AbstractVector, A_P::Array{<:Real,2}  (nmode × nsnap)
- snapshot_params(λ1s, λ2s, θs) returns a (3 × nsnap) param matrix whose
  column order matches your snapshot order with λ₂ varying fastest for each λ₁,θ.
"""
function pod_surface_plots(mode_cutoff::Int, λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector; 
    pod_dir::AbstractString, outdir::AbstractString)
    
    mkpath(outdir)
    Σ_U, A_U, Σ_P, A_P = read_POD("POD.jld2"; dir=pod_dir)
    nmode, nsnap, _ = size(A_U)
    r = min(mode_cutoff, nmode)
    X = snapshot_params(λ1s, λ2s, θs)  # 3 × nsnap
    @out X
    λ1_all, λ2_all, θ_all = eachrow(X)
    perθ = length(λ1s)*length(λ2s)

    # Helper: reshape snapshot-aligned vector → (λ1, λ2) grid for a given θ
    to_grid = function(vals::AbstractVector{<:Real}, θval)
        J = findall(i -> θ_all[i] == θval, 1:nsnap)
        @assert length(J) == perθ "Unexpected snapshot count for θ=$(θval)."
        # λ₂ varies fastest → reshape(cols=λ2) then transpose so rows=λ1, cols=λ2
        reshape(vals[J], length(λ2s), length(λ1s))'
    end

    # Field list: (label, extractor)
    fields = [
        ("Ux", i -> view(A_U, :, :, 1)),
        ("Uy", i -> view(A_U, :, :, 2)),
        ("Uz", i -> view(A_U, :, :, 3)),
        ("P",  i -> A_P),
    ]

    # Loop fields → modes → θ
    for (fname, _getter) in fields
        C = _getter(nothing)  # coefficients array for this field
        # C has shape:
        #   U*: (nmode × nsnap)
        #   P : (nmode × nsnap)
        Cmat = fname == "P" ? C : reshape(C, size(C,1), size(C,2))  # ensure 2D

        for k in 1:r
            coeffs_k = vec(Cmat[k, :])  # length == nsnap

            for θv in θs
                Z = to_grid(coeffs_k, θv)  # size == (length(λ1s), length(λ2s))
                @out Z
                fig = Figure(size = (900, 700))
                ax  = Axis3(fig[1,1],
                            xlabel="λ₁", ylabel="λ₂", zlabel="aₖ",
                            title="POD Coefficient — $(fname), k=$(k), θ=$(θv)")

                # 3D surface
                cm = cgrad(:algae, alpha = 0.9)
                CairoMakie.surface!(ax, λ1s, λ2s, Z; colormap = cm, shading = true, transparency = false)
                CairoMakie.wireframe!(ax, λ1s, λ2s, Z;
                color = :black, linewidth = 0.5)

                # --- Overlay the exact data points as dots ---
                # Z is size (length(λ1s), length(λ2s)) with rows=λ1, cols=λ2
                # Column-major vec(Z) orders by λ2 columns, then λ1 rows → match with:
                x_pts = repeat(λ1s, outer=length(λ2s))         # [λ1s..., λ1s..., ...] per λ2
                y_pts = repeat(λ2s, inner=length(λ1s))         # each λ2 repeated for all λ1
                z_pts = vec(Z)

                CairoMakie.scatter!(ax, x_pts, y_pts, z_pts;
                        marker = :circle,
                        markersize = 8,
                        color = :black)  # or :white if your surface is dark

                # (Optional) make the surface slightly see-through so dots pop out
                # translate!(ax.scene, 0, 0, 0)  # not required, just here as a reminder
                # surface!(ax, λ1s, λ2s, Z; transparency=true) # alternative approach

                ax.xticks = (λ1s, string.(λ1s))
                ax.yticks = (λ2s, string.(λ2s))
                hidespines!(ax)

                save(joinpath(outdir, "PODcoeff_$(fname)_k$(k)_theta_$(θv).pdf"), fig)
            end
        end
    end
    println("✅ Saved POD coefficient surface PDFs to: $outdir")
end

"""
pod_surface_sum_over_modes(; mode_cutoff, λ1s, λ2s, θs, pod_dir, outdir)

For each field (Ux,Uy,Uz,P) and each θ, plots ONE 3D surface whose Z-value is
the plain sum over modes 1:r of a_k (no norms). Also overlays the exact data points.
"""
function pod_surface_sum_over_modes( mode_cutoff::Int,
    λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector;
    pod_dir::AbstractString, outdir::AbstractString)

    mkpath(outdir)

    # --- Load POD ---
    Σ_U, A_U, Σ_P, A_P = read_POD("POD.jld2"; dir=pod_dir)
    nmode, nsnap, _ = size(A_U)
    r = min(mode_cutoff, nmode)

    # --- Params in snapshot order (must match your snapshot construction order) ---
    X = snapshot_params(λ1s, λ2s, θs)   # 3 × nsnap
    λ1_all, λ2_all, θ_all = eachrow(X)
    perθ = length(λ1s) * length(λ2s)

    # reshape helper: vector (nsnap) -> (length(λ1s), length(λ2s)) for a given θ
    to_grid = function(vals::AbstractVector{<:Real}, θval)
        J = findall(i -> θ_all[i] == θval, 1:nsnap)
        @assert length(J) == perθ "Unexpected snapshot count for θ=$(θval)"
        reshape(vals[J], length(λ2s), length(λ1s))'  # rows=λ1, cols=λ2
    end

    fields = [
        ("Ux", view(A_U, :, :, 1)),   # nmode × nsnap
        ("Uy", view(A_U, :, :, 2)),
        ("Uz", view(A_U, :, :, 3)),
        ("P",  A_P)                   # nmode × nsnap
    ]

    for (fname, C) in fields
        C2 = reshape(C, size(C,1), size(C,2))   # ensure 2D (nmode × nsnap)

        # --- Sum over modes 1:r (plain algebraic sum; no norms) ---
        coeffs_sum = vec(sum(@view C2[1:r, :]; dims=1))  # length == nsnap

        for θv in θs
            Z = to_grid(coeffs_sum, θv)

            fig = Figure(size=(900, 700))
            ax  = Axis3(fig[1,1],
                        xlabel="λ₁", ylabel="λ₂", zlabel="∑_{k=1}^{r} a_k",
                        title="Sum of POD Coefficients — $(fname), θ=$(θv), r=$(r)")

            cm = cgrad(:algae, alpha = 0.9)
            CairoMakie.surface!(ax, λ1s, λ2s, Z; colormap = cm, shading = true, transparency = false)
            CairoMakie.wireframe!(ax, λ1s, λ2s, Z;
            color = :black, linewidth = 0.5)

            # overlay the actual data points
            x_pts = repeat(λ1s, outer=length(λ2s))
            y_pts = repeat(λ2s, inner=length(λ1s))
            z_pts = vec(Z)
            CairoMakie.scatter!(ax, x_pts, y_pts, z_pts; marker=:circle, markersize=10,
                     color=:black, strokecolor=:black, strokewidth=0.5)

            ax.xticks = (λ1s, string.(λ1s))
            ax.yticks = (λ2s, string.(λ2s))
            hidespines!(ax)

            save(joinpath(outdir, "PODcoeffSum_$(fname)_theta_$(θv).pdf"), fig)
        end
    end

    println("✅ Saved summed-coefficient surfaces to: $outdir")
end

# λ1 major, then λ2, then θ (θ is fastest)
function snapshot_col_index(λ1s::AbstractVector{<:Integer},
                            λ2s::AbstractVector{<:Integer},
                            θs::AbstractVector{<:Integer},
                            λ1::Integer, λ2::Integer, θ::Integer)
    i1 = findfirst(==(λ1), λ1s)
    i2 = findfirst(==(λ2), λ2s)
    iθ = findfirst(==(θ),  θs)
    @assert i1 !== nothing && i2 !== nothing && iθ !== nothing "Params not in training grid"

    n2 = length(λ2s)
    nθ = length(θs)
    return ((i1 - 1) * n2 + (i2 - 1)) * nθ + iθ
end

# --- Helper: parity (y vs x) for a coefficient vector ---
function plot_coeff_parity!(x_pred::AbstractVector, y_true::AbstractVector; dir::AbstractString, fname::AbstractString, q::Union{Symbol,String}=:Ux)
    @assert length(x_pred) == length(y_true)
    qsym = q isa Symbol ? q : Symbol(q)
    labels = (
        Ux = (L"a_{\star,r}^{(\overline{u}_{x})}", L"\a_{r}^{(\overline{u}_{x})}"),
        Uy = (L"a_{\star,r}^{(\overline{u}_{y})}", L"\a_{r}^{(\overline{u}_{y})}"),
        Uz = (L"a_{\star,r}^{(\overline{u}_{z})}", L"\a_{r}^{(\overline{u}_{z})}"),
        P  = (L"a_{\star,r}^{(\overline{P})}", L"\a_{r}^{(\overline{P})}")
    )
    hasproperty(labels, qsym) || throw(ArgumentError("q must be one of $(propertynames(labels))"))
    xlab, ylab = getproperty(labels, qsym)
    p = Plots.scatter(x_pred, y_true;
        xlabel = xlab,
        ylabel = ylab,
        size = (600,600),
        framestyle = :box, grid = true, legend = false)
    lo = min(minimum(x_pred), minimum(y_true))
    hi = max(maximum(x_pred), maximum(y_true))
    Plots.plot!(p, [lo, hi], [lo, hi]; label = "y = x", lw = 2)

    # R² annotation
    ss_res = sum((y_true .- x_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    R2 = 1 - ss_res/ss_tot
    Plots.annotate!(p, lo, hi, Plots.text(@sprintf("R² = %.3f", R2), 10, :left, :top))
    savefig(p, joinpath(dir, fname))
end

# ---------------------------------------------------
# 0) Helper to move the surrogate to GPU if available 
# ---------------------------------------------------
device = gpu_device()
dev(x) = CUDA.has_cuda() ? cu(x) : x

# --------------------------------
# 0) Helpers for error calculation
# --------------------------------
_mean_sq(x) = max(eps(Float32), mean(abs2.(x)))
se(y, ŷ)  = (y .- ŷ).^2
mse(y, ŷ)  = mean((y .- ŷ).^2)
ae(y, ŷ)  = abs.(y .- ŷ)

# ---------------------
# 1) Truncate POD modes  
# ---------------------
function truncate(mode_cutoff::Int; dir::AbstractString)
    Σ_U, A_U, Σ_P, A_P = read_POD("POD.jld2"; dir=dir)
    nmode, nsnap, _ = size(A_U)
    @info "Loaded POD" nmode nsnap
    r = min(mode_cutoff, nmode)
    if r==nmode
        println("Did not truncate modes: N mode = $(r)")
    else
        println("Truncated modes: N mode = $(r)")
    end
    A_Ur = A_U[1:r,:,:]          
    A_Pr = A_P[1:r,:]          
    Σ_Ur = Σ_U[1:r,:,:]
    Σ_Pr = Σ_P[1:r,:]
    C_Ux = A_Ur[:,:,1]; C_Uy = A_Ur[:,:,2]; C_Uz = A_Ur[:,:,3]
    C_P  = A_Pr  
    return r, nsnap, C_Ux, C_Uy, C_Uz, C_P
end

# -------------------------------
# 2) Create global marameter keys
# -------------------------------
function snapshot_params(λ1s, λ2s, θs)
    P = NTuple{3,Float32}[]
    for λ1 in λ1s
        for λ2 in λ2s
            for θ in θs
                push!(P, (Float32(λ1), Float32(λ2), Float32(θ)))
            end
        end
    end
    reduce(hcat, (collect(p) for p in P))  # 3×nsnap
end

# -----------------------------------------------------
# 3) Split the data into "training" and validation sets 
# -----------------------------------------------------
function split_data(training::Float64, nsnap::Int, Xparams::AbstractArray, C_Ux::AbstractArray, C_Uy::AbstractArray, C_Uz::AbstractArray, C_P::AbstractArray)
    Random.seed!(123)
    idx     = shuffle!(collect(1:nsnap))
    ntrain  = Int(floor(training*nsnap))
    itrain  = idx[1:ntrain]
    ival    = idx[ntrain+1:end]

    Xtr, Xval = Xparams[:, itrain], Xparams[:, ival]
    C_Ux_tr, C_Ux_val = C_Ux[:, itrain], C_Ux[:, ival]
    C_Uy_tr, C_Uy_val = C_Uy[:, itrain], C_Uy[:, ival]
    C_Uz_tr, C_Uz_val = C_Uz[:, itrain], C_Uz[:, ival]
    C_P_tr,  C_P_val  = C_P[:,  itrain], C_P[:,  ival]

    Xtr = dev(Xtr); Xval = dev(Xval)
    C_Ux_tr = dev(C_Ux_tr); C_Ux_val = dev(C_Ux_val)
    C_Uy_tr = dev(C_Uy_tr); C_Uy_val = dev(C_Uy_val)
    C_Uz_tr = dev(C_Uz_tr); C_Uz_val = dev(C_Uz_val)
    C_P_tr  = dev(C_P_tr);  C_P_val  = dev(C_P_val)
    return Xtr, Xval, C_Ux_tr, C_Ux_val, C_Uy_tr, C_Uy_val, C_Uz_tr, C_Uz_val, C_P_tr, C_P_val
end

function split_data_keep_edges(training::Float64, nsnap::Int, Xparams::AbstractArray, C_Ux::AbstractArray, C_Uy::AbstractArray, C_Uz::AbstractArray, C_P::AbstractArray;seed=123)

    @assert size(Xparams, 2) == nsnap "nsnap must equal size(Xparams, 2)"
    @assert nsnap ≥ 4 "Need at least 4 snapshots to keep first/last two in training."

    # 1) Indices forced into the training set (1-based in Julia)
    forced_idx = [1, 2, nsnap-1, nsnap]
    n_forced = length(forced_idx)

    # 2) Indices eligible for being train/val (the "inner" snapshots)
    other_idx = [i for i in 1:nsnap if i ∉ forced_idx]

    # 3) Compute nominal train/val sizes as if we did a plain random split
    n_train_nominal = Int(floor(training * nsnap))
    n_val_nominal   = nsnap - n_train_nominal

    @assert n_val_nominal ≤ length(other_idx) "Not enough inner snapshots for desired validation size."

    # 4) Shuffle only the inner indices, then take exactly n_val_nominal for validation
    Random.seed!(seed)
    shuffled_other = shuffle!(copy(other_idx))

    ival = shuffled_other[1:n_val_nominal]                    # validation indices
    itrain_rest = shuffled_other[n_val_nominal+1:end]         # extra training (besides edges)

    # Full training indices = forced edges + the remaining inner ones
    itrain = vcat(forced_idx, itrain_rest)

    # (Optional) sort indices if you like stable ordering; not required for ML
    sort!(itrain)
    sort!(ival)

    # 5) Slice data: all arrays use columns = snapshots
    Xtr, Xval = Xparams[:, itrain], Xparams[:, ival]
    C_Ux_tr, C_Ux_val = C_Ux[:, itrain], C_Ux[:, ival]
    C_Uy_tr, C_Uy_val = C_Uy[:, itrain], C_Uy[:, ival]
    C_Uz_tr, C_Uz_val = C_Uz[:, itrain], C_Uz[:, ival]
    C_P_tr,  C_P_val  = C_P[:,  itrain], C_P[:,  ival]

    # 6) Move to device (GPU/CPU) as in your original function
    Xtr = dev(Xtr); Xval = dev(Xval)
    C_Ux_tr = dev(C_Ux_tr); C_Ux_val = dev(C_Ux_val)
    C_Uy_tr = dev(C_Uy_tr); C_Uy_val = dev(C_Uy_val)
    C_Uz_tr = dev(C_Uz_tr); C_Uz_val = dev(C_Uz_val)
    C_P_tr  = dev(C_P_tr);  C_P_val  = dev(C_P_val)

    return Xtr, Xval, C_Ux_tr, C_Ux_val, C_Uy_tr, C_Uy_val, C_Uz_tr, C_Uz_val, C_P_tr,  C_P_val
end

# -------------------------
# 4) Record error histories
#--------------------------
function record!(H::Dict; ltr, ytr, ŷval, yval)
    mtr = Float32(ltr)
    push!(H[:mse_tr],  mtr)
    push!(H[:rmse_tr], sqrt(mtr))
    push!(H[:nmse_tr], mtr / _mean_sq(ytr))

    mval = Float32(Flux.mse(ŷval, yval))
    push!(H[:mse_val],  mval)
    push!(H[:rmse_val], sqrt(mval))
    push!(H[:nmse_val], mval / _mean_sq(yval))
end

# ------------------------
# Shared MLP constructor
# ------------------------
make_head(outdim) = Chain(
    Dense(3 => 64, gelu),
    Dense(64 => 64, gelu),
    Dense(64 => outdim),
) |> dev

# ------------------------
# Metric history factory
# ------------------------
make_hist(_) = Dict(
    :mse_tr   => Float32[],
    :rmse_tr  => Float32[],
    :nmse_tr  => Float32[],
    :mse_val  => Float32[],
    :rmse_val => Float32[],
    :nmse_val => Float32[],
)

# Cut all metric arrays to a given length
function trim_hist!(hist::Dict{Symbol,Vector{Float32}}, upto::Int)
    for v in values(hist)
        resize!(v, upto)
    end
end

# ------------------------
# Core single-head trainer
# ------------------------
function _train_one_head(name::AbstractString, outdim::Int,
                         Xtr::AbstractArray, Xval::AbstractArray,
                         C_tr::AbstractArray, C_val::AbstractArray,
                         nepoch::Int=1500, patience::Int=50, min_delta::Float32=1f-6)

    # Build model & optimizer
    model = make_head(outdim)
    opt   = Flux.setup(Flux.Adam(1e-3), model)

    # History
    hist = make_hist(nepoch)

    # Early-stop state
    best_val  = Inf32
    best_ep   = 0
    noimp     = 0
    best_model_cpu = nothing
    best_opt_cpu   = nothing
    stopped_epoch  = nepoch  # will overwrite when we actually stop

    for epoch in 1:nepoch
        # ---- train step
        L, gs = Flux.withgradient(model) do m
            Flux.mse(m(Xtr), C_tr)
        end
        Flux.update!(opt, model, gs[1])

        # ---- validation & record
        ŷval = model(Xval)
        record!(hist; ltr=L, ytr=C_tr, ŷval=ŷval, yval=C_val)

        val_mse = Float32(Flux.mse(ŷval, C_val))

        # ---- early-stopping check
        if val_mse < (best_val - min_delta)
            best_val  = val_mse
            best_ep   = epoch
            noimp     = 0
            best_model_cpu = cpu(deepcopy(model))
            best_opt_cpu   = cpu(deepcopy(opt))
        else
            noimp += 1
            if noimp >= patience
                # restore best snapshot & stop
                if best_model_cpu !== nothing
                    model = dev(best_model_cpu)
                    opt   = dev(best_opt_cpu)
                end
                stopped_epoch = best_ep
                trim_hist!(hist, best_ep)   # <<< keep only up to best epoch
                @info "[$(name)] Early-stopped at epoch=$(epoch); best @ epoch=$(best_ep), best val-MSE=$(round(best_val,digits=6))"
                break
            end
        end

        # optional progress log
        if epoch % 25 == 0
            @info "[$(name)] epoch=$(epoch)   MSE(tr)=$(round(Float32(L),digits=6))   MSE(val)=$(round(val_mse,digits=6))"
        end
    end

    # If never hit patience, still ensure we end at best snapshot
    if best_model_cpu !== nothing
        model = dev(best_model_cpu)
        opt   = dev(best_opt_cpu)
        stopped_epoch = (best_ep == 0 ? nepoch : best_ep)
        trim_hist!(hist, stopped_epoch)   # <<< trim to best
    end

    return hist, model, opt, stopped_epoch
end

train_MLP_Ux(nepoch::Int, r::Int, Xtr::AbstractArray, Xval::AbstractArray, 
    C_Ux_tr::AbstractArray, C_Ux_val::AbstractArray, patience::Int=50, min_delta::Float32=0f0) = 
_train_one_head("Ux", r, Xtr, Xval, C_Ux_tr, C_Ux_val, nepoch, patience, min_delta)

train_MLP_Uy(nepoch::Int, r::Int, Xtr::AbstractArray, Xval::AbstractArray,
    C_Uy_tr::AbstractArray, C_Uy_val::AbstractArray, patience::Int=50, min_delta::Float32=0f0) = 
_train_one_head("Uy", r, Xtr, Xval, C_Uy_tr, C_Uy_val, nepoch, patience, min_delta)


train_MLP_Uz(nepoch::Int, r::Int, Xtr::AbstractArray, Xval::AbstractArray,
    C_Uz_tr::AbstractArray, C_Uz_val::AbstractArray, patience::Int=50, min_delta::Float32=0f0) =
_train_one_head("Uz", r, Xtr, Xval, C_Uz_tr, C_Uz_val, nepoch, patience, min_delta)


train_MLP_P(nepoch::Int, r::Int, Xtr::AbstractArray, Xval::AbstractArray,
    C_P_tr::AbstractArray, C_P_val::AbstractArray, patience::Int=50, min_delta::Float32=0f0) = 
_train_one_head("P", r, Xtr, Xval, C_P_tr, C_P_val, nepoch, patience, min_delta)

# ----------------------
# 6) Save the surrogates 
# ----------------------
function save_surrogates(models, opts; dir::AbstractString, fname::AbstractString)
    path = joinpath(dir, fname)
    m_cpu = Flux.cpu(models)
    opt_cpu = Flux.cpu(opts)
    @save path m_cpu opt_cpu
    return nothing
end

# ------------------
# 7) Load surrogates 
# ------------------
function load_surrogates(;dir::AbstractString, fname::AbstractString)
    path = joinpath(dir, fname)
    @load path m_cpu
    m_gpu = dev(m_cpu)
    return m_gpu
end

# ---------------------------------
# 7) Load surrogates and optimizers 
# ---------------------------------
function load_surrogates_optimizers(;dir::AbstractString, fname::AbstractString)
    path = joinpath(dir, fname)
    @load path m_cpu opt_cpu
    m_gpu = dev(m_cpu)
    opt_gpu = dev(opt_cpu)
    return m_gpu, opt_gpu
end

# ------------------------------------------------------------------------------
# 8) Use the surrogates (predict U and P coefficients for a new parameter tuple)
# ------------------------------------------------------------------------------
function predict_coeffs(model, λ1, λ2, θ)
    x = dev(reshape(Float32[λ1,λ2,θ], 3, 1))
    a = model(x)            
    a = Array(a)
    return reshape(a, :)
end

# ------------------------------------------------------------------------
# 9) Reconstruct fields from predicted coeffs
#     (requires that you saved modes via recover_POD_modes)
#     Each "POD_mode_k.jld2" has Mode_U (Nx,Ny,Nz,3) and Mode_P (Nx,Ny,Nz)
# ------------------------------------------------------------------------
function reconstruct_fields(mode_cutoff::Union{Int,Float64}, aUx::AbstractVector, aUy::AbstractVector, aUz::AbstractVector, aP::AbstractVector; dir::AbstractString)
    obj1 = jldopen(joinpath(dir, @sprintf("POD_mode_%d.jld2", 1)))
    Mode_U = obj1["Mode_U"]; Mode_P = obj1["Mode_P"]
    close(obj1)
    Nx,Ny,Nz,_ = size(Mode_U)
    Urec = zeros(Float32, Nx,Ny,Nz,3)
    Prec = zeros(Float32, Nx,Ny,Nz)
    for k in 1:mode_cutoff
        fn = joinpath(dir, @sprintf("POD_mode_%d.jld2", k))
        obj = jldopen(fn)
        Uk = obj["Mode_U"]; Pk = obj["Mode_P"]
        close(obj)
        @inbounds begin
            Urec[:,:,:,1] .+= aUx[k]*Uk[:,:,:,1]
            Urec[:,:,:,2] .+= aUy[k]*Uk[:,:,:,2]
            Urec[:,:,:,3] .+= aUz[k]*Uk[:,:,:,3]
            Prec          .+= aP[k]*Pk
        end
    end
    return Urec, Prec
end

_lbl_double = (
    "streamwise velocity",
    "cross-stream velocity",
    "spanwise velocity",
    "pressure"
)

label_for_double(q::Integer) = begin
    @assert 1 ≤ q ≤ 4 "q must be 1..4"
    t = _lbl_double[q]
    lab1 = string("Ground truth ", t)
    lab2 = "Predicted $(t)"  # interpolation inserts a space you put before it
    lab1 = join(["Ground truth", t], " ")
    return lab1, lab2
end

function pod_surface_double(r, nsnap, A, C, q, λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector; outdir::AbstractString, fname::AbstractString)
    mkpath(outdir)
    lab1, lab2 = label_for_double(q)
    # --- Params in snapshot order (must match your snapshot construction order) ---
    X = snapshot_params(λ1s, λ2s, θs)   # 3 × nsnap
    λ1_all, λ2_all, θ_all = eachrow(X)
    perθ = length(λ1s) * length(λ2s)

    # reshape helper: vector (nsnap) -> (length(λ1s), length(λ2s)) for a given θ
    to_grid = function(vals::AbstractVector{<:Real}, θval)
        J = findall(i -> θ_all[i] == θval, 1:nsnap)
        @assert length(J) == perθ "Unexpected snapshot count for θ=$(θval)"
        reshape(vals[J], length(λ2s), length(λ1s))'  # rows=λ1, cols=λ2
    end

    C2 = reshape(C, size(C,1), size(C,2))
    A2 = reshape(A, size(A,1), size(A,2))    

    C2_sum = vec(sum(@view C2[1:r, :]; dims=1))
    A2_sum = vec(sum(@view A2[1:r, :]; dims=1))  

    for θv in θs
        Z_C2 = to_grid(C2_sum, θv)
        Z_A2 = to_grid(A2_sum, θv)

        fig = Figure(size=(1000, 700))
        ax = Axis3(
            fig[1, 1];
            title  = latexstring("\$\\theta = $θv\\degree\$"),
            titlesize = 22,
            xlabel = L"\lambda_{1}",  ylabel = L"\lambda_{2}",  zlabel = L"\sum_{i=1}^{r} a_{i}^{(q)}",
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsize = 14, yticklabelsize = 14, zticklabelsize = 14,
        )
        c_C2 = cgrad(:algae, alpha = 1)
        c_A2 = cgrad(:amp, alpha = 0.9)
        CairoMakie.surface!(ax, λ1s, λ2s, Z_C2; label = lab1, colormap = c_C2, shading = true, transparency = false)
        CairoMakie.surface!(ax, λ1s, λ2s, Z_A2; label = lab2, colormap = c_A2, shading = true, transparency = false)
        axislegend(ax; position = :rt, labelsize = 18)  # show the legend on the axis
        CairoMakie.wireframe!(ax, λ1s, λ2s, Z_C2; color = :green, linewidth = 0.5)
        CairoMakie.wireframe!(ax, λ1s, λ2s, Z_A2; color = :red, linewidth = 0.5)

        # overlay the actual data points
        x_pts = repeat(λ1s, outer=length(λ2s))
        y_pts = repeat(λ2s, inner=length(λ1s))
        z_pts_C2 = vec(Z_C2)
        z_pts_A2 = vec(Z_A2)
        CairoMakie.scatter!(ax, x_pts, y_pts, z_pts_C2; marker=:circle, markersize=10, color=:green, strokecolor=:black, strokewidth=0.5)
        CairoMakie.scatter!(ax, x_pts, y_pts, z_pts_A2; marker=:circle, markersize=10, color=:red, strokecolor=:black, strokewidth=0.5)
        ax.xticks = (λ1s, string.(λ1s))
        ax.yticks = (λ2s, string.(λ2s))
        hidespines!(ax)
        name = string(fname, θv, ".pdf")
        save(joinpath(outdir, name), fig)
    end
end

_lbl_single = (
    "streamwise velocity",
    "cross-stream velocity",
    "spanwise velocity",
    "pressure"
)

label_for_single(q::Integer) = begin
    @assert 1 ≤ q ≤ 4 "q must be 1..4"
    t = _lbl_single[q]
    lab1 = string("Ground truth ", t)
    lab1 = join(["Ground truth", t], " ")
    return lab1
end

function pod_surface_single(r, nsnap, A, q, λ1s::AbstractVector, λ2s::AbstractVector, θs::AbstractVector; outdir::AbstractString, fname::AbstractString)
    mkpath(outdir)
    lab1 = label_for_single(q)
    # --- Params in snapshot order (must match your snapshot construction order) ---
    X = snapshot_params(λ1s, λ2s, θs)   # 3 × nsnap
    λ1_all, λ2_all, θ_all = eachrow(X)
    perθ = length(λ1s) * length(λ2s)

    # reshape helper: vector (nsnap) -> (length(λ1s), length(λ2s)) for a given θ
    to_grid = function(vals::AbstractVector{<:Real}, θval)
        J = findall(i -> θ_all[i] == θval, 1:nsnap)
        @assert length(J) == perθ "Unexpected snapshot count for θ=$(θval)"
        reshape(vals[J], length(λ2s), length(λ1s))'  # rows=λ1, cols=λ2
    end

    A2 = reshape(A, size(A,1), size(A,2))    
    A2_sum = vec(sum(@view A2[1:r, :]; dims=1))  

    for θv in θs
        Z_A2 = to_grid(A2_sum, θv)

        fig = Figure(size=(1000, 700))
        ax = Axis3(
            fig[1, 1];
            title  = latexstring("\$\\theta = $θv\\degree\$"),
            titlesize = 22,
            xlabel = L"\lambda_{1}",  ylabel = L"\lambda_{2}",  zlabel = L"\sum_{i=1}^{r} a_{i}^{(q)}",
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsize = 14, yticklabelsize = 14, zticklabelsize = 14,
        )
        c_A2 = cgrad(:algae, alpha = 1)
        CairoMakie.surface!(ax, λ1s, λ2s, Z_A2; label = lab1, colormap = c_A2, shading = true, transparency = false)
        axislegend(ax; position = :rt, labelsize = 18)
        CairoMakie.wireframe!(ax, λ1s, λ2s, Z_A2; color = :green, linewidth = 0.5)

        # overlay the actual data points
        x_pts = repeat(λ1s, outer=length(λ2s))
        y_pts = repeat(λ2s, inner=length(λ1s))
        z_pts_A2 = vec(Z_A2)
        CairoMakie.scatter!(ax, x_pts, y_pts, z_pts_A2; marker=:circle, markersize=10, color=:green, strokecolor=:black, strokewidth=0.5)
        ax.xticks = (λ1s, string.(λ1s))
        ax.yticks = (λ2s, string.(λ2s))
        hidespines!(ax)
        name = string(fname, θv, ".pdf")
        save(joinpath(outdir, name), fig)
    end
end

function read_coeff(fname::AbstractString; dir::AbstractString="")
    jldopen(joinpath(dir, fname)) do f
        return f["aUx_pred"], f["aUy_pred"], f["aUz_pred"], f["aP_pred"], f["pred_params"]
    end
end

function read_recovered_fields(fname::AbstractString; dir::AbstractString="")
    jldopen(joinpath(dir, fname)) do f
        return f["P_rec"], f["U_rec"]
    end
end






