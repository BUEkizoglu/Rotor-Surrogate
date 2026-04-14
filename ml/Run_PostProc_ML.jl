include("ML.jl")

PODdir  = "/ml/data/rotor_BiotSimulation/POD/"
recdir  = "/ml/data/rotor_BiotSimulation/recovered/"
datadir = "/sims/data/rotor_BiotSimulation/"
surrogatedir = "/ml/data/rotor_BiotSimulation/surrogate/"
outdir = "/ml/tex/mode_coefficients/"
λ1s = [3,4,5]
λ2s = [3,4,5]
θs  = [0,15,30]
λ1_recs = [3,4,5]
λ2_recs = [3,4,5]
θ_recs  = [0,15,30]
mode_cutoff = 5
training = 0.8
nepoch = 5000
epochs = 1:nepoch
train = false
plot_err = true
recover_fields = true
field_err = true
z_vals = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5]
clims_p = (-3,3)
ustep = 0.25
pstep = 0.25

pod_path = joinpath(PODdir, "POD.jld2")
out_path = joinpath(PODdir, "POD.h5")

Sigma_U, A_U, Sigma_P, A_P = read_POD("POD.jld2"; dir=PODdir)
r, nsnap, C_Ux, C_Uy, C_Uz, C_P = truncate(mode_cutoff; dir=PODdir)

isfile(out_path) && rm(out_path; force=true)
h5open(out_path, "w") do f   # "w" = create or TRUNCATE
    # (Tip: ASCII names are friendlier for Python: Sigma_U instead of Σ_U)
    write(f, "Sigma_U", Sigma_U)
    write(f, "A_U", A_U)
    write(f, "Sigma_P", Sigma_P)
    write(f, "A_P", A_P)
end
println("✅ Saved HDF5 to: ", out_path)
Xparams = snapshot_params(λ1s, λ2s, θs)
Xtr, Xval, C_Ux_tr, C_Ux_val, C_Uy_tr, C_Uy_val, C_Uz_tr, C_Uz_val, C_P_tr, C_P_val = split_data_keep_edges(training,nsnap,Xparams,C_Ux,C_Uy,C_Uz,C_P;seed=2)
@out C_Ux_tr
@out C_Ux_val

function main()
    println("💾 Loading POD coefficients")
    r, nsnap, C_Ux, C_Uy, C_Uz, C_P = truncate(mode_cutoff; dir=PODdir)
        # store all predictions (every λ1_rec×λ2_rec×θ_rec)
    Nmax = length(λ1_recs) * length(λ2_recs) * length(θ_recs)
    T    = eltype(C_Ux)             # or Float32 / Float64
    aUx_pred = Matrix{T}(undef, r, Nmax)
    aUy_pred = Matrix{T}(undef, r, Nmax)
    aUz_pred = Matrix{T}(undef, r, Nmax)
    aP_pred  = Matrix{T}(undef, r, Nmax)
    pred_params = Vector{NTuple{3,Int}}(undef, Nmax)  # (λ1,λ2,θ) per row
    col = 0
    if train
        Xparams = snapshot_params(λ1s, λ2s, θs)
        @out Xparams
        println("🧮 Organizing training and validation set")
        Xtr, Xval, C_Ux_tr, C_Ux_val, C_Uy_tr, C_Uy_val, C_Uz_tr, C_Uz_val, C_P_tr, C_P_val = split_data_keep_edges(training,nsnap,Xparams,C_Ux,C_Uy,C_Uz,C_P;seed=2)
        @out Xtr
        @out Xval
        println("🧮 Training surrogates")
        histUx, mUx, optUx, epUx = train_MLP_Ux(nepoch, r, Xtr, Xval, C_Ux_tr, C_Ux_val, 50, 0f0)
        histUy, mUy, optUy, epUy = train_MLP_Uy(nepoch, r, Xtr, Xval, C_Uy_tr, C_Uy_val, 50, 0f0)
        histUz, mUz, optUz, epUz = train_MLP_Uz(nepoch, r, Xtr, Xval, C_Uz_tr, C_Uz_val, 50, 0f0)
        histP,  mP,  optP,  epP  = train_MLP_P(nepoch, r, Xtr, Xval, C_P_tr,  C_P_val, 50, 0f0)
        if plot_err
                epochs = 1:epUx
                println("🧮 Plotting error histories")
                Ux_rmse = Plots.plot(epochs, histUx[:rmse_tr], label=latexstring("\$\\overline{u}_{x}\$ RMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Ux_rmse, epochs, histUx[:rmse_val], label=latexstring("\$\\overline{u}_{x}\$ RMSE (validation)"), xlabel="epoch", ylabel="RMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Ux_rmse, string(@__DIR__) * "/tex/errors/Ux_rmse.pdf")
                
                Ux_mse = Plots.plot(epochs, histUx[:mse_tr], label=latexstring("\$\\overline{u}_{x}\$ MSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Ux_mse, epochs, histUx[:mse_val], label=latexstring("\$\\overline{u}_{x}\$ MSE (validation)"), xlabel="epoch", ylabel="MSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Ux_mse, string(@__DIR__) * "/tex/errors/Ux_mse.pdf")

                Ux_nmse = Plots.plot(epochs, histUx[:nmse_tr], label=latexstring("\$\\overline{u}_{x}\$ NMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Ux_nmse, epochs, histUx[:nmse_val], label=latexstring("\$\\overline{u}_{x}\$ NMSE (validation)"), xlabel="epoch", ylabel="NMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Ux_nmse, string(@__DIR__) * "/tex/errors/Ux_nmse.pdf")

                epochs = 1:epUy
                Uy_rmse = Plots.plot(epochs, histUy[:rmse_tr], label=latexstring("\$\\overline{u}_{y}\$ RMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uy_rmse, epochs, histUy[:rmse_val], label=latexstring("\$\\overline{u}_{y}\$ RMSE (validation)"), xlabel="epoch", ylabel="RMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uy_rmse, string(@__DIR__) * "/tex/errors/Uy_rmse.pdf")

                Uy_mse = Plots.plot(epochs, histUy[:mse_tr], label=latexstring("\$\\overline{u}_{y}\$ MSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uy_mse, epochs, histUy[:mse_val], label=latexstring("\$\\overline{u}_{y}\$ MSE (validation)"), xlabel="epoch", ylabel="MSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uy_mse, string(@__DIR__) * "/tex/errors/Uy_mse.pdf")

                Uy_nmse = Plots.plot(epochs, histUy[:nmse_tr], label=latexstring("\$\\overline{u}_{y}\$ NMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uy_nmse, epochs, histUy[:nmse_val], label=latexstring("\$\\overline{u}_{y}\$ NMSE (validation)"), xlabel="epoch", ylabel="NMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uy_nmse, string(@__DIR__) * "/tex/errors/Uy_nmse.pdf")

                epochs = 1:epUz
                Uz_rmse = Plots.plot(epochs, histUz[:rmse_tr], label=latexstring("\$\\overline{u}_{z}\$ RMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uz_rmse, epochs, histUz[:rmse_val], label=latexstring("\$\\overline{u}_{z}\$ RMSE (validation)"), xlabel="epoch", ylabel="RMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uz_rmse, string(@__DIR__) * "/tex/errors/Uz_rmse.pdf")

                Uz_mse = Plots.plot(epochs, histUz[:mse_tr], label=latexstring("\$\\overline{u}_{z}\$ MSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uz_mse, epochs, histUz[:mse_val], label=latexstring("\$\\overline{u}_{z}\$ MSE (validation)"), xlabel="epoch", ylabel="MSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uz_mse, string(@__DIR__) * "/tex/errors/Uz_mse.pdf")

                Uz_nmse = Plots.plot(epochs, histUz[:nmse_tr], label=latexstring("\$\\overline{u}_{z}\$ NMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(Uz_nmse, epochs, histUz[:nmse_val], label=latexstring("\$\\overline{u}_{z}\$ NMSE (validation)"), xlabel="epoch", ylabel="NMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(Uz_nmse, string(@__DIR__) * "/tex/errors/Uz_nmse.pdf")

                epochs = 1:epP
                P_rmse = Plots.plot(epochs, histP[:rmse_tr], label=latexstring("\$\\overline{P}\$ RMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(P_rmse, epochs, histP[:rmse_val], label=latexstring("\$\\overline{P}\$ RMSE (validation)"), xlabel="epoch", ylabel="RMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(P_rmse, string(@__DIR__) * "/tex/errors/P_rmse.pdf")

                P_mse = Plots.plot(epochs, histP[:mse_tr], label=latexstring("\$\\overline{P}\$ MSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(P_mse, epochs, histP[:mse_val], label=latexstring("\$\\overline{P}\$ MSE (validation)"), xlabel="epoch", ylabel="MSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(P_mse, string(@__DIR__) * "/tex/errors/P_mse.pdf")

                P_nmse = Plots.plot(epochs, histP[:nmse_tr], label=latexstring("\$\\overline{P}\$ NMSE (training)"), size=(600,600), xlims=(0,epochs[end]), linewidth=2)
                Plots.plot!(P_nmse, epochs, histP[:nmse_val], label=latexstring("\$\\overline{P}\$ NMSE (validation)"), xlabel="epoch", ylabel="NMSE", xlims=(0,epochs[end]), linewidth=2)
                savefig(P_nmse, string(@__DIR__) * "/tex/errors/P_nmse.pdf")
                println("✅ Plotting done!")
        end
        println("💾 Saving surrogates")
        save_surrogates(mUx, optUx; dir=surrogatedir, fname="mUx")
        save_surrogates(mUy, optUy; dir=surrogatedir, fname="mUy")
        save_surrogates(mUz, optUz; dir=surrogatedir, fname="mUz")
        save_surrogates(mP, optP; dir=surrogatedir, fname="mP")
        println("✅ Training done!")
    else
        println("💾 Loading surrogates")
        mUx = load_surrogates(;dir=surrogatedir, fname="mUx")
        mUy = load_surrogates(;dir=surrogatedir, fname="mUy")
        mUz = load_surrogates(;dir=surrogatedir, fname="mUz")
        mP = load_surrogates(;dir=surrogatedir, fname="mP")
        println("✅ Done!")
    end

    println("🧮 Postprocessing")
    all_pred = Dict(:aUx=>Float32[], :aUy=>Float32[], :aUz=>Float32[], :aP=>Float32[])
    all_true = Dict(:aUx=>Float32[], :aUy=>Float32[], :aUz=>Float32[], :aP=>Float32[])
    for λ1_rec in λ1_recs
        for λ2_rec in λ2_recs
            for θ_rec in θ_recs
                println("🧮 Offline prediction: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                aUx = predict_coeffs(mUx, λ1_rec, λ2_rec, θ_rec) 
                aUy = predict_coeffs(mUy, λ1_rec, λ2_rec, θ_rec)
                aUz = predict_coeffs(mUz, λ1_rec, λ2_rec, θ_rec)
                aP = predict_coeffs(mP, λ1_rec, λ2_rec, θ_rec)
                col += 1
                copyto!(view(aUx_pred,:,col), aUx)
                copyto!(view(aUy_pred,:,col), aUy)
                copyto!(view(aUz_pred,:,col), aUz)
                copyto!(view(aP_pred,:,col), aP)
                pred_params[col] = (λ1_rec, λ2_rec, θ_rec)

                if recover_fields
                    println("🧮 Recovering new flow field data: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                    U_rec, P_rec = reconstruct_fields(mode_cutoff, aUx, aUy, aUz, aP; dir=PODdir)
                    jldsave(recdir*"rec_fields_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2";
                        P_rec = Array(P_rec),
                        U_rec = Array(U_rec)
                    )

                    println("🧮 Plotting new flow fields: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                    D = 72
                    o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
                    o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
                    r  = (D-8)/2  
                    for z in z_vals.*D
                        nx = size(U_rec[:,:,:,1], 1)
                        ny = size(U_rec[:,:,:,1], 2)
                        xtick_vals = 1:D:nx
                        ytick_vals = 1:D:ny
                        xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                        ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                        xticks = (xtick_vals, xtick_labels)
                        yticks = (ytick_vals, ytick_labels)
                        Ux_clims = (-(max(λ1_rec,λ2_rec)-2),max(λ1_rec,λ2_rec)) 
                        plt_Ux_flood = Plots.heatmap(
                            U_rec[:,:,Int(z),1]';
                            xlabel = L"x/D", ylabel = L"y/D",
                            xticks = xticks, yticks = yticks,
                            xlims = (-5,nx+5), ylims = (-5,ny+5),
                            color = :vik50,
                            clims = (Ux_clims),
                            colorbar = true, colorbar_title = L"\overline{u}_{x}/U_{\infty}",
                            levels = 50,                     
                            size = (600, 350),
                            aspect_ratio = :equal,  
                            tickfontsize = 10, labelfontsize = 10,
                            legendfontsize = 10, legend = :topright,
                            left_margin = Plots.Measures.Length(:mm, 5),
                        )
                        Plots.contour!(permutedims(U_rec[:,:,Int(z),1]), levels=range(Ux_clims[1],Ux_clims[2],step=ustep), color=:black, linewidth=1)
                        ϕ = range(0, 2π; length=100)
                        Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        savefig(plt_Ux_flood, string(@__DIR__) * "/tex/Ux_rec/NN/Ux_rec_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                    end
                    
                    for z in z_vals.*D
                        nx = size(P_rec, 1)
                        ny = size(P_rec, 2)
                        xtick_vals = 1:D:nx
                        ytick_vals = 1:D:ny
                        xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                        ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                        xticks = (xtick_vals, xtick_labels)
                        yticks = (ytick_vals, ytick_labels)
                        Ux_clims = (-3,3) #(minimum(Ux_rec[:,:,Int(z)]), maximum(Ux_rec[:,:,Int(z)]))
                        plt_P_flood = Plots.heatmap(
                            P_rec[:,:,Int(z)]';
                            xlabel = L"x/D", ylabel = L"y/D",
                            xticks = xticks, yticks = yticks,
                            xlims = (-5,nx+5), ylims = (-5,ny+5),
                            color = :vik50,
                            clims = (Ux_clims),
                            colorbar = true, colorbar_title = L"\overline{P}",
                            levels = 50,                     
                            size = (600, 350),
                            aspect_ratio = :equal,  
                            tickfontsize = 10, labelfontsize = 10,
                            legendfontsize = 10, legend = :topright,
                            left_margin = Plots.Measures.Length(:mm, 5),
                        )
                        Plots.contour!(permutedims(P_rec[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=pstep), color=:black, linewidth=1)
                        ϕ = range(0, 2π; length=100)
                        Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                        savefig(plt_P_flood, string(@__DIR__) * "/tex/P_rec/NN/P_rec_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                    end
                end

                if (λ1_rec in λ1s) && (λ2_rec in λ2s) && (θ_rec in θs)
                    println("🧮 Calculating coefficient errors: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                    j = snapshot_col_index(λ1s,λ2s,θs,λ1_rec,λ2_rec,θ_rec)
                    append!(all_pred[:aUx], aUx); append!(all_true[:aUx], vec(C_Ux[:, j]))
                    append!(all_pred[:aUy], aUy); append!(all_true[:aUy], vec(C_Uy[:, j]))
                    append!(all_pred[:aUz], aUz); append!(all_true[:aUz], vec(C_Uz[:, j]))
                    append!(all_pred[:aP], aP); append!(all_true[:aP],  vec(C_P[:, j]))

                    if field_err
                        println("🧮 Calculating field errors: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                        P_tru, U_tru, _ = read_meanflow("meanflow_rotor_BiotSimulation_72_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2"; dir=datadir, stats=true, stats_turb=false)
                        Ux_tru, Uy_tru, Uz_tru, = U_tru[:,:,:,1], U_tru[:,:,:,2], U_tru[:,:,:,3]
                        Ux_err = ae(Ux_tru, U_rec[:,:,:,1])
                        Ux_mse = mse(Ux_tru, U_rec[:,:,:,1])
                        P_err = ae(P_tru, P_rec)
                        P_mse = mse(P_tru, P_rec)

                        println("🧮 Plotting field errors: λ₁ = $(λ1_rec), λ₂ = $(λ2_rec), θ = $(θ_rec)")
                        D = 72
                        o₁ = SA[1.5D+2, 2.5D+2, 3.5D]
                        o₂ = SA[4.5D+2, 2.5D+2, 3.5D]
                        r  = (D-8)/2  
                        for z in z_vals.*D
                            nx = size(Ux_err, 1)
                            ny = size(Ux_err, 2)
                            xtick_vals = 1:D:nx
                            ytick_vals = 1:D:ny
                            xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                            ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                            xticks = (xtick_vals, xtick_labels)
                            yticks = (ytick_vals, ytick_labels)
                            Ux_clims = (0,1)
                            plt_Ux_err_flood = Plots.heatmap(
                                Ux_err[:,:,Int(z)]';
                                xlabel = L"x/D", ylabel = L"y/D",
                                xticks = xticks, yticks = yticks,
                                xlims = (-5,nx+5), ylims = (-5,ny+5),
                                color = :vik50,
                                clims = (Ux_clims),
                                title = latexstring("MSE \$\\overline{u}_{x} = $(Ux_mse)\$"),
                                colorbar = true, colorbar_title = latexstring("Absolute Error \$\\overline{u}_{x}\$"),
                                levels = 50,                     
                                size = (600, 350),
                                aspect_ratio = :equal,  
                                titlefontsize = 10, tickfontsize = 10, labelfontsize = 10,
                                legendfontsize = 10, legend = :topright,
                                left_margin = Plots.Measures.Length(:mm, 5),
                            )
                            Plots.contour!(permutedims(Ux_err[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=0.1), color=:black, linewidth=1)
                            ϕ = range(0, 2π; length=100)
                            Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                            Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                            savefig(plt_Ux_err_flood, string(@__DIR__) * "/tex/Ux_err/NN/Ux_err_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                        end

                        for z in z_vals.*D
                            nx = size(P_err, 1)
                            ny = size(P_err, 2)
                            xtick_vals = 1:D:nx
                            ytick_vals = 1:D:ny
                            xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
                            ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
                            xticks = (xtick_vals, xtick_labels)
                            yticks = (ytick_vals, ytick_labels)
                            Ux_clims = (0,1) #(minimum(P_err[:,:,Int(z)]), maximum(P_err[:,:,Int(z)]))
                            plt_P_err_flood = Plots.heatmap(
                                P_err[:,:,Int(z)]';
                                xlabel = L"x/D", ylabel = L"y/D",
                                xticks = xticks, yticks = yticks,
                                xlims = (-5,nx+5), ylims = (-5,ny+5),
                                color = :vik50,
                                clims = (Ux_clims),
                                title = latexstring("MSE \$\\overline{P} = $(P_mse)\$"),
                                colorbar = true, colorbar_title = latexstring("Absolute Error \$\\overline{P}\$"),
                                levels = 50,                     
                                size = (600, 350),
                                aspect_ratio = :equal,  
                                titlefontsize = 10, tickfontsize = 10, labelfontsize = 10,
                                legendfontsize = 10, legend = :topright,
                                left_margin = Plots.Measures.Length(:mm, 5),
                            )
                            Plots.contour!(permutedims(P_err[:,:,Int(z)]), levels=range(Ux_clims[1],Ux_clims[2],step=0.1), color=:black, linewidth=1)
                            ϕ = range(0, 2π; length=100)
                            Plots.plot!(o₁[1] .+ r*cos.(ϕ), o₁[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                            Plots.plot!(o₂[1] .+ r*cos.(ϕ), o₂[2] .+ r*sin.(ϕ), color=:red, lw=2, label="")
                            savefig(plt_P_err_flood, string(@__DIR__) * "/tex/P_err/NN/P_err_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_at_$(z/D).pdf")
                        end
                    end
                end
            end
        end
    end
    paritydir="/ml/tex/errors/"
    r  = mode_cutoff                  
    x  = all_pred[:aUx]               
    y  = all_true[:aUx]               
    xf = 1e4; yf = 1e4
    nsnaps = length(x) ÷ r
    param_list = [(λ1, λ2, θ) for θ in θ_recs, λ2 in λ2_recs, λ1 in λ1_recs] |> vec
    cols = [
        :red, :blue, :green, :magenta, :orange, :cyan,
        :black, :purple, :brown, :olive, :navy, :teal,
        :pink, :gray, :gold, :indigo, :seagreen, :tomato,
        :coral, :darkorange, :darkgreen, :darkred, :dodgerblue,
        :chocolate, :slategray, :darkviolet, :turquoise, :darkkhaki,
        :deeppink
    ]
    @assert length(cols) >= nsnaps "Provide at least nsnaps colors in `cols`"
    snap_labels = [latexstring("\\lambda_{1}=$(λ1), \\lambda_{2}=$(λ2), \\theta=$(θ)") for (λ1,λ2,θ) in param_list][1:nsnaps]
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    Parity_Ux = Plots.plot(
        xlabel = L"a_{\star,i}^{(\overline{u}_{x})}\ \times 10^{4}",
        ylabel = L"a_{i}^{(\overline{u}_{x})}\ \times 10^{4}",
        size = (600, 600),
        framestyle = :box, grid = true, minorgrid = true,
        legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
        xformatter = v -> @sprintf("%.3f", v / xf),
        yformatter = v -> @sprintf("%.3f", v / yf),
    )
    for s in 1:nsnaps
        idx = (s-1)*r + 1 : s*r
        Plots.scatter!(
            Parity_Ux, x[idx], y[idx];
            color = cols[s], ms = 3, markerstrokewidth = 0.5,
            label = snap_labels[s]
        )
    end
    Plots.plot!(Parity_Ux, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
    savefig(Parity_Ux, joinpath(paritydir, "Parity_Ux.pdf"))

    x  = all_pred[:aUy]              
    y  = all_true[:aUy]              
    xf = 1e4; yf = 1e4
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    Parity_Uy = Plots.plot(
        xlabel = L"a_{\star,i}^{(\overline{u}_{y})}\ \times 10^{4}",
        ylabel = L"a_{i}^{(\overline{u}_{y})}\ \times 10^{4}",
        size = (600, 600),
        framestyle = :box, grid = true, minorgrid = true,
        legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
        xformatter = v -> @sprintf("%.3f", v / xf),
        yformatter = v -> @sprintf("%.3f", v / yf),
    )
    for s in 1:nsnaps
        idx = (s-1)*r + 1 : s*r
        Plots.scatter!(
            Parity_Uy, x[idx], y[idx];
            color = cols[s], ms = 3, markerstrokewidth = 0.5,
            label = snap_labels[s]
        )
    end
    Plots.plot!(Parity_Uy, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
    savefig(Parity_Uy, joinpath(paritydir, "Parity_Uy.pdf"))

    x  = all_pred[:aUz]              
    y  = all_true[:aUz]              
    xf = 1e3; yf = 1e3
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    Parity_Uz = Plots.plot(
        xlabel = L"a_{\star,i}^{(\overline{u}_{z})}\ \times 10^{3}",
        ylabel = L"a_{i}^{(\overline{u}_{z})}\ \times 10^{3}",
        size = (600, 600),
        framestyle = :box, grid = true, minorgrid = true,
        legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
        xformatter = v -> @sprintf("%.3f", v / xf),
        yformatter = v -> @sprintf("%.3f", v / yf),
    )
    for s in 1:nsnaps
        idx = (s-1)*r + 1 : s*r
        Plots.scatter!(
            Parity_Uz, x[idx], y[idx];
            color = cols[s], ms = 3, markerstrokewidth = 0.5,
            label = snap_labels[s]
        )
    end
    Plots.plot!(Parity_Uz, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
    savefig(Parity_Uz, joinpath(paritydir, "Parity_Uz.pdf"))

    x  = all_pred[:aP]              
    y  = all_true[:aP]              
    xf = 1e4; yf = 1e4
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    Parity_P = Plots.plot(
        xlabel = L"a_{\star,i}^{(\overline{P})}\ \times 10^{4}",
        ylabel = L"a_{i}^{(\overline{P})}\ \times 10^{4}",
        size = (600, 600),
        framestyle = :box, grid = true, minorgrid = true,
        legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
        xformatter = v -> @sprintf("%.3f", v / xf),
        yformatter = v -> @sprintf("%.3f", v / yf),
    )
    for s in 1:nsnaps
        idx = (s-1)*r + 1 : s*r
        Plots.scatter!(
            Parity_P, x[idx], y[idx];
            color = cols[s], ms = 3, markerstrokewidth = 0.5,
            label = snap_labels[s]
        )
    end
    Plots.plot!(Parity_P, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
    savefig(Parity_P, joinpath(paritydir, "Parity_P.pdf"))   

    jldsave(recdir*"coefficients.jld2";
        aUx_pred = Array(aUx_pred),
        aUy_pred = Array(aUy_pred),
        aUz_pred = Array(aUz_pred),
        aP_pred  = Array(aP_pred),
        pred_params = collect(pred_params)
    )
    
    mseUx = mse(all_true[:aUx], all_pred[:aUx]); @out mseUx
    mseUy = mse(all_true[:aUy], all_pred[:aUy]); @out mseUy
    mseUz = mse(all_true[:aUz], all_pred[:aUz]); @out mseUz
    mseP = mse(all_true[:aP], all_pred[:aP]); @out mseP 

    outfile = joinpath(paritydir, "mse_summary.txt")

    open(outfile, "w") do io
        println(io, "# Mean-Squared Errors for POD-ML Prediction")
        @printf(io, "MSE_Ux = %.6e\n", mseUx)
        @printf(io, "MSE_Uy = %.6e\n", mseUy)
        @printf(io, "MSE_Uz = %.6e\n", mseUz)
        @printf(io, "MSE_P  = %.6e\n", mseP)
    end

    pod_surface_double(mode_cutoff, Nmax, aUx_pred, C_Ux, 1, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Ux_double")
    pod_surface_double(mode_cutoff, Nmax, aUy_pred, C_Uy, 2, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Uy_double")
    pod_surface_double(mode_cutoff, Nmax, aUz_pred, C_Uz, 3, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Uz_double")
    pod_surface_double(mode_cutoff, Nmax, aP_pred, C_P, 4, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_P_double")

    pod_surface_single(mode_cutoff, Nmax, C_Ux, 1, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Ux_single")
    pod_surface_single(mode_cutoff, Nmax, C_Uy, 2, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Uy_single")
    pod_surface_single(mode_cutoff, Nmax, C_Uz, 3, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_Uz_single")
    pod_surface_single(mode_cutoff, Nmax, C_P, 4, λ1s, λ2s, θs; outdir=outdir, fname="PODSurf_P_single")
    println("✅ Done!")
end
main()