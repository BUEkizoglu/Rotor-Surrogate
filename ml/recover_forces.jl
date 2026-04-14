include("../setup/3D.jl")
include("../help/MeanFlow_PostProc.jl")
include("../help/Forces_PostProc.jl")
include("ML.jl")
using JLD2
using Printf

# read helper
function read_recovered_fields(fname::AbstractString; dir::AbstractString="")
    jldopen(joinpath(dir, fname)) do f
        return f["P_rec"], f["U_rec"]
    end
end

D = 72                 # diameter resolutions
λ1s, λ2s = [3,4,5], [3,4,5]
backend = Array
L = (8,5,8)
Re = 1000
T = Float32
θs = [0,15,30]
Bayesian = false
calc_forces = true
postprocess = true
u_probe_loc = (3,2.5,3.5) # in D
u_probe_component = 1
z_vals = T.([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5])

datadir = "/sims/data/rotor_BiotSimulation/"
recdir  = "/ml/data/rotor_BiotSimulation/recovered/"
fdir    = "/ml/data/rotor_BiotSimulation/recovered/"

function decompose_vec(v)
    v1, v2, v3 = -v[1], -v[2], -v[3]
    return v1, v2, v3
end

function calculate_coeffs(f1, f2, D)
    ft = f1 .+ f2
    ftx, fty, ftz = decompose_vec(ft)
    Fx = ftx
    Fy = fty
    Fz = ftz
    A = ((35*π*D^2)/4) + (2*π*(2+√2))
    Cx = Fx./(0.5*A)
    Cy = Fy./(0.5*A)
    Cz = Fz./(0.5*A)
    return (; Fx, Fy, Fz, Cx, Cy, Cz)
end

function main()
    # Store forces for *all* (D, λ₁, λ₂, θ)
    # key = (D, λ₁, λ₂, θ)
    rforces = Dict{NTuple{4,Int}, NamedTuple}()
    tforces = Dict{NTuple{4,Int}, NamedTuple}()
    if calc_forces
        for λ₁ in λ1s
            for λ₂ in λ2s
                for θ in θs
                    key = (D, λ₁, λ₂, θ)
                    @printf("Processing D=%d, λ₁=%d, λ₂=%d, θ=%d\n", D, λ₁, λ₂, θ)

                    fname = Bayesian ?
                        "rec_fields_B_$(λ₁)_$(λ₂)_$(θ).jld2" :
                        "rec_fields_$(λ₁)_$(λ₂)_$(θ).jld2"

                    # Recovered fields
                    P_rec, U_rec = read_recovered_fields(fname; dir=recdir)

                    # True meanflow fields
                    P, U, t = read_meanflow("meanflow_rotor_BiotSimulation_$(D)_$(λ₁)_$(λ₂)_$(θ).jld2";dir=datadir, stats = true, stats_turb = false)

                    # Build a simulation with the correct geometry
                    U₊ = (T(cosd(θ)), T(-sind(θ)), zero(T))
                    sim, Rotor₁, Rotor₂ = ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, backend; L, Re, T)

                    @assert size(sim.flow.p) == size(P)
                    sim.flow.u .= U
                    sim.flow.p .= P

                    println("True case: U = $(sim.U); L = $(sim.L)")
                    p_tforce₁ = WaterLily.pressure_force(sim.flow, Rotor₁)
                    p_tforce₂ = WaterLily.pressure_force(sim.flow, Rotor₂)
                    v_tforce₁ = WaterLily.viscous_force(sim.flow, Rotor₁) 
                    v_tforce₂ = WaterLily.viscous_force(sim.flow, Rotor₂)

                    # Reconstructed case
                    @assert size(sim.flow.p) == size(P_rec)
                    sim.flow.u .= U_rec
                    sim.flow.p .= P_rec

                    println("Recovered case: U = $(sim.U); L = $(sim.L)")
                    p_rforce₁ = WaterLily.pressure_force(sim.flow, Rotor₁)
                    p_rforce₂ = WaterLily.pressure_force(sim.flow, Rotor₂) 
                    v_rforce₁ = WaterLily.viscous_force(sim.flow, Rotor₁)
                    v_rforce₂ = WaterLily.viscous_force(sim.flow, Rotor₂)

                    # Convert to Fx,Fy,Fz,Cx,Cy,Cz
                    rforce₁ = calculate_coeffs(p_rforce₁, v_rforce₁, D)
                    rforce₂ = calculate_coeffs(p_rforce₂, v_rforce₂, D)
                    tforce₁ = calculate_coeffs(p_tforce₁, v_tforce₁, D)
                    tforce₂ = calculate_coeffs(p_tforce₂, v_tforce₂, D)

                    # Store in dictionaries
                    rforces[key] = (rotor1 = rforce₁, rotor2 = rforce₂)
                    tforces[key] = (rotor1 = tforce₁, rotor2 = tforce₂)

                    # Simple error diagnostics
                    errCx₁ = abs(rforce₁.Cx - tforce₁.Cx)/abs(tforce₁.Cx) * 100
                    errCx₂ = abs(rforce₂.Cx - tforce₂.Cx)/abs(tforce₂.Cx) * 100
                    errCxt = abs((rforce₁.Cx + rforce₂.Cx) - (tforce₁.Cx + tforce₂.Cx))/abs(tforce₁.Cx + tforce₂.Cx) * 100
                    println("Error Cx₁ = $(errCx₁)%"); println("Error Cx₂ = $(errCx₂)%"); println("Error Cxt = $(errCxt)%")

                    errCy₁ = abs(rforce₁.Cy - tforce₁.Cy)/abs(tforce₁.Cy) * 100
                    errCy₂ = abs(rforce₂.Cy - tforce₂.Cy)/abs(tforce₂.Cy) * 100
                    errCyt = abs((rforce₁.Cy + rforce₂.Cy) - (tforce₁.Cy + tforce₂.Cy))/abs(tforce₁.Cy + tforce₂.Cy) * 100
                    println("Error Cy₁ = $(errCy₁)%"); println("Error Cy₂ = $(errCy₂)%"); println("Error Cyt = $(errCyt)%")
                end
            end
        end


        outpath = Bayesian ?
            joinpath(fdir, "forces_B.jld2") :
            joinpath(fdir, "forces_NN.jld2")
        @info "Saving forces to $outpath"
        jldsave(outpath; rforces = rforces, tforces = tforces)
        println("✅ Saved forces for $(length(rforces)) cases to $outpath")
    end

    if postprocess 
        fname = Bayesian ?
            "forces_B.jld2" :
            "forces_NN.jld2"
        rforces, tforces = jldopen(joinpath(fdir, fname)) do f
            f["rforces"], f["tforces"]
        end

        # rforcesB2, tforcesB2 = jldopen(joinpath(fdir, "forces_B_2.jld2")) do f
        #     f["rforces"], f["tforces"]
        # end

        # Collect and sort keys so snapshot index is consistent
        keyB = sort(collect(keys(rforces)))  # sorted by (D, λ₁, λ₂, θ)
        nsnap = length(keyB)
        @out keyB

        Cx_true = zeros(T, nsnap)
        Cx1_true = zeros(T, nsnap)
        Cx2_true = zeros(T, nsnap)
        Cx_rec  = zeros(T, nsnap)
        Cx1_rec  = zeros(T, nsnap)
        Cx2_rec  = zeros(T, nsnap)
        Cy_true = zeros(T, nsnap)
        Cy1_true = zeros(T, nsnap)
        Cy2_true = zeros(T, nsnap)
        Cy_rec  = zeros(T, nsnap)
        Cy1_rec  = zeros(T, nsnap)
        Cy2_rec  = zeros(T, nsnap)

        for (i, key) in enumerate(keyB)
            rf = rforces[key]
            tf = tforces[key]

            # total = rotor1 + rotor2
            Cx_true[i] = tf.rotor1.Cx + tf.rotor2.Cx
            Cx1_true[i] = tf.rotor1.Cx
            Cx2_true[i] = tf.rotor2.Cx
            Cx_rec[i]  = rf.rotor1.Cx + rf.rotor2.Cx
            Cx1_rec[i] = rf.rotor1.Cx
            Cx2_rec[i] = rf.rotor2.Cx
            Cy_true[i] = tf.rotor1.Cy + tf.rotor2.Cy
            Cy1_true[i] = tf.rotor1.Cy
            Cy2_true[i] = tf.rotor2.Cy
            Cy_rec[i]  = rf.rotor1.Cy + rf.rotor2.Cy
            Cy1_rec[i] = rf.rotor1.Cy
            Cy2_rec[i] = rf.rotor2.Cy

        end
        
        paritydir = "/ml/tex/errors/"
        xf = 1.0            # scaling if you want (leave 1.0 for raw values)
        yf = 1.0

        cols = [
            :red, :blue, :green, :magenta, :orange, :cyan,
            :black, :purple, :brown, :olive, :navy, :teal,
            :pink, :gray, :gold, :indigo, :seagreen, :tomato,
            :coral, :darkorange, :darkgreen, :darkred, :dodgerblue,
            :chocolate, :slategray, :darkviolet, :turquoise, :darkkhaki,
            :deeppink
        ]

        snap_labels = [latexstring("\\lambda_{1}=$(λ1), \\lambda_{2}=$(λ2), \\theta=$(θ)") for (D, λ1, λ2, θ) in keyB][1:nsnap]
        lo = min(minimum(Cx_rec), minimum(Cx_true))
        hi = max(maximum(Cx_rec), maximum(Cx_true))
        Parity_Cx = Plots.plot(
            xlabel = L"C^{\star}_{x,\mathrm{total}}",
            ylabel = L"C_{x,\mathrm{total}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cx, [Cx_rec[s]], [Cx_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cx, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cx_total_Bayesian_jl.pdf" :
            "Parity_Cx_total_NN_jl.pdf"
        savefig(Parity_Cx, joinpath(paritydir, sname))

        lo = min(minimum(Cy_rec), minimum(Cy_true))
        hi = max(maximum(Cy_rec), maximum(Cy_true))
        Parity_Cy = Plots.plot(
            xlabel = L"C^{\star}_{y,\mathrm{total}}",
            ylabel = L"C_{y,\mathrm{total}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cy, [Cy_rec[s]], [Cy_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cy, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cy_total_Bayesian_jl.pdf" :
            "Parity_Cy_total_NN_jl.pdf"
        savefig(Parity_Cy, joinpath(paritydir, sname))

        lo = min(minimum(Cx1_rec), minimum(Cx1_true))
        hi = max(maximum(Cx1_rec), maximum(Cx1_true))
        Parity_Cx1 = Plots.plot(
            xlabel = L"C^{\star}_{x_{1}}",
            ylabel = L"C_{x_{1}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cx1, [Cx1_rec[s]], [Cx1_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cx1, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cx1_Bayesian_jl.pdf" :
            "Parity_Cx1_NN_jl.pdf"
        savefig(Parity_Cx1, joinpath(paritydir, sname))

        lo = min(minimum(Cx2_rec), minimum(Cx2_true))
        hi = max(maximum(Cx2_rec), maximum(Cx2_true))
        Parity_Cx2 = Plots.plot(
            xlabel = L"C^{\star}_{x_{2}}",
            ylabel = L"C_{x_{2}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cx2, [Cx2_rec[s]], [Cx2_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cx2, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cx2_Bayesian_jl.pdf" :
            "Parity_Cx2_NN_jl.pdf"
        savefig(Parity_Cx2, joinpath(paritydir, sname))

        lo = min(minimum(Cy1_rec), minimum(Cy1_true))
        hi = max(maximum(Cy1_rec), maximum(Cy1_true))
        Parity_Cy1 = Plots.plot(
            xlabel = L"C^{\star}_{y_{1}}",
            ylabel = L"C_{y_{1}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cy1, [Cy1_rec[s]], [Cy1_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cy1, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cy1_Bayesian_jl.pdf" :
            "Parity_Cy1_NN_jl.pdf"
        savefig(Parity_Cy1, joinpath(paritydir, sname))

        lo = min(minimum(Cy2_rec), minimum(Cy2_true))
        hi = max(maximum(Cy2_rec), maximum(Cy2_true))
        Parity_Cy2 = Plots.plot(
            xlabel = L"C^{\star}_{y_{2}}",
            ylabel = L"C_{y_{2}}",
            size = (600, 600),
            framestyle = :box, grid = true, minorgrid = true,
            legend = :outerright, legendfontsize = 8, tickfontsize = 8, labelfontsize = 10,
            xformatter = v -> @sprintf("%.3f", v / xf),
            yformatter = v -> @sprintf("%.3f", v / yf),
        )
        for s in 1:nsnap
            Plots.scatter!(
                Parity_Cy2, [Cy2_rec[s]], [Cy2_true[s]];
                color = cols[s], ms = 3, markerstrokewidth = 0.5,
                label = snap_labels[s],
            )
        end
        Plots.plot!(Parity_Cy2, [lo, hi], [lo, hi]; color = :black, lw = 2, label = "y = x")
        sname = Bayesian ?
            "Parity_Cy2_Bayesian_jl.pdf" :
            "Parity_Cy2_NN_jl.pdf"
        savefig(Parity_Cy2, joinpath(paritydir, sname))
    end

    mse_Cx_total = mse(Cx_true,  Cx_rec)
    mse_Cy_total = mse(Cy_true,  Cy_rec)

    mse_Cx1 = mse(Cx1_true, Cx1_rec)
    mse_Cx2 = mse(Cx2_true, Cx2_rec)
    mse_Cy1 = mse(Cy1_true, Cy1_rec)
    mse_Cy2 = mse(Cy2_true, Cy2_rec)

    @info "MSE Cx_total = $(mse_Cx_total)"
    @info "MSE Cy_total = $(mse_Cy_total)"
    @info "MSE Cx₁      = $(mse_Cx1)"
    @info "MSE Cx₂      = $(mse_Cx2)"
    @info "MSE Cy₁      = $(mse_Cy1)"
    @info "MSE Cy₂      = $(mse_Cy2)"

    outfile = Bayesian ?
        joinpath(paritydir,"mse_summary_force_B_jl.txt") :
        joinpath(paritydir,"mse_summary_force_NN_jl.txt")
    open(outfile, "w") do io
        println(io, "# Mean-Squared Errors for POD-ML Prediction")
        @printf(io, "MSE Cx total = %.6e\n", mse_Cx_total)
        @printf(io, "MSE Cy total = %.6e\n", mse_Cy_total)
        @printf(io, "MSE Cx1 = %.6e\n", mse_Cx1)
        @printf(io, "MSE Cx2  = %.6e\n", mse_Cx2)
        @printf(io, "MSE Cy1  = %.6e\n", mse_Cy1)
        @printf(io, "MSE Cy2  = %.6e\n", mse_Cy2)
    end
end
main()