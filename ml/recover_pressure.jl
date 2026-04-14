    using WaterLily, BiotSavartBCs, StaticArrays
    include("ML.jl")

    # read helper
    function read_recovered_fields(fname::AbstractString; dir::AbstractString="")
        jldopen(joinpath(dir, fname)) do f
            return f["P_rec"], f["U_rec"]
        end
    end

    backend = Array
    recdir  = "/ml/data/rotor_BiotSimulation/recovered/"
    datadir = "/sims/data/rotor_BiotSimulation/"

    z_vals  = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5]

    function main()
        λ1_rec = 3; λ2_rec = 3; θ_rec = 0

        # Load recovered mean fields (we only use U_rec here)
        # _, U_rec = read_recovered_fields("rec_fields_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2"; dir=recdir)
        _, U_rec, UU, τ, t_meanflow = read_meanflow("meanflow_rotor_BiotSimulation_72_$(λ1_rec)_$(λ2_rec)_$(θ_rec).jld2"; dir=datadir, stats=true, stats_turb=true)
        @out size(t_meanflow)
        Δt = t_meanflow[end]/72 - t_meanflow[end-1]/72

        T   = eltype(U_rec)
        ND  = ndims(U_rec) - 3             # 2 or 3
        Ng  = size(U_rec)[1:ndims(U_rec)-1]
        N   = ntuple(i -> Ng[i]-2, ND)     # interior

        # Problem/geometry params (keep 'D' as your diameter resolution)
        Re = 1000
        D  = 72
        U∞ = one(T)
        ν  = U∞*D/Re
        Lchar = D

        # Inflow direction from θ_rec
        uBC = (T(cosd(θ_rec)), T(-sind(θ_rec)), zero(T))  # tuple is fine for BiotSimulation

        # Rotor geometry (same as your sim)
        r  = D/2;  rₑ = D;  h = 7D;  ϵ = one(T);  thk = 2ϵ + sqrt(T(2))
        o₁ = SA[T(1.5D), T(2.5D), T(3.5D)]
        o₂ = SA[T(4.5D), T(2.5D), T(3.5D)]
        θ₁ = (λ1_rec)*(U∞/r)
        θ₂ = (λ2_rec)*(U∞/r)
        R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]

        function sdf(xyz, t)
            x,y,z = xyz
            d_radial = √sum(abs2, SA[x,y,zero(T)]) - r
            d_top    =  z - h/2
            d_bottom = -z - h/2
            sdf_rotor = max(d_radial, max(d_top, d_bottom))
            sdf_endplate_top    = max(√sum(abs2, SA[x,y,zero(T)]) - rₑ, abs(z - h/2) - thk/2)
            sdf_endplate_bottom = max(√sum(abs2, SA[x,y,zero(T)]) - r,  abs(z + h/2) - thk/2)
            return min(sdf_rotor, sdf_endplate_top, sdf_endplate_bottom)
        end

        Rotor₁ = AutoBody(sdf, (xyz,t)->R(θ₁*t)*(xyz - o₁))
        Rotor₂ = AutoBody(sdf, (xyz,t)->R(θ₂*t)*(xyz - o₂))
        body   = Rotor₁ + Rotor₂

        # Build BiotSimulation (keep your non-Biot face)
        bsim = BiotSimulation((8D,5D,8D), uBC, Lchar; ν=ν, body=body, mem=backend, T=T, nonbiotfaces=(-3,))

        # Inject your recovered mean velocity
        bsim.flow.u .= U_rec

        # Geometry & Poisson operator refresh
        measure!(bsim)
        WaterLily.update!(bsim.pois)

        # 2) Build RHS = -∂i∂j(UU_ij)
        b = bsim.pois
        fill!(b.z, 0)
        # ND = length(bsim.flow.N)
        # accumulate: use @loop so you can do multiple lines or loops
        WaterLily.@loop b.z[I] = 0 over I ∈ inside(bsim.flow.p)
        for i in 1:ND, j in 1:ND
            UUij = view(UU, :, :, :, i, j) # 3D slice on pressure grid
            WaterLily.@loop b.z[I] += -WaterLily.∂(i, I, WaterLily.∂(j, I, UUij)) over I ∈ inside(bsim.flow.p)
        end

        # 3) Solve A p̄ = b (no Δt scaling)
        fill!(b.x, 0)
        WaterLily.solver!(b)
        p = Array(b.x)
        # p .-= mean(p[WaterLily.inside(p)])  # remove constant before plotting

        # --- Plotting slices ---
        # rotor outlines (slightly inset from your sim’s plotting choices)
        o₁p = SA[T(1.5D+2), T(2.5D+2), T(3.5D)]
        o₂p = SA[T(4.5D+2), T(2.5D+2), T(3.5D)]
        rplt = (D-8)/2
        clims_p = (-0.5, 0.5)

        nx, ny, nz = size(p)
        for zD in z_vals
            kz = clamp(Int(round(zD*D)), 2, nz-1)  # guard ghosts

            xtick_vals   = 1:D:nx
            ytick_vals   = 1:D:ny
            xtick_labels = [@sprintf("%.1f", x / D) for x in xtick_vals]
            ytick_labels = [@sprintf("%.1f", y / D) for y in ytick_vals]
            xticks = (xtick_vals, xtick_labels)
            yticks = (ytick_vals, ytick_labels)

            plt = Plots.heatmap(
                p[:,:,kz]';
                xlabel = L"x/D", ylabel = L"y/D",
                xticks = xticks, yticks = yticks,
                xlims = (-5, nx+5), ylims = (-5, ny+5),
                color = :vik25, clims = clims_p,
                colorbar = true, colorbar_title = L"\overline{P}",
                levels = 25, size = (600, 350),
                aspect_ratio = :equal,
                tickfontsize = 10, labelfontsize = 10,
                legendfontsize = 10, legend = :topright,
                left_margin = Plots.Measures.Length(:mm, 5),
            )

            Plots.contour!(permutedims(p[:,:,kz]),
                    levels=range(clims_p[1], clims_p[2], step=0.25),
                    color=:black, linewidth=1, label="")

            ϕ = range(0, 2π; length=200)
            Plots.plot!(o₁p[1] .+ rplt*cos.(ϕ), o₁p[2] .+ rplt*sin.(ϕ), color=:red, lw=2, label="")
            Plots.plot!(o₂p[1] .+ rplt*cos.(ϕ), o₂p[2] .+ rplt*sin.(ϕ), color=:red, lw=2, label="")
            savefig(plt, joinpath(@__DIR__, "tex", "rotor_BiotSimulation_$(D)_$(λ1_rec)_$(λ2_rec)_$(θ_rec)_P_flood_at_$(zD).pdf"))
        end
    end

    main()