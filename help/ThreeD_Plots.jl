using WaterLily
using GLMakie
using StaticArrays
using Meshing, GeometryBasics
# GLMakie.activate!()
function makie_video!(makie_plot,sim,dat,obs_update!;remeasure=false,name="file.mp4",duration=1,step=0.1,framerate=30,compression=20)
    # Set up viz data and figure
    obs = obs_update!(dat,sim) |> Observable;
    f = makie_plot(obs)
    
    # Run simulation and update figure data
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    GLMakie.record(f, name, t; framerate, compression) do tᵢ
        sim_step!(sim,tᵢ;remeasure)
        obs[] = obs_update!(dat,sim)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return 
end


function body_mesh(sim,t=0)
    a = sim.flow.σ; R = inside(a)
    WaterLily.measure_sdf!(a,sim.body,t)
    normal_mesh(GeometryBasics.Mesh(a[R]|>Array,MarchingCubes(),origin=Vec(0,0,0),widths=size(R)))
end
function flow_λ₂!(dat,sim)
    a = sim.flow.σ
    @inside a[I] = max(0,log10(-min(-1e-6,WaterLily.λ₂(I,sim.flow.u)*(sim.L/sim.U)^2))+.25)
    copyto!(dat,a[inside(a)])                  # copy to CPU
end
function flow_λ₂(sim)
    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array
    flow_λ₂!(dat,sim)
    dat
end

function Q(I::CartesianIndex{3},u)
    J = @SMatrix [WaterLily.∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    0.5*( sum(abs2, Ω) - sum(abs2, S))
end

function flow_Q!(dat,sim)
    a = sim.flow.σ
    @inside a[I] = Q(I,sim.flow.u)*((sim.L/sim.U)^2)
    copyto!(dat,a[inside(a)])      
end

function flow_Q(sim)
    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array
    flow_Q!(dat,sim)
    dat
end

function Q_iso_vertices_ω₃(sim; Qiso = 1f-5)
    a = sim.flow.σ; R = inside(a); D = sim.L
    Q = flow_Q(sim)                  
    nx, ny, nz = size(Q)
    ωtmp = similar(a)
    @inside ωtmp[I] = WaterLily.curl(3, I, sim.flow.u) * (sim.L / sim.U)
    ω = ωtmp[R] |> Array
    
    isosurf = Meshing.MarchingCubes(;iso=Qiso)
    x = 1:nx; y = 1:ny; z = 1:nz
    vts, fcs = Meshing.isosurface(Q,isosurf,x,y,z) 

    colors = Vector{Float32}(undef, length(vts))
    for (n, v) in enumerate(vts)
        x, y, z = v
        i = x |> ceil |> Int
        j = y |> ceil |> Int
        k = z |> ceil |> Int
        colors[n] = ω[i,j,k]
    end
    vertsM, facesM = to_mesh_matrices(vts, fcs)
    return vertsM, facesM, colors
end
           
function to_mesh_matrices(vts::Vector{<:NTuple{3,<:Real}}, fcs::Vector{<:NTuple{3,<:Integer}})
    nv = length(vts)
    nf = length(fcs)
    V = Array{Float32}(undef, nv, 3)
    for (i, (x, y, z)) in enumerate(vts)
        V[i, 1] = x
        V[i, 2] = y
        V[i, 3] = z
    end
    F = Array{Int}(undef, nf, 3)
    for (i, (a, b, c)) in enumerate(fcs)
        F[i, 1] = a
        F[i, 2] = b
        F[i, 3] = c
    end
    return V, F
end