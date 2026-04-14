
using WaterLily
using Test
using GPUArrays
using CUDA
using OutMacro
using Plots
using WriteVTK
using ReadVTK
include("../setup/3D.jl")
include("../setup/2D.jl")

D = 32
backend = CUDA.CuArray
L = (10,10,3)
Re = 1000
T = Float32
U = 1
ν = U*D/Re
Nx, Ny, Nz = L[1]*D+2, L[2]*D+2, L[3]*D+2
x_scale = 10D 

# aim = Plate_2D_approx(D,backend;Re,T)
sim =  ones(D,backend;L,Re,T)

#---------------------------------------------------------------------------------------------------#
# VTK writer
#---------------------------------------------------------------------------------------------------#
# Define output fields, convert to CPU array for writing
velocity(a::Simulation) = a.flow.u |> Array
pressure(a::Simulation) = a.flow.p |> Array
_body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array)
# lambda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u); a.flow.σ |> Array)
vorticity_mag(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.ω_mag(I, a.flow.u); a.flow.σ |> Array)
# curlx(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(1,I,sim.flow.u)*sim.L/sim.U; a.flow.σ |> Array)
# curly(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(2,I,sim.flow.u)*sim.L/sim.U; a.flow.σ |> Array)
# curlz(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U; a.flow.σ |> Array)
# Custom VTK writer attributes
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => _body,
    # "Lambda" => lambda,
    "Vorticity mag." => vorticity_mag,
    # "Curl X" => curlx,
    # "Curl Y" => curly,
    # "Curl Z" => curlz,
)
# Setup VTK writer
writer = vtkWriter("Plate_D_$(D)_no_z"; attrib=custom_attrib)

# Time-stepping parameters
duration = 20
tstep = 0.1
t₀ = sim_time(sim)

# Run simulation loop
@time for tᵢ in range(t₀, t₀ + duration; step=tstep)
    sim_step!(sim, tᵢ, remeasure=true)
    WaterLily.save!(writer, sim)
    println("tU/L = ", round(tᵢ, digits=4), ", Δt = ", round(sim.flow.Δt[end], digits=3))
end
close(writer)