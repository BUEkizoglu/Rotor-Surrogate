
function save_simulation_vtk(sim; basename="output", duration, tstep, udf=nothing)
    # Define output fields
    velocity(a::Simulation) = a.flow.u |> Array
    pressure(a::Simulation) = a.flow.p |> Array
    _body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a)); a.flow.σ |> Array)
    lambda(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.λ₂(I, a.flow.u); a.flow.σ |> Array)
    vorticity_mag(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.ω_mag(I, a.flow.u); a.flow.σ |> Array)
    curlx(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(1,I,a.flow.u)*a.L/a.U; a.flow.σ |> Array)
    curly(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(2,I,a.flow.u)*a.L/a.U; a.flow.σ |> Array)
    curlz(a::Simulation) = (@inside a.flow.σ[I] = WaterLily.curl(3,I,a.flow.u)*a.L/a.U; a.flow.σ |> Array)

    custom_attrib = Dict(
        "Velocity" => velocity,
        "Pressure" => pressure,
        "Body" => _body,
        "Lambda" => lambda,
        "Vorticity mag." => vorticity_mag,
        "Curl X" => curlx,
        "Curl Y" => curly,
        "Curl Z" => curlz,
    )

    # Setup VTK writer
    writer = vtkWriter(basename; attrib=custom_attrib)

    # Time-stepping parameters
    t₀ = sim_time(sim)
    t_final = t₀ + duration

    # Simulation loop
    @time for tᵢ in range(t₀, t_final; step=tstep)
        sim_step!(sim, tᵢ, remeasure=true; udf=udf)
        save!(writer, sim)
        println("tU/L = ", round(tᵢ, digits=4), ", Δt = ", round(sim.flow.Δt[end], digits=3))
    end
    close(writer)
end