using WaterLily,BiotSavartBCs
using StaticArrays
using LinearAlgebra
using OutMacro

function circle(D)
    r=D/2
    o₁ = SA[1.5D,2.5D]
    o₂ = SA[4.5D,2.5D] 
    function sdf₁(xy,t)
        x,y = xy
        √sum(abs2, SA[x,y] .- o₁) - r
    end
    function sdf₂(xy,t)
        x,y = xy
        √sum(abs2, SA[x,y] .- o₂) - r
    end
    body₁ = WaterLily.AutoBody(sdf₁)
    body₂ = WaterLily.AutoBody(sdf₂)
    body₊ = body₁ + body₂
    return body₁, body₂, body₊
end

function Rotor_Validation_2D(D, λ₁, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2         
    o₁ = SA[1.5D,2.5D] 
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)        
    R(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]    
    function sdf(xy,t)
        x,y = xy
        √sum(abs2, SA[x,y]) - r
    end
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    body = Rotor₁
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T), Rotor₁
end

function Rotor_2D(D, λ₁, λ₂, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2         
    o₁ = SA[1.5D,2.5D]  
    o₂ = SA[4.5D,2.5D]  
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)        
    θ₂ = (λ₂)*(U/r)  
    R(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]    
    function sdf(xy,t)
        x,y = xy
        √sum(abs2, SA[x,y]) - r
    end
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    Rotor₂ = AutoBody(sdf,(xyz,t)->R(θ₂*t)*(xyz-o₂))
    body = Rotor₁ + Rotor₂
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T), Rotor₁, Rotor₂
end




