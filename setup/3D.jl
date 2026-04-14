using WaterLily,BiotSavartBCs
using StaticArrays
using LinearAlgebra
using OutMacro

function ThreeD_Plate_wBL(D,backend;L,Re,T)
    U = 1  
    ν = U*D/Re
    n = SA[0,0,1]
    function UBC(i, xyz, t)
        x, y, z = xyz
        x_scale = max(Float32(10D + (x)), Float32(10D))
        η = max(Float32((z) * √(1 / (ν * x_scale))), 0.f0)
        if i == 1
            return max(Float32(1f0 - 0.1608f0 * exp(-η)
                           - 1.2956f0 * η * exp(-η)
                           - 0.8392f0 * exp(-2f0 * η)), 0f0)
        else
            return 0f0
        end
    end
    body = AutoBody((xyz,t)->dot(xyz,n))
    Simulation(L.*D, UBC, D; U=U, ν=ν, body=body, mem=backend, perdir=(2,), exitBC=true, T=T, uλ=(i,x) -> UBC(i, x, 0f0))
end

function ThreeD_Cylinder_wPlate_wBL(D,backend;L,Re,T)
    U = 1
    ν = U*D/Re
    r = D/2    
    h = D      
    o = SA[2.5D, 5D, 0.5D]  
    n = SA[0,0,1]
    function UBC(i, xyz, t)
        x, y, z = xyz
        x_scale = max(Float32(10D + (x)), Float32(10D))
        η = max(Float32((z) * √(1 / (ν * x_scale))), 0.f0)
        if i == 1
            return max(Float32(1f0 - 0.1608f0 * exp(-η)
                           - 1.2956f0 * η * exp(-η)
                           - 0.8392f0 * exp(-2f0 * η)), 0f0)
        else
            return 0f0
        end
    end
    function cylinder_sdf(xyz, t)
        x,y,z = xyz .- o
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        return max(d_radial, max(d_top,d_bottom))
    end
    function plate_sdf(xyz,t)
        dot(xyz,n)
    end
    cylinder = AutoBody(cylinder_sdf)
    plate = AutoBody(plate_sdf)
    body = plate + cylinder
    Simulation(L.*D, UBC, D; U=U, ν=ν,body=body, mem=backend, perdir=(2,), exitBC=true, T=T, uλ=(i,x) -> UBC(i, x, 0f0))
end

function ThreeD_Cylinder_wPlate_wRot_wBL(D, λ, backend; L, Re, T)
    U = 1  
    ν = U*D/Re
    r = D/2     
    h = D    
    o = SA[2.5D, 5D, 0.5D]   
    n = SA[0, 0, 1]    
    θ₁ = (λ)*(U/r)
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]     
    function UBC(i, xyz, t)
        x, y, z = xyz
        x_scale = max(Float32(10D + (x)), Float32(10D))
        η = max(Float32((z) * √(1 / (ν * x_scale))), 0.f0)
        if i == 1
            return max(Float32(1f0 - 0.1608f0 * exp(-η)
                           - 1.2956f0 * η * exp(-η)
                           - 0.8392f0 * exp(-2f0 * η)), 0f0)
        else
            return 0f0
        end
    end
    function cylinder_sdf(xyz, t)
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        return max(d_radial, max(d_top,d_bottom))
    end
    function plate_sdf(xyz,t)
        return dot(xyz,n)
    end
    cylinder = AutoBody(cylinder_sdf,(xyz,t)->R(θ₁*t)*(xyz-o))
    plate = AutoBody(plate_sdf,(xyz,t)->R(0*t)*(xyz-o))
    body = cylinder + plate
    Simulation(L.*D, UBC, D; U=U, ν=ν,body=body, mem=backend, perdir=(2,), exitBC=true, T=T, uλ=(i,x) -> UBC(i, x, 0f0))
end

function ThreeD_Rotor(D, λ₁, λ₂, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    rₑ = D      
    h = 7D       
    n = SA[0, 0, 1]    
    o₁ = SA[2.5D,4D,9D]  
    o₂ = SA[5.5D,4D,9D]   
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)     
    θ₂ = (λ₂)*(U/r)     
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h
        d_bottom = -z-h
        sdf_rotor = max(d_radial, max(d_top,d_bottom))
        sdf_endplate_top = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z-h)-thk/2)
        sdf_endplate_bottom = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z+h)-thk/2)
        min(sdf_rotor, sdf_endplate_top, sdf_endplate_bottom)
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    Rotor₂ = AutoBody(sdf,(xyz,t)->R(θ₂*t)*(xyz-o₂)) 
    body = Rotor₁ + Rotor₂
    Simulation(L.*D, (U,0,0), D; ν=ν, body=body, mem=backend, perdir=(2,), exitBC=false, T=T)
end

function ThreeD_Rotor_BiotSimulation(D, λ₁, λ₂, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    rₑ = D      
    h = 7D       
    n = SA[0, 0, 1]    
    o₁ = SA[1.5D,2.5D,7.5D]  
    o₂ = SA[4.5D,2.5D,7.5D]   
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)     
    θ₂ = (λ₂)*(U/r)     
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h
        d_bottom = -z-h
        sdf_rotor = max(d_radial, max(d_top,d_bottom))
        sdf_endplate_top = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z-h)-thk/2)
        sdf_endplate_bottom = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z+h)-thk/2)
        min(sdf_rotor, sdf_endplate_top, sdf_endplate_bottom)
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    Rotor₂ = AutoBody(sdf,(xyz,t)->R(θ₂*t)*(xyz-o₂)) 
    body = Rotor₁ + Rotor₂
    return BiotSimulation(L.*D, (U,0,0), D; ν=ν, body=body, mem=backend, T=T), Rotor₁, Rotor₂ 
end

function ThreeD_Rotor_NonBiotFaces(D, λ₁, λ₂, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    rₑ = D      
    h = 7D          
    o₁ = SA[1.5D,2.5D,3.5D]  
    o₂ = SA[4.5D,2.5D,3.5D]   
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)     
    θ₂ = (λ₂)*(U/r)     
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        sdf_rotor = max(d_radial, max(d_top,d_bottom))
        sdf_endplate_top = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z-h/2)-thk/2)
        sdf_endplate_bottom = max(√sum(abs2, SA[x,y,0]) - r, abs(z+h/2)-thk/2)
        min(sdf_rotor, sdf_endplate_top, sdf_endplate_bottom)
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    Rotor₂ = AutoBody(sdf,(xyz,t)->R(θ₂*t)*(xyz-o₂)) 
    body = Rotor₁ + Rotor₂
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T,nonbiotfaces=(-3,)), Rotor₁, Rotor₂ 
end

function ThreeD_Rotor_NonBiotFaces_Single(D, λ₁, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    rₑ = D      
    h = 7D        
    o₁ = SA[1.5D,2.5D,3.5D]  
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)        
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        sdf_rotor = max(d_radial, max(d_top,d_bottom))
        sdf_endplate_top = max(√sum(abs2, SA[x,y,0]) - rₑ, abs(z-h/2)-thk/2)
        sdf_endplate_bottom = max(√sum(abs2, SA[x,y,0]) - r, abs(z+h/2)-thk/2)
        min(sdf_rotor, sdf_endplate_top, sdf_endplate_bottom)
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    body = Rotor₁
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T,nonbiotfaces=(-3,)), Rotor₁
end


function ThreeD_Rotor_Validation(D, λ₁, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    # rₑ = D/2      
    h = 10D        
    o₁ = SA[1.5D,2.5D,6D]  
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)        
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        max(d_radial, max(d_top,d_bottom))
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    body = Rotor₁
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T), Rotor₁
end

function ThreeD_Rotor_Validation_Free_Slip(D, λ₁, U₊, backend; L, Re, T,ϵ=1,thk=2ϵ+√2)
    U = 1 
    ν = U*D/Re
    r = D/2     
    # rₑ = D/2      
    h = 5D       
    o₁ = SA[1.5D,2.5D,2.5D]  
    # Defining rotation:
    θ₁ = (λ₁)*(U/r)        
    R(θ) = SA[cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]    
    function sdf(xyz,t) 
        x,y,z = xyz
        d_radial = √sum(abs2, SA[x,y,0])-r
        d_top = z-h/2
        d_bottom = -z-h/2
        max(d_radial, max(d_top,d_bottom))
    end
    # Body + Rotation:
    Rotor₁ = AutoBody(sdf,(xyz,t)->R(θ₁*t)*(xyz-o₁))
    body = Rotor₁
    return BiotSimulation(L.*D,U₊,D;ν=ν,body=body,mem=backend,T=T,nonbiotfaces=(-3,)), Rotor₁
end

function circle₁(D;Re=500,U=1,mem=Array)
    r=D/2
    o₁ = SA[1.5D,2.5D]
    function sdf₁(xy,t)
        x,y = xy
        √sum(abs2, SA[x,y] .- o₁) - radius
    end
    body = AutoBody(sdf₁)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem)
end




