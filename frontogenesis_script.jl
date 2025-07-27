using Oceananigans #use v.1.10
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.BoundaryConditions
using CairoMakie 
using NCDatasets





# Model parameters
Nx = 240
Ny = 80
f = 1e-4               # Coriolis frequency [s⁻¹]
L_front = 10kilometers  # Initial front width [m]
aspect_ratio = 100      # L/H
δ = 0                   # Strain ratio (α/f)
Ro = 1               # Rossby number (defines M^2)
F = Inf                 # Froude number (N² = M²/(F²H))
Re_h = 5e2         # horizontal reynolds number
Re_v = +Inf              #vertical reynolds number
n = 2              #diffusivity number                 

sponge_width = 8kilometers
damping_rate = f

# Derived parameters
H_front = L_front/aspect_ratio
α = f*δ
M² = (Ro^2*f^2*L_front)/H_front
N² = (M²*L_front)/(F^2*H_front)
Bu = Ro/F
Δb = M²*L_front
κh = (sqrt(Δb*H_front)*L_front^(n-1))/Re_h
κv = κh*(Re_h/Re_v)*(H_front/L_front)^n


filename = "δ="*string(δ)*"_Ro="*string(Ro)*"_F = "*string(F)*"_Re_h="*string(Re_h)

println("Filename: ", filename)
println("\nDerived parameters:")
println("H = ", H_front/1000, " km")
println("α = ", α, " s⁻¹")
println("M² = ", M², " s⁻²")
println("N² = ", N², " s⁻²")
println("Bu = ", Bu)
println("κh = ", κh)
println("κv = ", κv)




Lx = 16*L_front #40kilometers
Lz = H_front

grid = RectilinearGrid(size = (Nx, Ny), #in ST15 they use Nx = 200*L_front, Nz = 100*H_front
                       x = (-Lx/2, Lx/2),
                       z = (-Lz, 0),
                       topology = (Bounded, Flat, Bounded))




                       # advective forcing term 
u_background = XFaceField(grid)
u_background .= - α * xnodes(grid, Face(), Center(), Center())
background_flow = AdvectiveForcing(u = u_background)

# no addtional u forcing

# v forcing
v_forcing_func(x, z, t, v, α) = - 2*α*v
v_forcing = Forcing(v_forcing_func, parameters = α, field_dependencies = :v )

# w forcing
w_forcing_func(x, z, t, w, α) = - α*w
w_forcing = Forcing(w_forcing_func, parameters = α, field_dependencies = :w )

# b forcing
b_forcing_func(x, z, t, α, b ) = - α*b
b_forcing = Forcing(b_forcing_func, parameters = α, field_dependencies= :b )



Δb = L_front * M²

target_buoyancy_left(x,z,t) = z*N²
target_buoyancy_right(x,z,t) = z*N² + Δb
    




left_mask_3D   = GaussianMask{:x}(center=-grid.Lx/2, width = sponge_width)
left_mask(x,z) = left_mask_3D(x,0,z)
uvw_sponge_left = Relaxation(rate=damping_rate, mask=left_mask)
b_sponge_left = Relaxation(rate=damping_rate, mask=left_mask, target = target_buoyancy_left) 

right_mask_3D  = GaussianMask{:x}(center=grid.Lx/2,width = sponge_width)
right_mask(x,z) = right_mask_3D(x,0,z)
uvw_sponge_right = Relaxation(rate=damping_rate, mask=right_mask)
b_sponge_right = Relaxation(rate=damping_rate, mask=right_mask, target = target_buoyancy_right ) 

#b_sponge = Relaxation(rate=damping_rate, target = target_buoyancy)





using Oceananigans.BoundaryConditions

# Free-slip for u and v (∂u/∂z = ∂v/∂z = 0)
free_slip = FieldBoundaryConditions(
    top = GradientBoundaryCondition(0.0),
    bottom = GradientBoundaryCondition(0.0)
)

# No vertical flow (w = 0 at top/bottom)
no_penetration = FieldBoundaryConditions(
    top = ValueBoundaryCondition(0.0),
    bottom = ValueBoundaryCondition(0.0)
)

velocity_bcs = (
    u = free_slip,
    v = free_slip,
    w = no_penetration
)





∂b∂z_top = N²
∂b∂z_bottom = N²

buoyancy_bcs = FieldBoundaryConditions(
    top = GradientBoundaryCondition(∂b∂z_top),
    bottom = GradientBoundaryCondition(∂b∂z_bottom)
)
    

horizontal_closure = HorizontalScalarDiffusivity(ν=κh, κ=κh )
vertical_closure = VerticalScalarDiffusivity(ν=κv , κ=κv )
closure = (horizontal_closure, vertical_closure)


#with sponge

model = NonhydrostaticModel(; grid,
                coriolis = FPlane(f = f),
                buoyancy = BuoyancyTracer(),
                tracers = :b,
                advection = WENO(),
                forcing = (; u = (background_flow, uvw_sponge_left, uvw_sponge_right),
                             v = (background_flow , v_forcing, uvw_sponge_left, uvw_sponge_right) , 
                             w = (background_flow , w_forcing, uvw_sponge_left, uvw_sponge_right),
                             b = (background_flow, b_forcing, b_sponge_left, b_sponge_right)),
                boundary_conditions = (; b=buoyancy_bcs, velocity_bcs),
                closure = closure
                )


#without sponge 
#=
model = NonhydrostaticModel(; grid,
                coriolis = FPlane(f = f),
                buoyancy = BuoyancyTracer(),
                tracers = :b,
                advection = WENO(),
                forcing = (; u = (background_flow),
                             v = (background_flow , v_forcing) , 
                             w = (background_flow , w_forcing),
                             b = (background_flow, b_forcing))
                #boundary_conditions = (; b=buoyancy_bcs, velocity_bcs)
                )
=#


#inital setup

Δb = L_front * M²       # buoyancy jump across front
ϵb = 1e-2 * Δb   


bᵢ(x, z) = N² * z + Δb * 1/2* (tanh(x/L_front) + 1)  

set!(model, b=bᵢ, u = 0, v = 0, w = 0)  # Start from rest



simulation = Simulation(model, Δt=20minutes, stop_time=10days)



conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, max_Δt=20minutes)




using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, u), maximum(abs, v), maximum(abs, w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(100))




# Output setup
u, v, w = model.velocities
ζ = ∂z(u) - ∂x(w)  # Vorticity in x-z plane
b = model.tracers.b
Ri = N² / (∂z(u)^2 + ∂z(v)^2) 

#=
#For Julia animation
simulation.output_writers[:fields] = JLD2Writer(
    model, (; b, ζ , u, v, w),
    filename=filename * ".jld2",
    schedule=TimeInterval(0.5day),
    overwrite_existing=true
    )
=#


#For python viusalisation
simulation.output_writers[:fields] = NetCDFWriter(
    model, (; b, u, v, w, Ri), filename=filename * ".nc", schedule=TimeInterval(10minutes), overwrite_existing=true)




@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)


