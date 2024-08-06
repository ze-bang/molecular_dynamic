using ClassicalSpinMC
using DifferentialEquations
using LinearAlgebra
using HDF5
using Einsum
include("honeycomb.jl")
include("../parallel_tempering/landau_lifshitz.jl")
#------------------------------------------
# set lattice parameters 
#------------------------------------------
L = 36
S = 1.0

#------------------------------------------
# set interaction parameters 
#------------------------------------------
K = -1.0
h = 0.01
h_vec = h*[1,1,1]/sqrt(3)
inparams = Dict("K"=>K, "h"=>h)  # dictionary of human readable input parameters for output 

#------------------------------------------
# set MC parameters 
#------------------------------------------
mcparams  = Dict( "t_thermalization" => Int(1e4),
                  "t_deterministic" => Int(1e6),
                  "overrelaxation_rate"   => 10)
outpath   = string(pwd(), "/kitaev_h111_0.01_L=36/")

# target temperature
T = 1e-3

# generate honeycomb unit cell
H = Honeycomb()

# add Hamiltonian terms
addInteractionsKitaev!(H, Dict("K"=>K))
addZeemanCoupling!(H, 1, h_vec)
addZeemanCoupling!(H, 2, h_vec)

# create lattice 
kitaevlattice = Lattice( (L,L), H, S) 

# initialize MC struct

num_runs = 1000

for i in 1:num_runs
    prefix = string("configuration",i)
    mc = MonteCarlo(T, kitaevlattice, mcparams, outpath=outpath, outprefix=prefix, inparams=inparams)

    # perform MC tasks 
    simulated_annealing!(mc, x ->1.0*0.9^x, 1.0)
    # deterministic_updates!(mc)

    # write to file 
    write_MC_checkpoint(mc)
    lat = mc.lattice
    spins = lat.spins


    print(get_magnetization(lat))

    prob = ODEProblem(landau_lifshitz_ODE!, spins, (0.0, 1000), lat)
    sol = solve(prob, Tsit5(), dt=1e-2)

    tosave = Array{Float64, 3}(undef, size(sol.u,1), 3, mc.lattice.size)
    for i=1:size(sol.u,1)
        tosave[i,:,:] = sol.u[i]
    end

    file = h5open(string(mc.outpath[1:end-3], prefix, "_time_evolved.h5"), "w")
    file["spins"] = tosave
    file["t"] = sol.t
    file["site_positions"] = mc.lattice.site_positions
    close(file)
end