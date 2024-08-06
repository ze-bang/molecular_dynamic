using ClassicalSpinMC
using MPI
using DifferentialEquations
using LinearAlgebra
using HDF5
include("honeycomb.jl")
include("../parallel_tempering/landau_lifshitz.jl")

include("pyrochlore.jl")
include("input_file.jl")

# initialize MPI
MPI.Initialized() || MPI.Init()
commSize = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# get temperature on rank 
# create equal logarithmically spaced temperatures and get temperature on current rank 
temp = exp10.(range(log10(0.001), stop=log10(2), length=commSize) ) 
T = temp[rank+1] # target temperature for rank

# read command line arguments 
path = length(ARGS) == 0 ? string(pwd(),"/") : ARGS[1] 
B = 0.0


#------------------------------------------
# set lattice parameters 
#------------------------------------------
L = 8
S = 1.0

#------------------------------------------
# set interaction parameters 
#------------------------------------------
K = -1.0
h = 0.1
h_vec = h*[1,1,1]/sqrt(3)
inparams = Dict("K"=>K, "h"=>h)  # dictionary of human readable input parameters for output 

#------------------------------------------
# set MC parameters 
#------------------------------------------
outpath   = string(pwd(), "/kitaev_h111_0.1_T_scan/")

# target temperature

# generate honeycomb unit cell
H = Honeycomb()

# add Hamiltonian terms
addInteractionsKitaev!(H, Dict("K"=>K))
addZeemanCoupling!(H, 1, h_vec)
addZeemanCoupling!(H, 2, h_vec)

# create lattice 
kitaevlattice = Lattice((L,L), H, S) 

params = Dict("t_thermalization"=>t_thermalization, "t_measurement"=>t_measurement, 
                "probe_rate"=>probe_rate, "swap_rate"=>swap_rate, "overrelaxation_rate"=>overrelaxation, 
                "report_interval"=>report_interval, "checkpoint_rate"=>checkpoint_rate)

# create MC object 
mc = MonteCarlo(T, kitaevlattice, params, outpath=outpath)

# perform MC tasks 
parallel_tempering!(mc, [0]) # output measurements on rank 0