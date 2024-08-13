using LinearAlgebra
using DifferentialEquations
using MPI
using ClassicalSpinMC
using HDF5
using Einsum
include("pyrochlore.jl")
include("../parallel_tempering/landau_lifshitz.jl")
#------------------------------------------
# constants
#------------------------------------------
k_B = 1/11.6 # meV/K
mu_B = 0.67*k_B # K/T to meV/T



#------------------------------------------
# local z axis for each basis site 
#------------------------------------------
z1 = [1, 1, 1]/sqrt(3)
z2 = [1,-1,-1]/sqrt(3)
z3 = [-1,1,-1]/sqrt(3)
z4 = [-1,-1,1]/sqrt(3)



#------------------------------------------
# local x axis for each basis site 
#------------------------------------------
y = [[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]]/sqrt(2)

#------------------------------------------
# local y axis for each basis site 
#------------------------------------------
x = [[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]]/sqrt(6)



#------------------------------------------
# set MC parameters
#------------------------------------------
t_thermalization= Int(1e6)
t_deterministic = Int(1e6)
t_measurement= Int(1e6)
probe_rate=2000
swap_rate=50
overrelaxation=10
report_interval = Int(1e4)
checkpoint_rate=1000

function simulated_annealing_pyrochlore(Jxx, Jyy, Jzz, gxx, gyy, gzz, B, n, T0, Target, L, S, outpath, outprefix)
    
    #h in tesla
    h1 = mu_B*B*(n'*z1) * [gxx, gyy, gzz]
    h2 = mu_B*B*(n'*z2) * [gxx, gyy, gzz]
    h3 = mu_B*B*(n'*z3) * [gxx, gyy, gzz]
    h4 = mu_B*B*(n'*z4) * [gxx, gyy, gzz]
    
    # create unit cell 
    P = Pyrochlore()

    # add Hamiltonian terms
    interactions = Dict("Jxx"=>Jxx, "Jyy"=>Jyy, "Jzz"=>Jzz)
    addInteractionsLocal!(P, interactions) # bilinear spin term 
    addZeemanCoupling!(P, 1, h1) # add zeeman coupling to each basis site
    addZeemanCoupling!(P, 2, h2)
    addZeemanCoupling!(P, 3, h3)
    addZeemanCoupling!(P, 4, h4)
    # generate lattice
    lat = Lattice((L,L,L), P, S, bc="periodic") 

    params = Dict("t_thermalization"=>t_thermalization, "t_deterministic"=>t_deterministic, "t_measurement"=>t_measurement, 
                    "probe_rate"=>probe_rate, "swap_rate"=>swap_rate, "overrelaxation_rate"=>overrelaxation, 
                    "report_interval"=>report_interval, "checkpoint_rate"=>checkpoint_rate)

    # create MC object 
    mc = MonteCarlo(Target, lat, params, outpath=outpath, outprefix=outprefix)

    # perform MC tasks 
    simulated_annealing!(mc, x ->T0*0.9^x, T0)
    deterministic_updates!(mc) 

    # write to file 
    write_MC_checkpoint(mc)
    return mc
end

function parallel_tempering_pyrochlore(Jxx, Jyy, Jzz, gxx, gyy, gzz, B, n, Tmin, Tmax, L, S, outpath, outprefix)
    MPI.Initialized() || MPI.Init()
    commSize = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # get temperature on rank 
    # create equal logarithmically spaced temperatures and get temperature on current rank 
    temp = exp10.(range(log10(Tmin), stop=log10(Tmax), length=commSize) ) 
    T = temp[rank+1]
    #h in tesla
    h1 = mu_B*B*(n'*z1) * [gxx, gyy, gzz]
    h2 = mu_B*B*(n'*z2) * [gxx, gyy, gzz]
    h3 = mu_B*B*(n'*z3) * [gxx, gyy, gzz]
    h4 = mu_B*B*(n'*z4) * [gxx, gyy, gzz]
    
    # create unit cell 
    P = Pyrochlore()

    # add Hamiltonian terms
    interactions = Dict("Jxx"=>Jxx, "Jyy"=>Jyy, "Jzz"=>Jzz)
    addInteractionsLocal!(P, interactions) # bilinear spin term 
    addZeemanCoupling!(P, 1, h1) # add zeeman coupling to each basis site
    addZeemanCoupling!(P, 2, h2)
    addZeemanCoupling!(P, 3, h3)
    addZeemanCoupling!(P, 4, h4)
    # generate lattice
    lat = Lattice((L,L,L), P, S, bc="periodic") 

    params = Dict("t_thermalization"=>t_thermalization, "t_deterministic"=>t_deterministic, "t_measurement"=>t_measurement, 
                    "probe_rate"=>probe_rate, "swap_rate"=>swap_rate, "overrelaxation_rate"=>overrelaxation, 
                    "report_interval"=>report_interval, "checkpoint_rate"=>checkpoint_rate)

    # create MC object 
    mc = MonteCarlo(Target, lat, params, outpath=outpath, outprefix=outprefix)

    # perform MC tasks 
    parallel_tempering!(mc, [0]) # output measurements on rank 0
    return mc
end

function scan_line(Jxx, Jyy, Jzz, gxx, gyy, gzz, hmin, hmax, nScan, n, Target, L, S)
    dirString = ""
    if n == [1, 1, 1]/sqrt(3)
        dirString = "111"
    elseif n == [1, 1, 0]/sqrt(2)
        dirString = "110"
    elseif n == [0, 0, 1]
        dirString = "001"
    end
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    hs = LinRange(hmin, hmax, nScan)

    nb = nScan/size

    leftK = Int16(rank*nb)
    rightK = Int16((rank+1)*nb)

    currJH = hs[leftK+1:rightK]
    outpath   = string(pwd(), "/Jxx_", Jxx, "_Jyy_", Jyy, "_Jzz_", Jzz, "_gxx_", gxx, "_gyy_", gyy, "_gzz_", gzz)
    for i in currJH
        outprefix = string("/h_",dirString,"=",i)
        run_pyrochlore(Jxx, Jyy, Jzz, gxx, gyy, gzz, i, n, Target, L, S, outpath, outprefix)
    end

end

function convergence_field(n)
    scan_line(-0.6, 1.0, -0.6, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 4, 1/2)
    scan_line(-0.2, 1.0, -0.2, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 4, 1/2)
    scan_line(0.2, 1.0, 0.2, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 4, 1/2)
    scan_line(0.6, 1.0, 0.6, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 4, 1/2)
end

# run_pyrochlore(-0.6, 1.0, -0.6, 0, 0, 1, 0.0, [1, 1, 1]/sqrt(3), 1e-7, 1, 1/2, "test", "")

# scan_line(-0.6, 1.0, -0.6, 0, 0, 1, 0.0, 1.0, 40, [1, 1, 0]/sqrt(2), 1e-7, 4, 1/2)
# scan_line(0.6, 1.0, 0.6, 0, 0, 1, 0.0, 1.0, 40, [1, 1, 0]/sqrt(2), 1e-7, 2, 1/2)
# scan_line(-0.6, 1.0, -0.6, 0, 0, 1, 0.0, 1.0, 40, [0, 0, 1], 1e-7, 2, 1/2)
# scan_line(0.6, 1.0, 0.6, 0, 0, 1, 0.0, 1.0, 40, [0, 0, 1], 1e-7, 2, 1/2)

# convergence_field([1,1,1]/sqrt(3))
# convergence_field([1,1,0]/sqrt(2))
# convergence_field([0,0,1]) 

prefix = "Ce2Zr2O7"
mc = simulated_annealing_pyrochlore(0.062, 0.063, 0.011, 0, 0, 2.18, 1.5, [1, 1, 0]/sqrt(2), 14*k_B, 0.03*k_B, 8, 1/2, "simulated_annealing_T=0.3K_B110=1.5T_L=8", "Ce2Zr2O7")
sol = time_evolve!(mc, 1e3)

tosave = Array{Float64, 3}(undef, size(sol.u,1), 3, mc.lattice.size)
for i=1:size(sol.u,1)
    tosave[i,:,:] = sol.u[i]
end

file = h5open(string(mc.outpath[1:end-3], prefix, "_time_evolved.h5"), "w")
file["spins"] = tosave
file["t"] = sol.t
file["site_positions"] = mc.lattice.site_positions
close(file)

# parallel_tempering_pyrochlore(0.062, 0.063, 0.011, 0, 0, 2.18, 1.5, [1, 1, 0]/sqrt(2), 0.09*k_B, 14*k_B, 8, 1/2, "simulated_annealing_low_T_B110=1.5T", "Ce2Zr2O7")

# n = [0, 0, 1]
# scan_line(0.6, 1.0, 0.6, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 2, 1/2)
# scan_line(0.2, 1.0, 0.2, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 2, 1/2)
# scan_line(-0.2, 1.0, -0.2, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 2, 1/2)
# scan_line(-0.6, 1.0, -0.6, 0, 0, 1, 0.0, 2.0, 40, n, 1e-7, 2, 1/2)
