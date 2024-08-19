
function landau_lifshitz_step(mc::MonteCarlo, dt::Float64, T::Float64)
    lat = mc.lattice
    spins = lat.spins
    size = lat.size
    # Create an array to store the updated spin configuration
    out = length(mc.outpath) > 0

    # Loop over each spin in the array
    for i in 1:size
        # Get the current spin value
        spin = get_spin(spins, i)
        local_field = get_local_field(mc.lattice, i)
        new_spin = tuple_add(spin, tuple_s_mult(dt,tuple_cross(spin, local_field)))
        new_spin = tuple_normalized(new_spin)
        set_spin!(lat.spins, new_spin, i)
    end

    # Return the updated spin configuration
    if out
        write_MC_checkpoint_t(mc, T)
    end
end


function get_local_field_spin(spins::Array{Float64, 2},lattice::Lattice, point::Int64)
    @inbounds js = ClassicalSpinMC.get_bilinear_sites(lattice, point)
    @inbounds Js = ClassicalSpinMC.get_bilinear_matrices(lattice, point)
    @inbounds cs = ClassicalSpinMC.get_cubic_sites(lattice, point)
    @inbounds rs = ClassicalSpinMC.get_quartic_sites(lattice, point)
    @inbounds h = ClassicalSpinMC.get_field(lattice, point)
    @inbounds o = ClassicalSpinMC.get_onsite(lattice, point)
    @inbounds Rs = ClassicalSpinMC.get_quartic_tensors(lattice, point)
    @inbounds Cs = ClassicalSpinMC.get_cubic_tensors(lattice, point)

    # sum over all interactions
    Hx = 0.0
    Hy = 0.0
    Hz = 0.0

    # on-site interaction
    @inbounds six = spins[1, point]
    @inbounds siy = spins[2, point]
    @inbounds siz = spins[3, point]

    @inbounds Hx += 2 * ( o.m11 * six + o.m12 * siy + o.m13 * siz)
    @inbounds Hy += 2 * ( o.m21 * six + o.m22 * siy + o.m23 * siz)
    @inbounds Hz += 2 * ( o.m31 * six + o.m32 * siy + o.m33 * siz)

    # bilinear interaction 
    for n in eachindex(js)
        J = Js[n]
        @inbounds sjx = spins[1,js[n]]
        @inbounds sjy = spins[2,js[n]]
        @inbounds sjz = spins[3,js[n]]

        @inbounds Hx += J.m11 * sjx + J.m12 * sjy + J.m13 * sjz 
        @inbounds Hy += J.m21 * sjx + J.m22 * sjy + J.m23 * sjz 
        @inbounds Hz += J.m31 * sjx + J.m32 * sjy + J.m33 * sjz 
    end

    # cubic interaction 
    for n in eachindex(cs)
        C = Cs[n]
        j, k = cs[n]
        @inbounds sj = spins[j]
        @inbounds sk = spins[k]

        @einsum Hx += C[1, a, b] * sj[a] * sk[b]
        @einsum Hy += C[2, a, b] * sj[a] * sk[b]
        @einsum Hz += C[3, a, b] * sj[a] * sk[b]
    end

    # quartic interaction 
    for n in eachindex(rs)
        R = Rs[n]
        j, k, l = rs[n]
        @inbounds sj = spins[j]
        @inbounds sk = spins[k]
        @inbounds sl = spins[l]

        @einsum Hx += R[1, a, b, c] * sj[a] * sk[b] * sl[c]
        @einsum Hy += R[2, a, b, c] * sj[a] * sk[b] * sl[c]
        @einsum Hz += R[3, a, b, c] * sj[a] * sk[b] * sl[c]
    end
    return [Hx-h[1], Hy-h[2], Hz-h[3]]
end

function landau_lifshitz_ODE!(dS::Matrix{Float64}, S::Matrix{Float64},lattice::Lattice, t::Float64)
    for i in 1:size(S, 2)
        dS[:, i] = -cross(S[:, i], get_local_field_spin(S, lattice, i))
    end
end

function time_evolve_euler!(mc::MonteCarlo, T::Float64, dt::Float64)
    nsteps = trunc(Int64, T / dt)
    for i in 1:nsteps
        landau_lifshitz_step(mc, dt, T)
    end
end

function time_evolve!(mc::MonteCarlo, T::Float64)
    lat = mc.lattice
    spins = deepcopy(lat.spins)
    prob = ODEProblem(landau_lifshitz_ODE!, spins, (0.0, T), lat)
    sol = solve(prob, Tsit5(), dt=1e-2)
    return sol
end