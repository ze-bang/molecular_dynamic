using DifferentialEquations
prob = ODEProblem((u, p, t) -> u, (p, t0) -> p[1], (p) -> (0.0, p[2]), (2.0, 1.0))
sol = solve(prob, Tsit5())
using Plots; gr()
plot(sol)