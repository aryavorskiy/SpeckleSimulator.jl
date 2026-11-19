using SpeckleSimulator, CairoMakie, Distributions, Random
Random.seed!(20261021)

const LAMBDAS = [1, 3, 10, 15, 30, 100]
const NSAMPLES = 100_000

sample_approx(λ, n) = [SpeckleSimulator._sample_poisson(Val(true), Int, λ) for _ in 1:n]

fig = Figure(size=(1200, 800))
for (k, λ) in enumerate(LAMBDAS)
    exact = rand(Poisson(λ), NSAMPLES)
    approx = sample_approx(λ, NSAMPLES)
    ax = Axis(fig[fld1(k, 3), mod1(k, 3)];
        title="λ = $(round(Int, λ))",
        xlabel="Poisson quantiles",
        ylabel="Approximation quantiles",
        aspect=1)
    qqplot!(ax, exact, approx; qqline=:identity, markersize=6)
    println("λ = $λ: mean ", round(mean(approx), digits=3), " (", λ, "), var ",
        round(var(approx), digits=3), " (", λ, ")")
end
Label(fig[0, :], "Poisson vs. Gaussian approximation in readout", fontsize=20)

save(joinpath(@__DIR__, "poisson_qq.png"), fig)
fig
