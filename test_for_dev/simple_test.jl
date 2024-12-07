using DrWatson
@quickactivate
using MKL
include("../src/iTEBD.jl")
import .iTEBD: iMPS, product_iMPS, gtrm, applygate!
using LinearAlgebra
using TensorOperations
using KrylovKit

"""
    get_gate(g::Float64, h::Float64, t::Float64)

quantum Ising model with transverse field g and longitudinal field h, with fixed J = -1.0
"""
function get_gate_QIM(g::Float64, h::Float64, t::Float64)
    sz = [1.0/2 0; 0 -1.0/2]
    sx = [0 1.0/2; 1.0/2 0]
    id = [1.0 0; 0 1.0]

    h2 = zeros(2,2,2,2)
    @tensor h2[:] = -1.0 * sz[-1,-3] * sz[-2,-4] - g/2 * sx[-1,-3] * id[-2,-4] - g/2 * id[-1,-3] * sx[-2,-4] - h/2 * sz[-1,-3] * id[-2,-4] - h/2 * id[-1,-3] * sz[-2,-4]

    math2 = reshape(h2,4,4)
    matGhalf = exp(-t/2 * (1.0im) * math2)
    matG = exp(-t * (1.0im) * math2)
    return matGhalf, matG
end


function iTEBDmain(g::Float64, h::Float64, dt::Float64, nt::Int64, maxdim::Int64, cutoff::Float64, renormalize::Bool=true, psi0 = :x)
    lst = (1:nt) .* dt

    Ghalf, G = get_gate_QIM(g, h, dt/2)

    if psi0 == :x
        v0 = [1/√2, 1/√2]
    elseif psi0 == :z
        v0 = [1.0, 0.0]
    else
        error("Invalid initial state")
    end
    imps = product_iMPS(ComplexF64,fill(v0, 2))

    mate = zeros(ComplexF64, nt, 2)

    for ti in 1:nt
        if mod(ti, 100) == 1
            @show ti
        end
        applygate!(
            imps,
            Ghalf,
            1,
            2;
            renormalize = renormalize,
            maxdim = maxdim,
            cutoff = cutoff
        )

        applygate!(
            imps,
            G,
            2,
            1;
            renormalize = renormalize,
            maxdim = maxdim,
            cutoff = cutoff
        )

        applygate!(
            imps,
            Ghalf,
            1,
            2;
            renormalize = renormalize,
            maxdim = maxdim,
            cutoff = cutoff
        )

        Θ = gtrm(
            conj(imps),
            imps
        )
        lsem, _, _ = eigsolve(
            Θ,
            rand(size(Θ,2)),
            2,
            :LM;
            issymmetric = false,
            ishermitian = false,
            eager = true
        )
        if length(lsem) == 1
            mate[ti,1] = lsem[1]
        else
            mate[ti,:] = lsem[1:2]
        end
    end

    lsrf = -2 .* log.(abs.(mate[:,1])) ./ 2
    return lsrf, mate, lst, imps
end

lsnt = [100, 1500]
lsD = [100, 200, 300]
for D in lsD, nt in lsnt
    timeused = @elapsed lsrf, mate, lst, _ = iTEBDmain(0.56, 0.004, 0.04, nt, D, 1e-8, false)
    rslt = @strdict lsrf mate lst D timeused
    fullfn = datadir("D=$D.jld2")
    wsave(fullfn, rslt)
end
