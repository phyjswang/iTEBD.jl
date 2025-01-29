#-------------------------------------------------------------------------------------------
# Basic Tensor Multiplication
#-------------------------------------------------------------------------------------------
"""
    tensor_lmul!(λ, Γ)

Contraction of:
       |
  --λ--Γ--
"""
function tensor_lmul!(λ::AbstractVector{<:Number}, Γ::AbstractArray)
    α = size(Γ, 1)
    Γ_reshaped = reshape(Γ, α, :)
    lmul!(Diagonal(λ), Γ_reshaped)
end

#-------------------------------------------------------------------------------------------
"""
    tensor_rmul!(Γ, λ)

Contraction of:
    |
  --Γ--λ--
"""
function tensor_rmul!(Γ::AbstractArray, λ::AbstractVector{<:Number})
    β = size(Γ)[end]
    Γ_reshaped = reshape(Γ, :, β)
    rmul!(Γ_reshaped, Diagonal(λ))
end

#-------------------------------------------------------------------------------------------
"""
    tensor_umul(umat, Γ)

Contraction of:
    |
    U
    |
  --Γ--
"""
function tensor_umul(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 3})
    @tensor Γ[:] := umat[-2,1] * Γ[-1,1,-3]
    Γ
end

"""
    tensor_umul(umat, Γ)

Contraction of:
    |
    U
    |
  --Γ--
    |
"""
function tensor_umul(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 4})
    @tensor Γ[:] := umat[-2,1] * Γ[-1,1,-3,-4]
    Γ
end

#-------------------------------------------------------------------------------------------
# Tensor Grouping
#-------------------------------------------------------------------------------------------
"""
    tensor_group(Γs)

Contraction:
    |  |   ...   |           |
  --Γ--Γ-- ... --Γ--  ==>  --Γs--
"""
function tensor_group(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    tensor = Γs[1]
    n = length(Γs)
    for i in 2:n
        χ = size(Γs[i], 1)
        tensor = reshape(tensor, :, χ) * reshape(Γs[i], χ, :)
    end
    α, β = size(Γs[1], 1), size(Γs[n], 3)
    reshape(tensor, α, :, β)
end

"""
    tensor_group(Γs)

Contraction:
    |  |   ...   |           |
  --Γ--Γ-- ... --Γ--  ==>  --Γs--
    |  |         |           |
"""
function tensor_group(Γs::AbstractVector{<:AbstractArray{<:Number, 4}})
    tensor = Γs[1]
    D1, d1, d2, _ = size(tensor)
    n = length(Γs)
    for i in 2:n
        χ, d21, d22, _ = size(Γs[i])
        tensor = reshape(tensor, :, χ) * reshape(Γs[i], χ, :)
        tensor = reshape(tensor, D1, d1, d2, d21, d22, :)
        tensor = PermutedDimsArray(tensor, [1,2,4,3,5,6])
        d1 *= d21
        d2 *= d22
    end
    α, β = size(Γs[1], 1), size(Γs[n], 4)
    reshape(tensor, α, d1, d2, β)
end

#-------------------------------------------------------------------------------------------
# Tensor Decomposition
#-------------------------------------------------------------------------------------------
function _try_svd(m)
    try
        return svd(m)
    catch
        @warn "DivideAndConquer() failed, use QRIteration() instead"
        return svd(m; alg = LinearAlgebra.QRIteration())
    end
end

"""
    svd_trim(mat; maxdim, cutoff, renormalize)

SVD with compression.

Parameters:
-----------
- mat        : matrix
- maxdim     : the maximum number of singular values to keep
- cutoff     : set the desired truncation error of the SVD
- renormalize: renormalize the singular values
"""
function svd_trim(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    cutoff::Real=CUTOFF,
    renormalize::Bool=false
)
    # res = svd(mat)
    res = try
        _try_svd(mat)
    catch e
        fn = tempname() * "_mat.jld2"
        FileIO.save(fn, "mat", mat)
        println(*(fill("=",42)...))
        println("svd error! mat saved to $fn")
        rethrow(e)
    end
    vals = res.S
    (maxdim > length(vals)) && (maxdim = length(vals))
    len::Int64 = 1
    while true
        if isless(vals[len], cutoff)
            len -= 1
            break
        end
        isequal(len, maxdim) && break
        len += 1
    end
    if len == 0
        @show vals
        error("No singular value is larger than the cutoff.")
    end
    U = res.U[:, 1:len]
    S = vals[1:len]
    V = res.Vt[1:len, :]
    if renormalize
        S ./= norm(S)
    end
    U, S, V
end

#-------------------------------------------------------------------------------------------
"""
    tensor_svd(T; maxdim, curoff, renormalize)

Tensor SVD with compression:
    |   |           |     |
  --BLOCK--  ==>  --U--S--V--

Parameters:
-----------
- T          : 4-leg tensor
- maxdim     : the maximum number of singular values to keep
- cutoff     : set the desired truncation error of the SVD
- renormalize: renormalize the singular values
"""
function tensor_svd(
    T::AbstractArray{<:Number, 4};
    maxdim=MAXDIM, cutoff=CUTOFF, renormalize=false
)
    α, d1, d2, β = size(T)
    mat = reshape(T, α*d1, :)
    u, S, v = svd_trim(mat; maxdim, cutoff, renormalize)
    U = reshape(u, α, d1, :)
    V = reshape(v, :, d2, β)
    U, S, V
end

"""
    tensor_svd(T; maxdim, curoff, renormalize)

Tensor SVD with compression:
    |   |           |     |
  --BLOCK--  ==>  --U--S--V--
    |   |           |     |

Parameters:
-----------
- T          : 6-leg tensor
- maxdim     : the maximum number of singular values to keep
- cutoff     : set the desired truncation error of the SVD
- renormalize: renormalize the singular values
"""
function tensor_svd(
    T::AbstractArray{<:Number, 6};
    maxdim=MAXDIM,
    cutoff=CUTOFF,
    renormalize=false
)
    α, d11, d12, d21, d22, β = size(T)
    mat = reshape(T, α*d11*d12, :)
    u, S, v = svd_trim(mat; maxdim, cutoff, renormalize)
    U = reshape(u, α, d11, d12, :)
    V = reshape(v, :, d21, d22, β)
    U, S, V
end

#-------------------------------------------------------------------------------------------
"""
    tensor_decomp!(Γ, λl, n; maxdim, cutoff, renormalize)

Multiple decomposition:
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ--

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_decomp!(
    Γ::AbstractArray{<:Number, 3},
    λl::AbstractVector{<:Real},
    n::Integer;
    maxdim=MAXDIM, cutoff=CUTOFF, renormalize=false
)
    β = size(Γ, 3)
    d = round(Int, size(Γ, 2)^(1/n))
    Γs = Vector{Array{eltype(Γ), 3}}(undef, n)
    λs = Vector{Vector{eltype(λl)}}(undef, n-1)
    Ti, λi = Γ, λl
    for i=1:n-2
        Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
        Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
        tensor_lmul!(1 ./ λi, Ai)
        tensor_rmul!(Ai, Λ)
        tensor_lmul!(Λ, Ti)
        Γs[i] = Ai
        λs[i] = Λ
        λi = Λ
    end
    Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
    Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
    tensor_lmul!(1 ./ λi, Ai)
    tensor_rmul!(Ai, Λ)
    Γs[n-1] = Ai
    λs[n-1] = Λ
    Γs[n] = Ti
    Γs, λs
end

"""
    tensor_decomp!(Γ, λl, n; maxdim, cutoff, renormalize)

Multiple decomposition:
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--
    |              |   |        |
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ--
    |          |

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_decomp!(
    Γ::AbstractArray{<:Number, 4},
    λl::AbstractVector{<:Real},
    n::Integer;
    maxdim=MAXDIM,
    cutoff=CUTOFF,
    renormalize=false
)
    α, d1, _, β = size(Γ)
    d = round(Int, d1^(1/n))
    Γs = Vector{Array{eltype(Γ), 4}}(undef, n)
    λs = Vector{Vector{eltype(λl)}}(undef, n-1)
    Ti, λi = Γ, λl
    for i=1:n-2
        Ti = reshape(Ti, α, d, div(d1,d), d, div(d1,d), β)
        Ti = PermutedDimsArray(Ti, [1,2,4,3,5,6])
        Ai, Λ, Ti = tensor_svd(Ti; maxdim, cutoff, renormalize)
        tensor_lmul!(1 ./ λi, Ai)
        tensor_rmul!(Ai, Λ)
        tensor_lmul!(Λ, Ti)
        Γs[i] = Ai
        λs[i] = Λ
        λi = Λ
        d1 = div(d1, d)
    end
    Ti = reshape(Ti, α, d, div(d1,d), d, div(d1,d), β)
    Ti = PermutedDimsArray(Ti, [1,2,4,3,5,6])
    Ai, Λ, Ti = tensor_svd(Ti; maxdim, cutoff, renormalize)
    tensor_lmul!(1 ./ λi, Ai)
    tensor_rmul!(Ai, Λ)
    Γs[n-1] = Ai
    λs[n-1] = Λ
    Γs[n] = Ti
    Γs, λs
end
