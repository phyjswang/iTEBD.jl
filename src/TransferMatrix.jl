#---------------------------------------------------------------------------------------------------
# General transfer matrix
#---------------------------------------------------------------------------------------------------
function gtrm(
    T1::AbstractArray{<:Number, 3},
    T2::AbstractArray{<:Number, 3}
)
    i1, j1, k1 = size(T1)
    i2, j2, k2 = size(T2)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-1,1,-3] * T2[-2,1,-4]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
function gtrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = gtrm(T1s[1], T2s[1])
    for i=2:n
        M = M * gtrm(T1s[i], T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
trm(T) = gtrm(T, T)
#---------------------------------------------------------------------------------------------------
# Operator transfer matrix
#---------------------------------------------------------------------------------------------------
function otrm(
    T1::AbstractArray{<:Number, 3},
    O::AbstractMatrix{<:Number},
    T2::AbstractArray{<:Number, 3}
)
    i1, j1, k1 = size(T1)
    i2, j2, k2 = size(T2)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(O), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-1,1,-3] * O[1,2] * T2[-2,2,-4]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::AbstractMatrix{<:Number},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], O, T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], O, T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::Vector{<:AbstractMatrix{<:Number}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], O[1], T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], O[i], T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
otrm(mps1::iMPS, O::AbstractMatrix, mps2::iMPS) = otrm(mps1.Γ, O, mps2.Γ)
#---------------------------------------------------------------------------------------------------
# Dominent eigensystem
#---------------------------------------------------------------------------------------------------
export dominent_eigen
function dominent_eigen(
    mat::AbstractMatrix
)
    vals, vecs = eigen(mat)
    vals_abs = abs.(vals)
    pos = argmax(vals_abs)
    vals[pos], vecs[:, pos]
end
#---------------------------------------------------------------------------------------------------
export dominent_eigval
function dominent_eigval(
    mat::AbstractMatrix
)
    vals = eigvals(mat)
    maximum(abs.(vals))
end
#---------------------------------------------------------------------------------------------------
function dominent_eigvecs(
    mat::AbstractMatrix;
    tol::AbstractFloat=SORTTOL
)
    vals, vecs = eigen(mat)
    vals_abs = abs.(vals)
    pos = argmax(vals_abs)
    right_vec = vecs[:,pos]
    left_vec = inv(vecs)[pos,:]
    vals[pos], right_vec, left_vec
end
#---------------------------------------------------------------------------------------------------
export inner_product
inner_product(T) = dominent_eigval(trm(T))
inner_product(T1, T2) = dominent_eigval(gtrm(T1, T2))
inner_product(T1, O, T2) = dominent_eigval(otrm(T1, O, T2))
#---------------------------------------------------------------------------------------------------
# Symmetry representation
#---------------------------------------------------------------------------------------------------
export symrep
function symrep(
    T,
    U;
    tr::Bool=false
)
    tT = tr ? conj.(T) : T
    M = otrm(T, U, tT)
    de, dv = dominent_eigen(M)
    χ = round(Int, sqrt(length(dv)))
    reshape(dv, χ, χ)
end

