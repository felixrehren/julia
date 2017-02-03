using Primes

import Base.LinAlg.sylvester

sylvester(a::Union{Real,Complex},b::Union{Real,Complex},c::Union{Real,Complex}) = -c / (a + b)

function _sqrtm(A::UpperTriangular)
    realmatrix = false
    if isreal(A)
        realmatrix = true
        for i = 1:Base.LinAlg.checksquare(A)
            if real(A[i,i]) < 0
                realmatrix = false
                break
            end
        end
    end
    _sqrtm(A,Val{realmatrix})
end
function _sqrtm{T,realmatrix}(A::UpperTriangular{T},::Type{Val{realmatrix}})
    B = A.data
    n = Base.LinAlg.checksquare(B)
    t = realmatrix ? typeof(sqrt(zero(T))) : typeof(sqrt(complex(zero(T))))
    R = zeros(t, n, n)
    tt = typeof(zero(t)*zero(t))
    @inbounds for j = 1:n
        R[j,j] = realmatrix ? sqrt(B[j,j]) : sqrt(complex(B[j,j]))
        for i = j-1:-1:1
            r::tt = B[i,j]
            @simd for k = i+1:j-1
                r -= R[i,k]*R[k,j]
            end
            r==0 || (R[i,j] = sylvester(R[i,i],R[j,j],-r))
        end
    end
    return UpperTriangular(R)
end

function rootm(A::Matrix,q::Int)
    (q < 0) && return inv(rootm(A,-q))
    (q == 0) && error("What is the 0th root? Come on")
    (q == 1) && return A
    #(q == 2) && return sqrtm(A)

    F = schurfact(complex(A)) # get triangular, not quasi-triangular, Schur
    realm = isreal(A) && all(x -> isreal(x) && (real(x) >= 0), F[:values])
    S = realm ? real(F[:Schur]) : F[:Schur]
    v = realm ? real(F[:vectors]) : F[:vectors]
    isdiag(S) && return v*(S.^(1/q))*v'

    X = UpperTriangular(S)
    for p in Primes.factor(Vector,q)
        # X = (p == 2) ? _sqrtm(X,Val{realm}) : _rootm(X,p,Val{realm})
        X = _rootm(X,p,Val{realm})
    end
    return v*X*v'
end

# Let X = (x_ij) be an upper-triangular matrix.
# Then X^p = (x_p;ij) is
#     x_p;ij = x_ij * sum(x_ii^(p-1-q)*x_jj^q for q in 0:(p-1))
#         + sum(x_ii^(q-1)*sum(x_ik*x_{p-q;kj} for k in (i+1):(j-1)) for q in 1:(p-1))
# This recursion allows us to solve for x_ij given p and X^p,
# i.e. find the pth root of a given matrix.
# Our starting point: on the diagonal, x_ii is the pth root of x_p;ii.
# Then, with increasing distance from the diagonal,
# we compute x_q;ik and x_q;kj for 1<q<p and i<k<j
# (using x_ik, x_kj, which are closer to the diagonal than x_ij)
# and derive x_ij by rearranging the above recurrence relation.

# In the below code, X^p is A,
# x_q;ij is R[q,i],
# sum(x_ik*x_{p-q;kj} for k in (i+1):(j-1)) is B[q,i].
# (R and B are indep. of j because, once finished with col. j,
# we never need it again and can overwrite with numbers for col. j+1)

function _rootm{realm}(A::UpperTriangular,p::Int,::Type{Val{realm}})
    z = zero(A[1,1]^(1/p))
    z = realm ? z : complex(z)
    t = typeof(z)
    n = Base.LinAlg.checksquare(A)
    X = zeros(t,n,n)
    R = Matrix{t}(p-1,n)
    B = Matrix{t}(p-1,n)

    @inbounds for j in 1:n
        # compute the diagonal
        X[j,j] = c = xjj = A[j,j]^(1/p)
        for q in 1:(p-1)
            # R[q,j] = X[j,j]^q
            R[q,j] = c
            B[q,j] = z
            c *= xjj
        end

        for i in (j-1):-1:1
            # pre-compute the helper sum B[q,i]
            if i+1 == j
                B[:,i] .= z
            else
                for q in 1:(p-1)
                    # B[q,i] = sum(X[i,k]*R[q,k] for k in (i+1):(j-1))
                    d = z
                    for k in (i+1):(j-1)
                        d += X[i,k]*R[q,k]
                    end
                    B[q,i] = d
                end
            end

            # find xij
            xii = X[i,i]
            r = xjj/xii
            xij = A[i,j]
            c = one(xii)
            @simd for q in 1:(p-1)
                # xij -= xii^(q-1)*B[p-q,i]
                xij -= c*B[p-q,i]
                c *= xii
            end
            # d = sum(xii^(p-1-q)*xjj^q for q in 0:(p-1))
            d = c # = xii^(p-1)
            for q in 1:(p-1)
                c *= r
                d += c
            end
            X[i,j] = xij /= d
            
            # if not finished with this column, compute the helper sum R[q,i]
            if i > 1
                R[1,i] = xij
                for q in 2:(p-1)
                    # R[q,i] = xij*sum(xii^(q-1-s)*xjj^s for s in 0:(q-1))
                    #        + sum(xii^(s-1)*B[s,j] for s in 1:(q-1))
                    c = xii^(q-1)
                    d = e = z
                    f = one(xii)
                    for s in 0:(q-1)
                        d += c
                        c *= r
                    end
                    for s in 1:(q-1)
                        e += f*B[s,j]
                        f *= xii
                    end
                    R[q,i] = xij*d + e
                end
            end
        end
    end
    X
end
