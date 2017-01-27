module SchurPade

export powerm

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
    t = realmatrix ? typeof(sqrt(zero(T))) : typeof(sqrt(complex(zero(T))))
    n = Base.LinAlg.checksquare(A)
    R = zeros(t, n, n)
    tt = typeof(zero(t)*zero(t))
    @inbounds begin
        for j = 1:n
            R[j,j] = realmatrix ? sqrt(A[j,j]) : sqrt(complex(A[j,j]))
            for i = j-1:-1:1
                r::tt = A[i,j]
                @simd for k = i+1:j-1
                    r -= R[i,k]*R[k,j]
                end
                r==0 || (R[i,j] = r / (R[i,i] + R[j,j]))
            end
        end
    end
    return UpperTriangular(R)
end

#   Reference: N. J. Higham and L. Lin, A Schur--Pad\'e Algorithm for
#   Fractional Powers of a Matrix, The University of Manchester 2010.
#   Algorithm in that paper: Algorithm 5.1/SPade.

#   powerm(A,P) computes X = A^P,
#   for arbitrary real P and A a matrix without nonpositive real eigenvalues,
#   by the Schur-Pade algorithm, to within machine epsilon accuracy.
#   powerm(A, P) = [X,NSQ,M] returns the number NSQ of square roots
#   and the degree M of the Pade approximant used.

function powerm{T<:Number}(A::Matrix{T},p::Real,maxsqrt::Int = 64)
    n = Base.LinAlg.checksquare(A)
    # 0th power -> identity mat
    (p == 0) && return (eye(A), 0, 0)

    # decompose power p into pint, pfrac
    pint = sign(p)*floor(abs(p)) # round towards 0
    pfrac = p - pint    # -1 < pfrac < 1
    (pfrac == 0) && return (A^p, 0, 0)

    # Schur factorisation F (i.e. conjugation to upper-triangular mat)
    # A = F[:vectors]*F[:Schur]*F[:vectors]
    F = schurfact(complex(A))
    # (0 in F[:values]) && warn("The matrix power may not exist, as the matrix is singular")
    # answer will be complex?
    realm = isreal(A) && all(x -> isreal(x) && (real(x) >= 0), F[:values])
    # realm || warn("The principal matrix power is not defined for matrices with "
    #         *"negative real eigenvalues. A non-principal power "
    #         *"will be returned.")
    S = realm ? real(F[:Schur]) : F[:Schur]
    v = realm ? real(F[:vectors]) : F[:vectors]
    isdiag(F[:Schur]) && return (v*(S.^p)*v', 0, 0)
    n == 2 && return (v*powerm2by2(S,pfrac)*v', 0, 0)

    X, nsq, m = _powerm(S,pfrac,maxsqrt,Val{realm})
    X = v*X*v'
    return (A^pint) * X, nsq, m
end

#   _powerm specialised to the situation A upper-triangular and -1 < pfrac <1
function _powerm{U<:Number,realm}(A::Matrix{U},pfrac::Real,maxsqrt::Int,::Type{Val{realm}})
    n = Base.LinAlg.checksquare(A)

    # take sqrts until matrix has norm close to 1
    # the closer the norm, the more easily approximated
    # chose Pade degree trading off cost of sqrts with cost of Pade approx
    # (lower Pade degree, lower cost to calculation of Pade approx)
    # subject to answer coming out with Float64-accuracy
    thetam = [ # Max norm(I-X) such that degree m Pade approximant to (I-X)^p
                # has Float64 = 2^-53 accuracy
          1.512666672122460e-005    # m = 1
          2.236550782529778e-003    # m = 2
          1.882832775783885e-002    # m = 3 being used
          6.036100693089764e-002    # m = 4 being used
          1.239372725584911e-001    # m = 5 being used
          1.998030690604271e-001    # m = 6 being used
          2.787629930862099e-001    # m = 7 being used
          3.547373395551596e-001    # m = 8
          4.245558801949280e-001    # m = 9
          4.870185637611313e-001    # m = 10
          5.420549053918690e-001    # m = 11
          5.901583155235642e-001    # m = 12
          6.320530128774397e-001    # m = 13
          6.685149002867240e-001    # m = 14
          7.002836650662422e-001    # m = 15
          7.280253837034645e-001    # m = 16
          9.152924199170567e-001    # m = 32
          9.764341682154458e-001 ]; # m = 64
    padeDegree = normdiff -> findfirst(x -> normdiff <= x, thetam[3:7])::Int + 2

    T = UpperTriangular(A)
    nsq = 0 # number of sqrts computed
    m = 16 # (max Pade degree reasonably possible in Float64)
    while nsq < maxsqrt
        normdiff = norm(T-I,1)
        if normdiff <= thetam[7]
            # we have a close-to-reasonable Pade degree
            m = padeDegree(normdiff)
            if m - padeDegree(normdiff/2) > 0
                # taking another sqrt would reduce the Pade degree further
                # (n.b. at most 1 extra sqrt can be worth it)
                T = _sqrtm(T,Val{realm})
                nsq += 1
                m = padeDegree(norm(T-I,1))
            end
            break
        end
        T = _sqrtm(T,Val{realm})
        nsq += 1
    end
    #return X, nsq, m

    # compute the [m/m] Pade approximant of matrix power (I-T)^p
    B = I - T
    k = 2*m
    X = I + _coeff(pfrac,k)*B
    for j = (k-1):-1:1 # bottom-up
        X = I + _coeff(pfrac,j) * (X\B)
    end

    # repeated sqr
    for s in 0:nsq
        if s != 0
            X = X*X
        end
        for i in 1:(n-1)
            X[i:i+1,i:i+1] = _powerm2by2(A[i:i+1,i:i+1],pfrac/(2^(nsq-s)))
        end
    end
    return X, nsq, m
end

# _coeff evaluate continued fraction representation in bottom-up fashion
function _coeff(p::Real,i::Int)::Real
    (i == 1) && return -p
    j = i/2
    if mod(i,2) == 0
        c = (-j + p) / (2*(2*j-1))
    else
        j = floor(j)
        c = (-j - p) / (2*(2*j+1))
    end
    return c
end

# _powerm2by2    Power of 2-by-2 upper triangular matrix.
#   POWERM2BY2(A,p) is the pth power of the 2-by-2 upper
#   triangular matrix A, where p is an arbitrary real number.
function _powerm2by2{T<:Number}(A::Array{T,2},p::Real)
    a1 = A[1,1]
    a2 = A[2,2]
    a1p = a1^p
    a2p = a2^p
    loga1 = log(a1)
    loga2 = log(a2)

    X = diagm([a1p,a2p])

    if a1 == a2
        X[1,2] = p*A[1,2] * a1^(p-1)
    elseif abs(a1) < 0.5*abs(a2) || abs(a2) < 0.5*abs(a1)
        X[1,2] =  A[1,2] * (a2p - a1p) / (a2 - a1)
    else # close eigenvalues
        w = atanh((a2-a1)/(a2+a1)) + 1im*pi*ceil((imag(loga2-loga1) - pi)/(2*pi))
        dd = 2 * exp(p*(loga1+loga2)/2) * sinh(p*w) / (a2-a1)
        X[1,2] = A[1,2]*dd
    end
    return X
end

end
