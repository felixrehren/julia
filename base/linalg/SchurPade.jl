module SchurPade

export powerm

#   Reference: N. J. Higham and L. Lin, A Schur--Pad\'e Algorithm for
#   Fractional Powers of a Matrix.
#   MIMS EPrint 2010.91, The University of Manchester, October 2010,
#   revised February 2011.
#   Name of corresponding algorithm in that paper: Algorithm 5.1/SPade.

#   Nicholas J. Higham and Lijing Lin, February 22, 2011.


#   POWERM_PADE(A,P) computes the P'th power X of the matrix A,
#   for arbitrary real P and A with no nonpositive real eigenvalues,
#   by the Schur-Pade algorithm.
#   [X,NSQ,M] = POWERM_PADE(A, P) returns the number NSQ of matrix
#   square roots computed and the degree M of the Pade approximant used.
#   If A is singular or has any eigenvalues on the negative real axis,
#   a warning message is printed.

function powerm{T<:Number}(A::Array{T,2},p::Real)
    # initial checks
    n = size(A,1)
    #@assert n == size(A,2) # square matrix

    if p == 0 # 0th power -> identity mat
        return eye(A), 0, 0
    end

    # decompose power p into pint, pfrac
    pint = sign(p)*floor(abs(p)) # round towards 0
    pfrac = p - pint    # -1 < pfrac < 1
    if pfrac == 0
        return A^p, 0, 0
    end

    # Schur factorisation (i.e. conj. of upper-triangular mat)
    # A = F[:vectors]*F[:Schur]*F[:vectors]
    F = schurfact(A)
    if isdiag(F[:Schur])
        return F[:vectors]*(F[:Schur].^p)*F[:vectors]', 0, 0
    end
    if 0 in F[:values]
        warn("Matrix power may not exist, as the matrix A is singular.")
    end
    if any(x -> isreal(x) && real(x) < 0, F[:values])
        warn("The principal matrix power is not defined for A with "
            *"negative real eigenvalues. A non-principal matrix power "
            *"will be returned.")
    end

    maxsqrt = 64 # ??
    # apply fractional power
    X, nsq, m = _powerm_triang(F[:Schur],pfrac,maxsqrt)
    X = F[:vectors]*X*F[:vectors]'
    return (A^pint) * X, nsq, m
end

#   POWERM_TRIANG   Power of triangular matrix by Pade-based algorithm.
#   POWERM_TRIANG(T,P,MAXSQRT) computes the P'th power X of
#   the upper triangular matrix T, for an arbitrary real number P in the
#   interval (-1,1),and T with no nonpositive real eigenvalues, by a
#   Pade-based algorithm. At most MAXSQRT matrix square roots are computed.
#   [X, NSQ,M] = POWERM_TRIANG(T,P) returns the number NSQ of
#   square roots computed and the degree M of the Pade approximant used.

function _powerm_triang{S<:Number}(A::Array{S,2},pfrac::Real,maxsqrt::Int)
    n = size(A,1)
    if n == 2 # go directly to 2x2 case
        return powerm2by2(A,pfrac), 0, 0
    end

    nsq = 0 # number of sqrts computed
    q = 0
    I = eye(A)

    xvals = [ # Max norm(X) for degree m Pade approximant to (I-X)^p.
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

    # repeated sqrt
    T = A
    local m
    while true
        normdiff = norm(T-I,1)
        if normdiff <= xvals[7]
            q = q+1
            j1 = find(x -> normdiff <= x, xvals[3:7])[1] + 2
            j2 = find(x -> normdiff/2 <= x, xvals[3:7])[1] + 2
            if j1 - j2 <= 1 || q == 2
                m = j1
                break
            end
        end
        if nsq == maxsqrt
            m = 16
            break
        end
        T = sqrtm(T)
        nsq = nsq + 1
    end

    # pade approximant for coefficients
    X = _powerm_cf(I-T,pfrac,m)

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

# _coeff something something pade
function _coeff(p::Real,i::Int)::Real
    if i == 1
        return -p
    end
    jj = i/2
    if mod(i,2) == 0
        c = (-jj + p) / (2*(2*jj-1))
    else
        jj = floor(jj)
        c = (-jj - p) / (2*(2*jj+1))
    end
    return c
end

# _powerm_cf Evaluate Pade approximant bottom up
#   POWERM_CF(Y,p,m) computes the [m/m] Pade approximant of the
#   matrix power (I-Y)^p by evaluating a continued fraction
#   representation in bottom-up fashion.
function _powerm_cf{T<:Number}(A::Array{T,2},pfrac::Real,m::Int)
    k = 2*m
    n = size(A,1)
    I = eye(A)
    S = _coeff(pfrac,k)
    for j = (k-1):-1:1 # bottom-up
        S = _coeff(pfrac,j) * ((I+S)\A)
    end
    return I + S
end

# _powerm2by2    Power of 2-by-2 upper triangular matrix.
#   POWERM2BY2(A,p) is the pth power of the 2-by-2 upper
#   triangular matrix A, where p is an arbitrary real number.
function _powerm2by2{T<:Number}(A::Array{T,2},p::Number)
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
