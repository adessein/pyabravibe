# -*- coding: utf-8 -*-
"""
Python version of Anders Brandt AbraVibe Matlab Toolbox

ABRAVIBE
A MATLAB/Octave toolbox for Noise and Vibration Analysis and Teaching
Revision 1.2

Anders Brandt
Department of Technology and Innovation
University of Southern Denmark
abra@iti.sdu.dk


Converted to Python by

Arnaud Dessein
Siemens Gamesa Renewable Energy A/S
arnaud.dessein@siemens.com
"""

import numpy as np
from math import pi
from numpy.linalg import inv
from scipy import linalg


def alinspec(y, fs, w, M=1, ovlp=0):
    """
    %ALINSPEC Calculate linear (rms) spectrum from time data
    %
    %           [Lyy,f] = alinspec(y,fs,w,M,ovlp)
    %
    %           Lyy         Linear spectrum of time signal y
    %           f           Frequency vector for Pyy, N/2+1-by-1
    %
    %           y           Time data in column vector(s). If more than one
    %                       column, each column is treated separately
    %           fs          Sampling frequency for y
    %           w           Time window with length(FFT blocksize), power of 2
    %                       (1024, 2048,...)
    %           M           Number of averages (FFTs), default is 1
    %           ovlp        Overlap in percent, default is 0
    %
    %           D           Number of vectors (columns) in y
    %
    % Example:
    %           [Lyy,f]=alinspec(y,1000,aflattop(1024),10,50)
    % Computes a linear spectrum using a flattop window with 1024 blocksize, 10
    % averages, with 50% overlap
    %
    % ALINSPEC produces a linear, rms weighted spectrum as if y was a periodic
    % signal. A peak in Lyy is interpreted as a sine at that frequency with an
    % RMS value corresponding to the peak value in Lyy.
    %
    % See also winacf apsdw ahann aflattop

    % Copyright (c) 2009-2011 by Anders Brandt
    % Email: abra@iti.sdu.dk
    % Version: 1.0 2011-06-23
    % This file is part of ABRAVIBE Toolbox for NVA
    """

    # Make copy of input arrays in order to preserve them
    _y = np.copy(y)

    # Set up parameters
    N = len(w)  # FFT block size
    df = fs/N   # Frequency increment
    acf = len(w)/sum(w)  # Window amplitude correction factor
    K = int(np.floor((1-ovlp/100)*N))  # Overlap in samples
    _y = np.asfarray(_y)    # Necessary in prython (convert integers to floats)

    if np.shape(_y)[0] < N:
        raise Exception('Not enough data, not even one time block!')

    # Process each time block (column) in _y
    if len(np.shape(_y)) == 1:
        _y = np.reshape(_y, (-1, 1))

    Nsamp, Nvectors = np.shape(_y)
    # Check that specified overlap and number of FFTs does not exhaust data
    L = N+(M-1)*K
    if L > Nsamp:
        raise Exception("Not enough data in y to perform requested number of "
                        "averages!")

    Pyy = np.zeros((N, Nvectors))
    for vec in range(0, Nvectors):
        _y[:, vec] = _y[:, vec] - np.mean(_y[:, vec])     # Remove mean
        n = 0                        # Block number
        i1 = n*K               # Index into x
        i2 = i1+N
        y_tmp = _y[i1:i2, vec]
        Y = acf*np.fft.fft(np.multiply(y_tmp, w)/N)      # Scaled, windowed FFT
        Pyy[:, vec] = np.square(np.abs(Y))      # Window (amplitude) correction
        n = 1                        # Next block number
        i1 = n*K               # Index into y
        i2 = i1+N
        while n <= M:
            y_tmp = _y[i1:i2, vec]
            Y = acf*np.fft.fft(np.multiply(y_tmp, w))/N
            # Linear average accumulation
            Pyy[:, vec] = (n-1)/n*Pyy[:, vec]+np.square(np.abs(Y))/n
            n = n+1
            i1 = n*K           # Index into x
            i2 = i1+N

    # Convert to single-sided spectra and take square root
    Pyy = Pyy[0:int(np.floor(N/2)+1), :]
    Pyy[1:, :] = 2*Pyy[1:, :]
    Lyy = np.sqrt(Pyy)
    f = np.arange(0, int(np.floor(N/2)+1)*df, df)
    return (Lyy, f)


def alinspecp(y, x, fs, w, M=1, ovlp=0):
    """
    %ALINSPECP Calculate linear (rms) spectrum of time data, with phase
    %
    %           [Lyx,f] = alinspecp(y,x,fs,w,M,ovlp)
    %
    %           Lyx        Linear spectrum of time signal y with phase from Gyx
    %           f          Frequency vector for Lyx, N/2+1-by-1
    %
    %           y          Time data in column vector(s). If more than one
    %                      column, each column is treated separately
    %           x          Time data for phase reference
    %           fs         Sampling frequency for y
    %           w          Time window with length(FFT blocksize), power of 2
    %                      (1024, 2048,...)
    %           M          Number of averages (FFTs), default is 1
    %           ovlp       Overlap in percent, default is 0
    %
    %           D          Number of vectors (columns) in y
    %
    % Example:
    %           [Lyx,f]=alinspecp(y,x,1000,aflattop(1024),10,50)
    %
    % ALINSPECP produces a linear, rms weighted spectrum as if y was a periodic
    % signal. A peak in Lyx is interpreted as a sine at that frequency with an
    % RMS value corresponding to the peak value in Lyx and with phase relative
    % to signal x.
    %
    % See also alinspec winacf apsdw ahann aflattop

    % Copyright (c) 2009-2011 by Anders Brandt
    % Email: abra@iti.sdu.dk
    % Version: 1.0 2011-06-23
    %          1.1 2011-10-07 Fixed new syntax, was not working
    % This file is part of ABRAVIBE Toolbox for NVA

    % Set up depending on input parameters
    """

    # Make copy of input arrays in order to preserve them
    _y = np.copy(y)

    # Set up parameters
    N = len(w)  # FFT block size
    df = fs/N   # Frequency increment
    acf = len(w)/sum(w)  # Window amplitude correction factor
    K = int(np.floor((1-ovlp/100)*N))  # Overlap in samples
    _y = np.asfarray(_y)    # Necessary in prython (convert integers to floats)

    if np.shape(_y)[0] < N:
        raise Exception('Not enough data, not even one time block!')

    # Process each time block (column) in _y
    if len(np.shape(_y)) == 1:
        _y = np.reshape(_y, (-1, 1))

    Nsamp, Nvectors = np.shape(_y)
    # Check that specified overlap and number of FFTs does not exhaust data
    L = N+(M-1)*K
    Mmax = (Nsamp-N)/K + 1
    if L > Nsamp:
        raise Exception("Not enough data in y to perform requested number of "
                        "averages! Maximum is {}".format(Mmax))

    Pyy = np.zeros((N, Nvectors))
    Pyx = np.zeros((N, Nvectors), dtype=np.complex128)
    Ayx = np.zeros((N, Nvectors), dtype=np.complex128)

    for vec in range(0, Nvectors):
        _y[:, vec] = _y[:, vec] - np.mean(_y[:, vec])     # Remove mean
        n = 0                        # Block number
        i1 = n*K               # Index into x
        i2 = i1+N
        y_tmp = _y[i1:i2, vec]
        x_tmp = x[i1:i2]
#        print(len(y_tmp),len(w))
        Y = acf*np.fft.fft(np.multiply(y_tmp, w)/N)      # Scaled, windowed FFT
        YX = np.multiply(np.fft.fft(y_tmp), np.conj(np.fft.fft(x_tmp)))
        Pyy[:, vec] = np.square(np.abs(Y))      # Window (amplitude) correction
        Pyx[:, vec] = YX
        n = 1                        # Next block number
        i1 = n*K               # Index into y
        i2 = i1+N
        while n < M:
            y_tmp = _y[i1:i2, vec]
            Y = acf*np.fft.fft(np.multiply(y_tmp, w))/N
            # Linear average accumulation
            Pyy[:, vec] = (n-1)/n*Pyy[:, vec]+np.square(np.abs(Y))/n
            Pyx[:, vec] = (n-1)/n*Pyx[:, vec]+(YX)/n
            n = n+1
            i1 = n*K           # Index into x
            i2 = i1+N
        # Phased power spectrum
        Ayx[:, vec] = np.multiply(Pyy[:, vec],
                                  np.exp(1j*np.angle(Pyx[:, vec])))

    # Convert to single-sided spectra and take square root
    Ayx = Ayx[0:int(np.floor(N/2)+1), :]
    Lyx = np.empty_like(Ayx, dtype=np.complex128)
    Ayx[1:, :] = 2*Ayx[1:, :]
    for vec in range(0, Nvectors):
        Lyx[:, vec] = np.multiply(np.sqrt(np.abs(Ayx[:, vec])),
                                  np.exp(1j*np.angle(Ayx[:, vec])))
    f = np.arange(0, int(np.floor(N/2)+1)*df, df)
    return (Lyx, f)


def mck2frf(f, M, C, K, indof=(0,), outdof=(0,), typefrf='v'):
    """
    %MCK2FRF Calculate FRF(s) from M, C, K matrices
    %
    %       H = mck2frf(f,M,C,K,indof,outdof,type)
    %
    %               H       Frequency response matrix in [(m/s)/N] (matrix) N-by-D-by-R
    %               N       length(f), number of frequency values
    %               D       length(outdof), number of responses
    %               R       length(indof), number of references (inputs)
    %
    %               f       Frequency vector in [Hz]
    %               M       Mass matrix in [kg]
    %               C       Damping matrix in [Ns/m]
    %               K       Stiffness matrix in m/N
    %               indof   Input DOF(s), may be a vector for many reference
    %                       DOFs, (default = (0,)
    %               outdof  Output DOF(2) may be a vector for many responses
    %                       (default = (0,)
    %               typefrf Type of output FRF as string:
    %                       'Flexibility' or 'd' generates displacement/force
    %                       'Mobility' or 'v'    generates velocity/force (Default)
    %                       'Accelerance' or 'a' generates acceleration/force
    % Example:
    %               H = mck2frf(f,M,C,K,[1 2 4],[5:12],'v');
    %
    % Calculates mobilities with columns corresponding to force in
    % DOFs 1, 2, and 4, and responses in DOFs 5 to 12. H will in this case be
    % of dimension (N, 8, 3) where N is the number of frequency values.

    % Copyright (c) 2009-2011 by Anders Brandt
    % Email: abra@iti.sdu.dk
    % Version: 1.0 2011-06-23
    % This file is part of ABRAVIBE Toolbox for NVA
    """

    # Parse Input Parameters
    if typefrf.upper() == 'FLEXIBILITY' :
        typefrf = 'D'
    elif typefrf.upper() == 'MOBILITY' :
        typefrf = 'V'
    elif typefrf.upper() == 'ACCELERANCE' :
        typefrf = 'A'
    elif typefrf.upper() in ['D', 'V', 'A']:
        typefrf = typefrf.upper()
    else:
        raise Exception('Wrong input type!')

    # Find dimensions
    N = len(f)
    D = len(outdof)
    R = len(indof)

    # Allocate H MATRIX for output
    H = np.zeros((N,D,R), dtype=np.complex)

    # Main
    # Loop through frequencies and use inverse of system impedance matrix:
    # B(s)*X(s)=F(s) ==> B(s) in form of B=F/X
    # H(s) = inv(B(s)) ==> X(s)/F(s), so that H(s)*F(s)=X(s)

    for n in range(N):  # Frequency index
        w = 2*pi*f[n]  # Omega for this frequency
        Denom = -(w**2)*M+1j*w*C+K           # Newton's equation in denominator of Hv
        Denom = np.matrix(Denom)
        InvDenom = inv(Denom);    # Inverse denominator
        for r in range(R):
            W = np.ones_like(H[n,:,r])
            W.fill(w)
            if typefrf == 'D':
                H[n,:,r] = InvDenom[outdof,indof[r]]
            elif typefrf == 'V':
                H[n,:,r] = 1j*W*InvDenom[outdof,indof[r]]
            else:
                H[n,:,r] = -(W**2)*InvDenom[outdof,indof[r]]

    return H


def mck2modal(*args):
    """
    % MCK2MODAL     Compute modal model (poles and mode shapes) from M,(C),K
    %
    %       p       Column vector with poles, (or eigenfrequencies if undamped) in rad/s
    %       V       Matrix with mode shapes in columns
    %       Prop    Logical, 1 if C is proportional damping, otherwise 0
    %
    %       M       Mass matrix
    %       C       (Optional) viscous damping matrix
    %       K       Stiffness matrix
    %
    % [p,V] = mck2modal(M,K) solves for the undamped system and returns
    % eigenfrequencies as purely imaginary poles (in rad/s), and mode shapes (normal modes).
    %
    % [p,V] = mck2modal(M,C,K) solves for the poles and mode shapes. If the
    % damping matrix C=aM+bK for konstants a and b, i.e. the system exhibits
    % proportional damping, then the undamped system is solved for mode shapes,
    % and the poles are calculated from the uncoupled equations in modal
    % coordinates. If the damping is not proportional, a general state space
    % formulation is used to find the (complex) mode shapes and poles.
    %
    % NOTE: The list of poles is limited to the poles with positive imaginary
    % part, as the other half of the poles can easily be calculated as the
    % complex conjugates of the first ones.
    %
    % Mode shape scaling:
    % Undamped mode shapes (normal modes) are scaled to unity modal mass
    % Mode shapes calculated with damping are scaled to unity modal A.
    % This means that the modal scaling constant, Qr = 1, that is, that all
    % residues are Apqr=psi_p*psi_q
    % This also means that the mode shapes are complex even for
    % proportionally damped case, but it is the most convenient scaling.
    %
    % See also UMA2UMM

    % Copyright (c) 2009-2011 by Anders Brandt
    % Email: abra@iti.sdu.dk
    % Version: 1.0 2011-06-23
    % This file is part of ABRAVIBE Toolbox for NVA

    % Note: The way we solve the various systems in this file are not
    % at all necessary, but is done for pedagogical reasons.
    % In principal the state space formulation could be used in all cases,
    % and would yield correct results.
    """

    if len(args) == 2:      # Undamped case
        # Solve the undamped case for eigenfrequencies and mode shapes
        M = args[0]
        K = args[1]
        [V, D] = linalg.eig(linalg.solve(M,K))
        [D, I] = np.sort(np.diag(D))    # Sort eigenvalues/frequencies, lowest first
        V = V[:,I]
        p = np.sqrt(-D)             # Poles (with positive imaginary part)
        Prop = None                # Undefined for undamped case!
        Mn = np.diag(V.conj().T*M*V)        # Modal Mass
        wd = np.imag(p)
        for n in range(len(Mn)):
         #  V(:,n)=V(:,n)/sqrt((j*2*wd(n))*Mn(n));    # Which is equivalent to Mr=1/(j2wd)
            V[:,n] = V[:,n]/np.sqrt((Mn[n]));    # Which is equivalent to Mr=1/(j2wd)
    elif len(args) == 3:
        M = args[0]
        C = args[1]
        K = args[2]
        # Find if damping is proportional. See for example
        # Ewins, D. J., Modal Testing: Theory, Practice and Application,
        # Research Studies Press, 2000.
        M1 = linalg.solve(M, K).dot(linalg.solve(M, C))
        M2 = linalg.solve(M, C).dot(linalg.solve(M, K))
        if linalg.norm(M1-M2) < 1e-6:           # If proportional damping
            # Solve the undamped case for mode shapes
            (D,V) = linalg.eig(linalg.solve(M, K))
            D = np.sort(D)    # Sort eigenvalues/frequencies, descending
            I = np.argsort(D)    # Sort eigenvalues/frequencies, descending
            V = V[:, I]
            wn = np.sqrt(D)             # Undamped natural frequencies
            # Now diagonalize M, C, K into modal coordinates
            Mn = np.diag(V.conj().T*M*V)       # Modal Mass
            for n in range(len(Mn)):
                V[:,n] = V[:,n]/np.sqrt(Mn[n])    # Unity modal mass
            Mn = np.diag(np.eye(np.shape(M)[0], np.shape(M)[1]))
            Kn = np.diag(V.conj().T*K*V)       # Modal Stiffness
            Cn = np.diag(V.conj().T*C*V)       # Modal Damping
            z = (Cn/2)/np.sqrt(Kn*Mn) # relative damping from uncoupled equations
            p = -z*wn+1j*wn*np.sqrt(1-z**2)    # Poles (with positive imaginary part)
            Prop=1
            wd=np.imag(p)
            for n in range(len(Mn)):                  # Rescale mode shapes to unity modal A
                V[:,n] = V[:,n]/np.sqrt((1j*2*wd[n]))    # Which is equivalent to Mr=1/(j2wd)
        else:
            # Non-proportional damping, solve state-space formulation
            # See for example:
            # Craig, R.R., Kurdila, A.J., Fundamentals of Structural Dynamics, Wiley 2006
            # With this formulation, coordinates are z={x ; x_dot}
            A = np.vstack((np.hstack((C,M)),np.hstack((M,np.zeros_like(M)))))
            B = np.vstack((np.hstack((K,np.zeros_like(K))),np.hstack((np.zeros_like(M),-M))))
            (D,V) = linalg.eig(B,-A)
            # Sort in descending order
            Dum = np.sort(np.abs(np.imag(D)))
            I = np.argsort(np.abs(np.imag(D)))
            p = D[I]
            V = V[:,I]
            # Rotate vectors to real first element (row 1)
            phi = np.angle(V[1, :])
            phi = np.diag(np.exp(-1j*phi))
            V = V * phi
            # Scale to unity Modal A
            Ma = V.transpose().dot(A).dot(V)
            for col in range(np.shape(V)[1]):
                V[:,col] = V[:,col]/np.sqrt(Ma[col,col])
            # Shorten to size N-by-N. NOTE! This means that in order to use the
            # modal model, you need to recreate the complex conjugate pairs!
            # See, e.g., MODAL2FRF
            [m,n] = np.shape(V)
            p = p[np.arange(0,m,2)]
            V = np.vstack((V[np.arange(0,m/2,dtype=int)],V[np.arange(0,n,2)]))
            Prop = 0
    return (p, V, Prop)