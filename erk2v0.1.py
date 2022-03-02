#!/usr/bin/python
# -*- coding: utf-8 -*- #
"""
        de la Cruz, H. - A simplified weak simulation method for the 
        probabilistic response analysis of nonlinear random vibration 
        problems
        
        Extended Weak order 2.0 Simplified Runge Kutta scheme.
                
        dX = a(X,t)*dt + b(X, t)*dW
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import expm

__author__ = 'Luiz Antônio Theodoro de Souza'
__version__ = '0.1'

randn = np.random.randn
rand = np.random.random
mm = np.matmul

np.random.seed(100)

def solve_sde(A = None, gamma=None, beta=None, Z0=None, dt=1.0, N=100, t0=0.0, DW=None):
    """
    Sintaxe:
    ----------
    solve_sde(alfa=None, beta=None, X0=None, dt=None, N=100, t0=0, DW=None)
    Parameters:
    ----------
        
        A      : uma função lambda com dois argumentos, o estado Z e o tempo,
                que define a matriz A de linearização.
        gamma  : uma função lambda com dois argumentos, o estado Z e o tempo, 
                que define a parte determinística não-linear da EDE após sua
                semi-linearização.
        beta   : uma função lambda com dois argumentos, o estado Z e o tempo, 
                que define a parte estocástica da EDE.
        Z0     : Condições iniciais da EDE. 
        dt     : O incremento de tempo da solução.
                (default: 1)
        N      : O número de incrementos.
                (default: 100)
        t0     : O tempo inicial da solução.
                (default: 0)
        DW     : O Processo Wiener em notação lambda.
                
    Exemplo:
    ----------
    == Mathematical model representing the dynamics of piezoeletric vibratory
    energy harvest:
    
    f(X) = (1-r)X + dX³
    f'(X) = (1-r) + 2dX²
    dX = V
    dY = -bY + V
    dV = -nV - f(X) - aY + dW
    
    AZx = lambda Z, t: Z[2] 
    AZy = lambda Z, t: -b*Z[1] + Z[2] 
    AZv = lambda Z, t: -n*Z[2] - f'(Z[0]) - a*Z[1]
    
    gZv = lambda X, t: f'(X) - f(X)
    
    A_sdof = lambda Z, t: np.array([AZx(Z, t), AZy(Z, t), AZv(Z, t)])
    gamma_sdof = lambda Z, t: np.array([0, 0, gZv(Z[0])])
    beta_sdof = lambda Z, t: np.array([ 0, 0, -s])
    
    Z0_sdof = [0.0, 0.0, 0.0]
    t, Y = solve_sde(alfa=alfa, beta=beta, X0=X0, dt=0.01, N=10000)
    """
    
    if A is None or gamma is None or beta is None:
        print("Erro: SDE não foi definida.")
        return
    Z, ti = np.zeros((N, len(Z0))), np.arange(N)*dt + t0
    Z[0, :], h = Z0, dt
    for n in range(N-1):
        t = ti[n]
        An = A(Z[n,:], t)
        Anh = An*h
        expAnh = expm(Anh)
        expAnhZn = mm(expAnh, Z[n,:])
        g = gamma(Z[n,:], t)
        b = beta(Z[n,:], t)
        DWn = DW(h)
        K = g*h + b*DWn
        gK = gamma(expAnhZn + K, t)
        Z[n+1,:] = expAnhZn + (g + gK + mm(An,K))*h*0.5 + b*DWn 
    return ti, Z


def MSV_erk2(A=None, gamma=None, beta=None, Z0=None, dt=1.0, N=100, \
             t0=0.0, DW=None, M=100):
    E = np.zeros((M,N,len(Z0)))
    for m in range(M):
        t, Y = solve_sde(A=A, gamma=gamma, beta=beta, Z0=Z0, dt=dt, N=N, t0=t0, DW=DW)
        E[m, :, :] = np.square(Y)
    ret = np.mean(E, axis=0)
    return t, ret


def dist3pts():
    f = lambda x: 0.0 if (x < 2/3) \
                    else 3.0**0.5 if (x >= 2/3) and (x < 5/6) \
                    else -(3.0**0.5)
    return f(np.random.random())

if __name__ == '__main__':
    a, b, n, d, r, s = 0.5625, 0.5, 0.02, 0.0, 0.5, 0.1
    
    f = lambda Z, t: (1-r)*Z[0] + d*Z[0]**3
    flin = lambda Z, t: (1-r) + 2*d*Z[0]**2
    
    gZv = lambda Z, t: flin(Z, t)*Z[0] - f(Z, t)
    
    #AZx = lambda Z, t: Z[2]
    #AZy = lambda Z, t: -b * Z[1] + Z[2]
    #AZv = lambda Z, t: -flin(Z[0], t) - a*Z[1] - n*Z[2]
    
    A_sdof = lambda Z, t: np.array([[0,0,1], [0,-b,1], [-flin(Z, t), -a, -n]])
    gamma_sdof = lambda Z, t: np.array([0, 0, gZv(Z, t)])
    beta_sdof = lambda Z, t: np.array([0, 0, -s])
    dW = lambda dt: dist3pts() * np.sqrt(dt)
    
    h = 0.5
    T = 400
    Z0_sdof = [0.0, 0.0, 0.0]
    M_sdof = 10
    N_sdof = int(T/h)
    
    t, MSV = MSV_erk2(A=A_sdof, gamma=gamma_sdof, beta=beta_sdof, \
                      Z0=Z0_sdof, dt=h, DW=dW, N=N_sdof, M=M_sdof)
        
    cm = 1/2.54
    fig = pl.figure(figsize=(35*cm, 30*cm))
    fig.tight_layout(pad=5.0*cm)
    
    axs1 = pl.subplot(131)
    pl.plot(t, MSV[:,0], label='ERK2')
    pl.plot(t, np.ones(N_sdof)*0.031,'r--', label='Exato')
    pl.xlabel('tempo')
    pl.ylabel('E[X(t)²]')
    axs1.tick_params(axis='both', which='major', labelsize=8)
    axs1.legend()
    
    axs2 = pl.subplot(132)
    pl.plot(t, MSV[:,1], label='ERK2')
    pl.plot(t, np.ones(N_sdof)*0.031,'r--', label='Exato')
    pl.xlabel('tempo')
    pl.ylabel('E[Y(t)²]')
    axs2.tick_params(axis='both', which='major', labelsize=8)
    axs2.legend()
    
    axs3 = pl.subplot(133)
    pl.plot(t, MSV[:,2], label='ERK2')
    pl.plot(t, np.ones(N_sdof)*0.031,'r--', label='Exato')
    pl.xlabel('tempo')
    pl.ylabel('E[V(t)²]')
    axs3.tick_params(axis='both', which='major', labelsize=8)
    axs3.legend()
    
    pl.show()