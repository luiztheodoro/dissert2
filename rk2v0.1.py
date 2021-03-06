#!/usr/bin/python
# -*- coding: utf-8 -*- #
"""
        de la Cruz, H. - A simplified weak simulation method for the 
        probabilistic response analysis of nonlinear random vibration 
        problems
        
        Weak order 2.0 Simplified Runge Kutta scheme.
                
        dX = a(X,t)*dt + b(X, t)*dW
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pl

__author__ = "Luiz Antônio Theodoro de Souza"
__version__ = "0.1"

randn = np.random.randn
rand = np.random.random

np.random.seed(1000)


def solve_sde(alfa=None, beta=None, Z0_sde=None, dt=1.0, N=100, t0=0.0, DW=None):
    """
    Sintaxe:
    ----------
    solve_sde(alfa=None, beta=None, X0=None, dt=None, N=100, t0=0, DW=None)
    Parameters:
    ----------
        alfa  : uma função lambda com dois argumentos, o estado Z e o tempo, 
                que define a parte determinística da EDE.
        beta  : uma função lambda com dois argumentos, o estado Z e o tempo, 
                que define a parte estocástica da EDE.
        Z0_sde: Condições iniciais da EDE. 
                (default: gaussian np.random)
        dt    : O incremento de tempo da solução.
                (default: 1)
        N     : O número de incrementos.
                (default: 100)
        t0    : O tempo inicial da solução.
                (default: 0)
        DW    : O Processo Wiener em notação lambda.
                (default: gerador gaussiano,
                    [lambda Y, dt: randn(len(X0)) * np.sqrt(dt)] )
    Exemplo:
    ----------
    == Mathematical model representing the dynamics of piezoeletric vibratory
    energy harvest:
    dX = V
    dY = -bY + V
    dV = -nV - (1-r)X - dX³ - aY + dW
    Zx = lambda Z, t: Z[2] 
    Zy = lambda Z, t: -b*Z[1] + Z[2] 
    Zv = lambda Z, t: -n*Z[2] - (1-r)*Z[0] - d*Z[0]**3 - a*Z[1]
    alfa_sdof = lambda Z, t: np.array([xZ(X,t), yZ(X,t), vZ(X,t)])
    beta_sdof = lambda Z, t: np.array([ 0, 0, -s])
    Z0_sdof = [0.0, 0.0, 0.0]
    t, Y = solve_sde(alfa=alfa, beta=beta, X0=X0, dt=0.01, N=10000)
    """
    if alfa is None or beta is None:
        print("Erro: SDE não foi definida.")
        return
    Z, ti = np.zeros((N, len(Z0_sde))), np.arange(N) * dt + t0
    Z[0, :], h = Z0_sde, dt
    for n in range(N - 1):
        t, tp1 = ti[n], ti[n + 1]
        a = alfa(Z[n, :], t)
        b = beta(Z[n, :], t)
        DWn = DW(h)
        K = Z[n, :] + a * h * 0.5 + b * DWn
        aK = alfa(K, tp1)
        Z[n + 1, :] = Z[n, :] + (a + aK) * h * 0.5 + b * DWn
    return ti, Z


def MSV_rk2(alfa=None, beta=None, Z0=None, dt=1.0, N=100, t0=0.0, DW=None, M=100):
    E = np.zeros((M, N, len(Z0)))
    for m in range(M):
        t, Y = solve_sde(alfa=alfa, beta=beta, Z0_sde=Z0, dt=dt, N=N, t0=t0, DW=DW)
        E[m, :, :] = np.square(Y)
    ret = np.mean(E, axis=0)
    return t, ret


def dist3pts():
    f = (
        lambda x: 0.0
        if (x < 2 / 3)
        else 3.0 ** 0.5
        if (x >= 2 / 3) and (x < 5 / 6)
        else -(3.0 ** 0.5)
    )
    return f(np.random.random())


def sdof_exact(a, b, n, r, s):
    f1 = 1 - r + b ** 2 + b * n
    f2 = b + n
    f3 = n * f1 + a * f2
    E = np.zeros(3)
    E[0] = (s ** 2 / (1 - r)) * (f1 / f3)
    E[1] = (s ** 2 / 2) * (1 / f3)
    E[2] = (s ** 2 / 2) * (f1 / f3)
    return E


if __name__ == "__main__":
    a, b, n, d, r, s = 0.5625, 0.5, 0.02, 0.5, 0.5, 0.5
    Zx = lambda Z, t: Z[2]
    Zy = lambda Z, t: -b * Z[1] + Z[2]
    Zv = lambda Z, t: -n * Z[2] - (1 - r) * Z[0] - d * Z[0] ** 3 - a * Z[1]
    alfa_sdof = lambda Z, t: np.array([Zx(Z, t), Zy(Z, t), Zv(Z, t)])
    beta_sdof = lambda Z, t: np.array([0, 0, -s])
    dW = lambda dt: dist3pts() * np.sqrt(dt)
    h = 0.3
    T = 1000
    Z0_sdof = [0.0, 0.0, 0.0]
    N_sdof = int(T / h)

    t, MSV = MSV_rk2(
        alfa=alfa_sdof, beta=beta_sdof, Z0=Z0_sdof, dt=h, DW=dW, N=N_sdof, M=100
    )

    Ex = sdof_exact(a, b, n, r, s)

    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "text.usetex": True,
            "font.family": "serif",
        }
    )

    cm = 1 / 2.54
    fig, axs = pl.subplots(nrows=1, ncols=3, constrained_layout=True)
    fig.set_size_inches(16 * cm, 8 * cm)
  
    axs[0].plot(t, MSV[:, 0], label="RK2")
    axs[0].plot(t, np.ones(N_sdof) * Ex[0], "r--", label="Exato")
    axs[0].set_xlabel("\\textit{tempo}", fontsize=8)
    axs[0].set_ylabel("$E{X^{(n)}}^2$", fontsize=8)
    axs[0].tick_params(axis="both", which="major", labelsize=8)
    axs[0].set_ylim(-50, 600)
    axs[0].set_xlim(0, 25)
    axs[0].legend(prop={'size': 8})
    axs[0].text(
        0.95,
        0.01,
        "seed: 1000",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=axs[0].transAxes,
        fontsize=8,
    )

    #axs[1] = pl.subplot(132)
    axs[1].plot(t, MSV[:, 1], label="RK2")
    axs[1].plot(t, np.ones(N_sdof) * Ex[1], "r--", label="Exato")
    axs[1].set_xlabel("\\textit{tempo}", fontsize=8)
    axs[1].set_ylabel("$E{Y^{(n)}}^2$", fontsize=8)
    axs[1].tick_params(axis="both", which="major", labelsize=8)
    axs[1].set_ylim(-50, 600)
    axs[1].set_xlim(0, 25)
    axs[1].legend(prop={'size': 8})
    axs[1].text(
        0.95,
        0.01,
        "seed: 1000",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=axs[1].transAxes,
        fontsize=8,
    )

    #axs[2] = pl.subplot(133)
    axs[2].plot(t, MSV[:, 2], label="RK2")
    axs[2].plot(t, np.ones(N_sdof) * Ex[2], "r--", label="Exato")
    axs[2].set_xlabel("\\textit{tempo}", fontsize=8)
    axs[2].set_ylabel("$E{V^{(n)}}^2$", fontsize=8)
    axs[2].tick_params(axis="both", which="major", labelsize=8)
    axs[2].set_ylim(-50, 600)
    axs[2].set_xlim(0, 25)
    axs[2].legend(prop={'size': 8})
    axs[2].text(
        0.95,
        0.01,
        "seed: 1000",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=axs[2].transAxes,
        fontsize=8,
    )
    
    #pl.show()
    pl.savefig("Simulacao04.pgf")
