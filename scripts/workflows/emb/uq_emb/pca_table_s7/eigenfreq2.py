import numpy as np
from dataclasses import dataclass
from typing import List

import yaml
import trimesh
from scipy.optimize import root, brentq
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
#import addcopyfighandler
from typing import Optional

# =========================
# Special functions
# =========================

def sph_hankel1(n: int, z: complex) -> complex:
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def sph_hankel1_prime(n: int, z: complex) -> complex:
    return spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)


# =========================
# Parameters
# =========================

@dataclass
class ShellFluidParams:
    nu: float
    beta: float = 0.0
    f_i: float = 0.0
    eta_i: float = 1.0
    f_o: float = 0.0
    eta_o: float = 1.0


# =========================
# Acoustic loading factor
# =========================

def acoustic_loading_chi(w: complex, n: int, p: ShellFluidParams) -> complex:
    """
    Acoustic loading factor.

    chi_n(w) = f_i * j_n(q_i)/(q_i j_n'(q_i))
             - f_o * h_n^(1)(q_o)/(q_o h_n^(1)'(q_o))

    with q_i = w / eta_i, q_o = w / eta_o.
    """
    chi = 0.0 + 0.0j

    if p.f_i != 0.0:
        q_i = w / p.eta_i
        jn = spherical_jn(n, q_i)
        jn_p = spherical_jn(n, q_i, derivative=True)
        chi += p.f_i * (jn / (q_i * jn_p))

    if p.f_o != 0.0:
        q_o = w / p.eta_o
        hn = sph_hankel1(n, q_o)
        hn_p = sph_hankel1_prime(n, q_o)
        chi -= p.f_o * (hn / (q_o * hn_p))

    return chi


# =========================
# Dispersion relation
# =========================

def dispersion(w: complex, n: int, p: ShellFluidParams) -> complex:
    r"""
    Full dimensionless bent-shell dispersion function F(w) = 0.
    """
    nu = p.nu
    beta = p.beta
    chi = acoustic_loading_chi(w, n, p)

    if n == 0:
        return (1.0 + chi) * w * w - 2.0 * (1.0 + nu)

    lam = n * (n + 1.0)

    A = ((1.0 + chi) * (1.0 - nu - lam) * (1.0 + beta)
         - 2.0 * (1.0 + nu)
         - beta * (lam * lam - 2.0 * lam * (1.0 - nu)))

    B = (-(1.0 + nu) * (
            2.0 * (1.0 - nu - lam) * (1.0 + beta)
            + lam * (1.0 + nu - beta * (1.0 - nu - lam))
         )
         - beta * (2.0 - lam) * (lam * lam - lam * (1.0 - nu)))

    return (1.0 + chi) * w**4 + A * w**2 + B


# =========================
# Empty-shell seed guesses
# =========================

def empty_shell_seeds(n: int, nu: float, beta: float = 0.0) -> List[float]:
    if n == 0:
        return [np.sqrt(2.0 * (1.0 + nu))]

    lam = n * (n + 1.0)

    A = ((1.0 - nu - lam) * (1.0 + beta)
         - 2.0 * (1.0 + nu)
         - beta * (lam * lam - 2.0 * lam * (1.0 - nu)))

    B = (-(1.0 + nu) * (
            2.0 * (1.0 - nu - lam) * (1.0 + beta)
            + lam * (1.0 + nu - beta * (1.0 - nu - lam))
         )
         - beta * (2.0 - lam) * (lam * lam - lam * (1.0 - nu)))

    disc = A * A - 4.0 * B
    if disc < 0.0:
        return []

    z1 = 0.5 * (-A + np.sqrt(disc))
    z2 = 0.5 * (-A - np.sqrt(disc))

    seeds = []
    for z in (z1, z2):
        if z > 1e-12:
            seeds.append(np.sqrt(z))
    return sorted(seeds)


# =========================
# Interior acoustic seeds
# =========================

def interior_derivative_zeros(n: int, q_max: float, num_scan: int = 4000) -> List[float]:
    """Find real zeros of j_n'(q) on (0, q_max)."""
    if q_max <= 0:
        return []

    xs = np.linspace(1e-8, q_max, num_scan)
    vals = spherical_jn(n, xs, derivative=True)

    roots = []
    for k in range(len(xs) - 1):
        a, b = xs[k], xs[k + 1]
        fa, fb = vals[k], vals[k + 1]

        if not np.isfinite(fa) or not np.isfinite(fb):
            continue

        if fa == 0.0:
            roots.append(a)
            continue

        if fa * fb < 0.0:
            try:
                r = brentq(lambda x: spherical_jn(n, x, derivative=True), a, b)
                roots.append(r)
            except ValueError:
                pass

    roots = np.array(sorted(roots))
    if len(roots) == 0:
        return []

    dedup = [roots[0]]
    for r in roots[1:]:
        if abs(r - dedup[-1]) > 1e-5:
            dedup.append(r)
    return dedup


# =========================
# Complex root refinement
# =========================

def refine_complex_root(z0: complex, n: int, p: ShellFluidParams,
                        tol: float = 1e-10, maxfev: int = 500) -> Optional[complex]:
    """Refine a complex root by solving Re(F)=0 and Im(F)=0."""

    def fun(xy):
        z = xy[0] + 1j * xy[1]
        F = dispersion(z, n, p)
        return np.array([F.real, F.imag], dtype=float)

    sol = root(fun, np.array([z0.real, z0.imag], dtype=float),
               tol=tol, options={"maxfev": maxfev})

    if not sol.success:
        return None

    z = sol.x[0] + 1j * sol.x[1]
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        return None

    Fz = dispersion(z, n, p)
    if abs(Fz) > 1e-7:
        return None

    return z


def deduplicate_roots(roots: List[complex], atol: float = 1e-6) -> List[complex]:
    out = []
    for z in roots:
        if not any(abs(z - w) < atol for w in out):
            out.append(z)
    return out


# =========================
# Main solver
# =========================

def solve_roots_for_n(
    n: int,
    w_max: float,
    params: ShellFluidParams,
    imag_seed: float = 1e-2,
    nscan_real: int = 3000,
    extra_scan_seeds: bool = True,
    keep_upper_half: bool = False,
) -> List[complex]:
    """Compute roots up to approximately Re(w) <= w_max for a fixed n."""
    seeds = []

    for s in empty_shell_seeds(n, params.nu, params.beta):
        if s <= w_max * 1.2:
            seeds.append(s - 1j * imag_seed if params.f_o != 0.0 else s + 0j)

    if params.f_i != 0.0:
        q_max = w_max / params.eta_i
        for q in interior_derivative_zeros(n, q_max):
            s = params.eta_i * q
            if s <= w_max * 1.2:
                seeds.append(s - 1j * imag_seed if params.f_o != 0.0 else s + 0j)

    if extra_scan_seeds:
        xs = np.linspace(1e-6, w_max, nscan_real)
        ys = np.array([abs(dispersion(x - 1j * imag_seed, n, params)) for x in xs])

        for k in range(1, len(xs) - 1):
            if ys[k] < ys[k - 1] and ys[k] < ys[k + 1]:
                seeds.append(xs[k] - 1j * imag_seed)

    roots = []
    for z0 in seeds:
        z = refine_complex_root(z0, n, params)
        if z is None:
            continue
        if z.real < -1e-8 or z.real > w_max + 1e-6:
            continue
        if not keep_upper_half and z.imag > 1e-8:
            continue
        roots.append(z)

    return sorted(deduplicate_roots(roots), key=lambda z: (z.real, z.imag))


def solve_all_roots(
    n_max: int,
    w_max: float,
    params: ShellFluidParams,
    imag_seed: float = 1e-2,
    nscan_real: int = 3000,
    extra_scan_seeds: bool = True,
    keep_upper_half: bool = False,
):
    all_roots = []
    for n in range(n_max + 1):
        all_roots.append(
            solve_roots_for_n(
                n,
                w_max,
                params,
                imag_seed=imag_seed,
                nscan_real=nscan_real,
                extra_scan_seeds=extra_scan_seeds,
                keep_upper_half=keep_upper_half,
            )
        )
    return range(n_max + 1), all_roots


# =========================
# Optional dimensionalization
# =========================

def shell_wave_speed(E_surface: float, rho_s: float, nu: float) -> float:
    """c_s = sqrt(E / (rho_s * (1 - nu^2)))"""
    return np.sqrt(E_surface / (rho_s * (1.0 - nu**2)))


def dimensional_frequency(w: complex, a: float, E_surface: float, rho_s: float, nu: float) -> complex:
    """omega_dimensional = (c_s / a) * w [rad/s]"""
    c_s = shell_wave_speed(E_surface, rho_s, nu)
    return (c_s / a) * w


def beta_from_material(kappa: float, E_surface: float, a: float, nu: float) -> float:
    """beta = kappa * (1 - nu^2) / (E_surface * a^2)"""
    return kappa * (1.0 - nu**2) / (E_surface * a * a)


def get_torsional_frequency(n: int, beta: float, nu: float) -> float:
    return np.sqrt((1 - nu) / 2 * (n * (n + 1) - 2))


def load_workspace_parameters(simnum: str = "00001"):
    with open(f"parameter/parameters-default{simnum}.yaml", "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)

    with open(f"parameter/parameters{simnum}.yaml", "rb") as f:
        parameters = yaml.load(f, Loader=yaml.CLoader)

    mesh_path = f"mesh/{parameters_default['objFile'].replace('.off', '')}{simnum}.off"
    mesh = trimesh.load(mesh_path)

    nu = parameters_default["nu"]
    ka = parameters["ka"]
    kb = parameters["kb"]
    kappa = kb * np.sqrt(3.0) / 2.0

    mass_factor = parameters["mass_factor"]
    rho_s = (parameters["mvert"] / mass_factor) * len(mesh.vertices) / mesh.area
    a = np.sqrt(mesh.area / (4.0 * np.pi))
    beta = beta_from_material(kappa, ka, a, nu)
    c_s = shell_wave_speed(ka, rho_s, nu)

    return {
        "nu": nu,
        "ka": ka,
        "kb": kb,
        "mass_factor": mass_factor,
        "beta": beta,
        "rho_s": rho_s,
        "a": a,
        "c_s": c_s,
        "mesh_path": mesh_path,
        "mesh": mesh,
        "parameters_default": parameters_default,
        "parameters": parameters,
    }


if __name__ == "__main__":
    workspace = load_workspace_parameters("00001")

    nu = workspace["nu"]
    beta = workspace["beta"]
    rho_s = workspace["rho_s"]
    a = workspace["a"]
    c_s = workspace["c_s"]

    eta_i = 1.0
    eta_o = 1.0
    f_i = 0.0
    f_o = 0.0

    nmodes = 70
    w_max = 100

    print(f"mesh = {workspace['mesh_path']}")
    print(f"a = {a:.3f}")
    print(f"rho_s = {rho_s:.6f}")
    print(f"beta = {beta:.3e}")
    print(f"c_s = {c_s:.3e}")

    p2 = ShellFluidParams(
        nu=nu,
        beta=beta,
        f_i=f_i,
        eta_i=eta_i,
        f_o=f_o,
        eta_o=eta_o,
    )

    n_all, roots_all = solve_all_roots(n_max=16, w_max=w_max, params=p2)
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ROOTS = []
    min_real_w = 1e-3
    for i, n in enumerate(n_all):
        roots_n = [z for z in roots_all[i] if z.real >= min_real_w]
        print(f"n={n}: {roots_n}")

        ax1.plot([n] * len(roots_n), np.real(roots_n), 'o', color="black", label="spheroidal" if n == 2 else None)
        if n > 1:
            ax1.plot([n], [get_torsional_frequency(n, beta, nu)], '^', color="C0", label="torsional" if n == 2 else None)
        ROOTS.extend(roots_n * (2 * n + 1))
        if n > 1:
            ROOTS.extend([get_torsional_frequency(n, beta, nu)] * (2 * n + 1))
    ax1.legend()
    ax1.set_xlabel("n")
    ax1.set_ylabel("Re(w)")

    theory_frequencies = np.sort(np.real(np.array(ROOTS)))[:nmodes] * c_s / a
    ax2.plot(theory_frequencies, 'o', color="black", label="Theory", zorder=10)
    ax2.set_xlabel("ordered mode index")
    ax2.set_ylabel("Re(w)")

    ax1.set_title("Dimensionless frequencies  (no degeneracy)")
    ax2.set_title("Dimensional frequencies (ordered)")

    #frequencies = np.loadtxt("eigvalues_NIKOS.txt")
    #frequencies = np.loadtxt("eigvalues_new.txt")    
    #frequencies = np.sqrt(1 / frequencies[::-1])
    #frequencies = np.sort(frequencies)
    #frequencies = frequencies[:nmodes]
    
    lambdas = np.loadtxt("eigvalues_new.txt")
    lambdas = np.sort(lambdas)[::-1]

    kbt = workspace["parameters"]["kbt"]
    mass_factor = workspace["mass_factor"]
    ut = workspace["parameters"]["ut"]

    omega_tau_sim = np.sqrt(kbt / lambdas)
    frequencies = omega_tau_sim * np.sqrt(mass_factor)
    frequencies = frequencies[:nmodes]
    # ka, kb, and the measured covariance are already expressed in the
    # current DPD unit system.  fscale entered the parameter construction;
    # applying it again here would overestimate Hz by 1/sqrt(fscale).
    frequency_hz = frequencies / (2.0 * np.pi * ut)

    frequency_table = np.column_stack((
        np.arange(len(frequencies), dtype=int),
        lambdas[:len(frequencies)],
        omega_tau_sim[:len(frequencies)],
        frequencies,
        frequency_hz,
    ))
    np.savetxt(
        "pca_frequencies.txt",
        frequency_table,
        header=(
            "mode eigenvalue omega_dpd_sim "
            "omega_dpd_mass_corrected frequency_hz"
        ),
        fmt=["%d", "%.12e", "%.12e", "%.12e", "%.12e"],
    )

    print(f"mass_factor = {mass_factor}")
    print(f"Hz per mass-corrected DPD angular frequency = "
          f"{1.0 / (2.0 * np.pi * ut):.12e}")
    print("saved pca_frequencies.txt")

    ax2.plot(frequencies, 'ob-', label="PCA (mass corrected)")
    ax2.set_ylabel("Angular frequency, mass-corrected DPD units")
    ax2.legend()
    #ax2.set_xlim(25, 50)
    #ax2.set_ylim(120,140)
    plt.savefig('modes_pca.png')
