"""Review 2026-09-13 item 1: are the three small-energy eigenvectors on the
DHHKW span primitive IN CLASS?  class(theta_a) = [D_a] exactly (fixed by the
log term), D_a.(-K) = 1 on dP3, so the class of sum_a c_a theta_a is primitive
iff sum_a c_a = 0.  Also print the character of the order-6 rotation on each
eigenvector (theta_a -> theta_{a+1} permutes the coefficients)."""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path[:0] = ['.', 'experiments/dp3', 'experiments/dp3kt']
import dhhkw_theta_sdenergy as m
from dhhkw_theta_compare import GROUP, mu_dhhkw
from scipy.linalg import qr

G = np.asarray(GROUP)
rot = [g for g in G if np.linalg.det(g) > 0]
# order the six rotations as successive powers of one generator
gen = next(g for g in rot if not np.allclose(g, np.eye(2)) and np.allclose(np.linalg.matrix_power(g, 6), np.eye(2))
           and not np.allclose(np.linalg.matrix_power(g, 2), np.eye(2)) and not np.allclose(np.linalg.matrix_power(g, 3), np.eye(2)))
rots = [np.linalg.matrix_power(gen, k) for k in range(6)]
print("six rotations = powers of a generator of order 6:", all(any(np.allclose(r, g) for g in rot) for r in rots))

ch, u, _, _ = m.load_dp3()
smp, wt = m.poly_quad(ch, 48, 48); wt = wt / wt.sum()
gs = jax.vmap(lambda s: m.base_metric(u, s))(smp)
mus = [(lambda s, g=jnp.asarray(g): mu_dhhkw(g @ s)) for g in rots]
As = jnp.stack([jax.vmap(lambda s, f=f: m.A_of(f, s, u))(smp) for f in mus], 1)
oms = jnp.transpose(jax.vmap(lambda AA: jax.vmap(m.wrap)(AA))(jnp.transpose(As, (1, 0, 2, 3))), (1, 0, 2, 3))
st = jax.vmap(lambda O, g: jax.vmap(lambda o: m.hodge(o, g))(O))(oms, gs)
L = jnp.linalg.cholesky(jnp.linalg.inv(gs))
def flat(O):
    Ot = jnp.einsum("nai,nkab,nbj->nkij", L, O, L)
    v = jnp.stack([Ot[:, :, i, j] for (i, j) in m.UT], -1)
    M = np.asarray(jnp.transpose(v, (0, 2, 1)).reshape(-1, v.shape[1]))
    return M * np.repeat(np.sqrt(wt), 6)[:, None]
X, Xs = flat(oms), flat(0.5 * (oms + st))
Q, R, piv = qr(X, mode="economic", pivoting=True)
d = np.abs(np.diag(R)); print("relative diagonal:", np.round(d / d[0], 4))
rank = 6
T = np.zeros((6, rank)); T[piv[:rank], :] = np.linalg.inv(R[:rank, :rank])
Z = Xs @ T
lam, V = np.linalg.eigh(0.5 * (Z.T @ Z + (Z.T @ Z).T))
C = T @ V                                  # eigenvectors in the theta_a basis, columns
shift = np.roll(np.eye(6), 1, axis=0)      # theta_a -> theta_{a+1}
print("\n energy        sum_a c_a / |c|     c.(Sc)/c.c  (a diagonal entry, NOT a character for a 2-dim block)")
for k in range(6):
    c = C[:, k]; c = c / np.linalg.norm(c)
    print(f" {lam[k]:.3e}   {abs(c.sum())/1.0:12.2e}        {float(c @ (shift @ c)):+.4f}")
# characters proper: trace of the generator on each isotypic block (review 2026-09-13 #2)
blocks = []                                   # group by relative energy gap, not by rounding
for k in range(6):
    v = C[:, k] / np.linalg.norm(C[:, k])
    if blocks and abs(lam[k] - blocks[-1][0]) <= 1e-6 * max(abs(lam[k]), 1e-300):
        blocks[-1][1].append(v)
    else:
        blocks.append((lam[k], [v]))
print("\n block energy   dim   character = trace of the order-6 generator on the block")
for key, vecs in blocks:
    Vb = np.stack(vecs, 1); G = Vb.T @ Vb; P = Vb @ np.linalg.solve(G, Vb.T)      # projector on the block
    print(f" {key:.3e}     {len(vecs)}     {np.trace(P @ shift):+.4f}")
print("\n(sum=0 <=> primitive class.  2-dim block: character -1, eigenvalues exp(+-2pi i/3);")
print(" 1-dim blocks: eigenvalue = character, -1 alternating, +1 trivial = Kahler direction)")
