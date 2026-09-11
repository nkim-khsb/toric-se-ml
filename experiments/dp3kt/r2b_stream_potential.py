"""[R2b-chi] Harmonic (1,1) forms on ANY regular toric transverse KE surface, from
ONE real potential obeying ONE linear equation.  Fully analytic reduction.

SETUP.  4d transverse KE part in action-angle coordinates (x1,x2,phi1,phi2),
u = G_Guillemin + psi,  U = Hess u,  g = U_ij dx_i dx_j + U^ij dphi_i dphi_j,
J = dx_i ^ dphi_i (self-dual), det g = 1.

STEP 1 (closed).  A torus-invariant 2-form has a ds^dphi block A plus P dx1^dx2 and
R dphi1^dphi2.  d omega = 0 forces R = const and A_ij = d_i alpha_j for an
ARBITRARY vector field alpha (mixed partials; A need NOT be symmetric).  Facet
smoothness kills the (P,R) sector: the dvartheta^dphi_perp coefficient must vanish
like l_a, while R is constant and transforms by det N, so R = 0, and ASD locks P to R.

STEP 2 (primitive).  tr A = div alpha = 0, so in 2d alpha = (d_2 chi, -d_1 chi) for
a STREAM FUNCTION chi.  Hence closedness AND primitivity are solved identically by
        A = Hess(chi) eps^T ,    eps = [[0,1],[-1,0]]
   i.e.  A = [[chi_12, -chi_11], [chi_22, -chi_12]] ,  and  Hess chi = A eps.
Decomposition: Phi = chi_12, Psi = (chi_22 - chi_11)/2, tau = -(1/2) Laplacian chi
(tau = the antisymmetric part of A -- a FREE FUNCTION, which is exactly what
r2b_harmonic_forms.py was missing).

STEP 3 (type (1,1)).  For general A the condition is A U = U A^T.  With A = -H eps,
H = Hess chi, it reads H eps U + U eps H = 0.  Use the 2d identities
eps U = adj(U) eps  and  eps M eps^{-1} = (tr M) I - M^T  to get, exactly,
        tr( adj(U) Hess chi ) = 0    <=>    U^{ij} d_i d_j chi = 0 .
ONE real potential, ONE linear second-order equation, valid for any regular toric
transverse KE surface.  NOTE: this is the NON-divergence form.  Since det g = 1 the
metric Laplacian on torus-invariant functions is the DIVERGENCE form
d_i(U^{ij} d_j f); the two differ by (d_i U^{ij}) d_j chi.  chi is NOT harmonic.

BY-PRODUCTS.
  * pointwise norm:  ||omega||^2 = 4 lambda^2 = -4 det Hess(chi)   (det A = det H,
    and for ASD A one has ||omega||^2 = 2 tr(U^{-1} A U A^T) = 2 tr(A^2)).
    So the KT warp source ||omega||^2 is minus the Hessian determinant of chi.
    Indefiniteness of Hess chi is forced -- chi is a saddle potential.
  * conifold, CLOSED FORM:  chi = x1 x2  (then chi_11 = chi_22 = 0 so the equation
    holds for ANY diagonal U).  Verified exactly: unique kernel at 2.3e-16.
  * ellipticity.  With mu = (U11-U22)/2 + i U12, T = tr U, k = 2 mu / T:
    T^2 - 4|mu|^2 = 4 det U, so |k| = sqrt(1 - 4 det U/T^2) < 1 inside; in Beltrami
    form |a| + |b| = |k|/(2-|k|).  At a facet U ~ n_a n_a^T/(2 l_a) is rank one, so
    lambda_min(U^{-1}) -> 2 l_a/|n_a|^2: UNIFORMLY ELLIPTIC in the interior and
    CRITICALLY DEGENERATE exactly on the facets, LINEARLY in the distance.
    Measured (dP3): lambda_min(U^{-1})/l_a = 1.049, 1.616, 1.883, 1.961, 1.988 at
    l_a = 1e-1 ... 1e-3 -> 2 = 2/|n_a|^2.  That degeneracy is why the facet
    boundary condition is not an extra constraint, and why a global polynomial
    basis is delicate there.

RESULT (2026-08-19), SD-energy Rayleigh quotient (0 = exactly ASD), D = degree of chi.
Reported on the DETERMINISTIC quadrature (poly_quad), with the old Monte-Carlo numbers
beside them because the comparison to r2b_harmonic_forms.py was made that way:

  dP3, D=16      MC   2.44e-12  5.11e-12  1.86e-11 | 4.03e-05   gap 2.2e6
                 det  6.16e-11  8.09e-11  3.19e-10 | 1.16e-01   gap 3.7e8
  Y^{3,2}, D=16  MC   1.71e-13 | 3.29e-05                       gap 1.9e8
                 det  8.79e-12 | 1.06e-01                       gap 1.2e10
  conifold       exact to machine precision at every D, both rules.

ATTRIBUTION (added 2026-08-19).  The curl-sector diagnosis is NOT ours: commit 0bb512d
(r2b_facet_forms.py, note-dp3kt.md section R2b+, 2026-07-23) found it a month earlier,
with two potentials chi_1, chi_2, and already concluded that the facet ell^{5/2}
prescription is not the fix.  What is ours is the SINGLE-potential form: primitivity is
built in identically (tr A = 0 by construction) rather than left to the solve, which
halves the parameter count and, with degree, goes several decades lower.  Like-for-like:

  Same deterministic quadrature, dP3, lowest three self-dual energies:
    R2b old  (1 potential + const antisym,  74 gens)  1.18e-2 1.17e-1 2.00e-1  no kernel
    R2b+ two potentials only              (117 gens)  3.55e-6 1.60e-5 1.60e-5  gap 9710x
    R2b+ full, + facet ell^5/2 + corners  (165 gens)  3.49e-6 1.30e-5 1.30e-5  gap 7265x
    stream potential, D=8                  (42 gens)  1.61e-5 1.61e-5 8.08e-5  gap 2206x
    stream potential, D=16                (150 gens)  5.78e-11 7.62e-11 3.00e-10 gap 3.1e8x

Three readings.  (i) The two potentials are the whole of the plateau fix (1.2e-2 ->
3.6e-6), which is 0bb512d's result.  (ii) The facet ell^{5/2} and corner terms buy almost
nothing (3.55e-6 -> 3.49e-6 for 48 more generators), strengthening that note's own
conclusion.  (iii) The single potential reaches the same place with 42 generators instead
of 117, and then keeps going: 5.8e-11 at D = 16.  So note-dp3kt.md's judgement that the
~1e-5 plateau was "a real function-space limit, not MC noise" is REFUTED -- it was the
extra non-traceless directions plus Monte-Carlo quadrature.  The honest headline is
~4.8 decades below R2b+ (3.5e-6 -> 5.8e-11), not eight below the original R2b.

Read this carefully.  The DETERMINISTIC numbers are the honest L^2 Rayleigh quotients.
Monte-Carlo sampling used an eps = 1.5e-2 interior margin, which EXCLUDES the boundary
strip -- precisely where a global polynomial ansatz is worst -- so it under-reported
everything: the non-kernel eigenvalues by ~3000x (4.0e-5 vs 1.2e-1) and the kernel by
~25x.  What survives unchanged, and is the point:
  * exactly b2^- eigenvalues collapse, with the GAP improving 170x (2.2e6 -> 3.7e8 on
    dP3), so the topological identification is sharper, not weaker;
  * tr A = 0 identically (0.0e+00) under both rules, primitivity being built in;
  * the "~8 decades against r2b_harmonic_forms.py's 4.09e-4" statement is WITHDRAWN:
    that baseline was already superseded inside the project by R2b+ (0bb512d) before
    this file existed.  The correct baseline is R2b+'s 3.5e-6, so the improvement to
    quote is ~4.8 decades, and it comes from building primitivity in, raising the
    degree, and deterministic quadrature -- not from the curl fix, which is theirs.
    The absolute figure to quote is the deterministic 5.8e-11, not 2.4e-12.

CAVEAT: for dP3 the printed R is meaningless -- with a 3-dimensional kernel eigh
returns an arbitrary member.  Fixing a basis needs a selection rule (prescribed
divisor periods, [P1] r2b_periods.py, which are vertex differences of the potential).

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_stream_potential.py
"""
import sys, numpy as np, jax, jax.numpy as jnp
from scipy.linalg import qr
jax.config.update("jax_enable_x64",True); sys.path.insert(0,"."); sys.path.insert(0,"experiments/dp3kt")
from sugrasol.ypq import sample_slice
from sugrasol.forms import hodge
from r2b_ypq_calibration import load_case, base_metric, S1,P1,S2,P2, affine_probe
from r2b_holomorphic import load_dp3
UT=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
EPSM=jnp.array([[0.,1.],[-1.,0.]])

def poly_quad(ch, nu=48, nv=48):
    """Thin wrapper on sugrasol.laplacian.polygon_quadrature (centroid fan + Duffy +
    tensor Gauss-Legendre).  THAT FUNCTION ALREADY EXISTED and its docstring already
    explained why it beats Monte-Carlo for exactly these integrands -- an earlier
    version of this file reinvented it.  Kept as a wrapper only so the call signature
    here is unchanged; nu is the GL order per direction and nv is ignored."""
    from sugrasol.laplacian import polygon_quadrature
    pts, wts = polygon_quadrature(np.asarray(ch.verts_s), ngl=nu)
    return jnp.asarray(np.asarray(pts)), np.asarray(wts)


def gen_forms(smp, D):
    pw=[(i,t-i) for t in range(2,D+1) for i in range(t+1)]
    def Aof(s,a,b):
        H=jax.hessian(lambda z: z[0]**a*z[1]**b)(s)
        return H @ EPSM.T                       # A = Hess(chi) eps^T
    def wrap(A):
        w=jnp.zeros((4,4))
        for a,ia in enumerate((S1,S2)):
            for b,ib in enumerate((P1,P2)):
                w=w.at[ia,ib].add(A[a,b]).at[ib,ia].add(-A[a,b])
        return w
    oms=jnp.stack([jax.vmap(lambda s,a=a,b=b: wrap(Aof(s,a,b)))(smp) for (a,b) in pw],1)
    As =jnp.stack([jax.vmap(lambda s,a=a,b=b: Aof(s,a,b))(smp) for (a,b) in pw],1)
    return oms, As, len(pw)

def run(ch,u,D,b2,n=4000,seed=5,eps=1.5e-2,tol=1e-12,nq=None):
    """nq = (nu, nv) uses the deterministic rule; nq=None keeps the old MC sampling
    so the two can be compared in one run."""
    if nq is None:
        smp=jnp.asarray(np.asarray(sample_slice(jax.random.PRNGKey(seed),ch,n,eps=eps)))
        wt=np.full(len(smp), 1.0/len(smp))
    else:
        smp, wt = poly_quad(ch, *nq)
        wt = wt/wt.sum()
    gs=jax.vmap(lambda s: base_metric(u,s))(smp)
    oms,As,K=gen_forms(smp,D)
    st=jax.vmap(lambda O,g: jax.vmap(lambda o: hodge(o,g))(O))(oms,gs)
    L=jnp.linalg.cholesky(jnp.linalg.inv(gs))
    def flat(O):
        Ot=jnp.einsum("nai,nkab,nbj->nkij",L,O,L)
        v=jnp.stack([Ot[:,:,i,j] for (i,j) in UT],-1)
        M=np.asarray(jnp.transpose(v,(0,2,1)).reshape(-1,v.shape[1]))
        return M*np.repeat(np.sqrt(wt),6)[:,None]
    X,Xs=flat(oms),flat(0.5*(oms+st))
    Q,R,piv=qr(X,mode="economic",pivoting=True)
    d=np.abs(np.diag(R)); r=int(np.sum(d>tol*d[0]))
    T=np.zeros((K,r)); T[piv[:r],:]=np.linalg.inv(R[:r,:r])
    Z=Xs@T; M=.5*(Z.T@Z+(Z.T@Z).T)
    lam,C=np.linalg.eigh(M)
    # primitivity, exact by construction
    trA=float(jnp.max(jnp.abs(jnp.trace(As,axis1=2,axis2=3))))
    # lambda from -det Hess chi  = -det A
    A0=jnp.einsum("nkab,k->nab",As,jnp.asarray(T@C[:,0]))
    lamv=np.sqrt(np.maximum(-np.asarray(jnp.linalg.det(A0)),0.0))
    rr,_,Rv=affine_probe(smp,lamv,np.asarray(ch.verts_s))
    return lam,K,r,trA,rr,Rv

if __name__ == "__main__":
  for tag,(ch,u,y1,y2),b2 in [("conifold",load_case("conifold"),1),
                              ("Y^{3,2}",load_case((3,2)),1),
                              ("dP3",load_dp3(),3)]:
      Rh=None if y1 is None else ((1-y1)/(1-y2))**2
      print(f"\n=== {tag}  b2^- = {b2}" + (f"   HEK R = {Rh:.6f}" if Rh else "") + " ===")
      print(f"  {'quad':>6} {'D':>3} {'basis':>6} {'rank':>5} {'max|trA|':>9} | "
            f"lowest 5 SD eigenvalues        | gap")
      for nq in (None,(48,48)):
          for D in (8,12,16):
              lam,K,r,trA,rr,Rv=run(ch,u,D,b2,nq=nq)
              gp=lam[b2]/lam[b2-1] if lam[b2-1]>0 else float("inf")
              print(f"  {'MC' if nq is None else 'det':>6} {D:>3} {K:>6} {r:>5} {trA:>9.1e} | "
                    +"  ".join(f"{v:8.2e}" for v in lam[:5])+f" | {gp:8.0f}x")
