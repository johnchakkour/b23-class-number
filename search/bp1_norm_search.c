/* ============================================================================
 * bp1_norm_search.c
 *
 * Exhaustive search for algebraic integers with small coefficients
 * (in {-A,...,A}, A = --coeff, default 1) and small norm in B_{p,1} = the degree-p subfield of Q(zeta_{p^2}),
 * i.e. the first layer of the cyclotomic Z_p-extension of Q.
 *
 * Targets: p = 23 (B_{23,1}, rd = 402.8) and p = 29 (B_{29,1}, rd = 666.7).
 * p = 13 is supported as a VALIDATION case: the program reproduces Miller's
 * two published elements of norm 19 and 23 (see --selftest).
 *
 * ---------------------------------------------------------------- basis ----
 * Let zeta = exp(2 pi i / p^2) and let g be a primitive root mod p^2, so that
 * sigma : zeta -> zeta^g generates Gal(Q(zeta_{p^2})/Q), cyclic of order
 * p(p-1). B_{p,1} is the subfield fixed by <sigma^p> (order p-1). Following
 * Miller, the integral basis is b_0 = 1 together with the Gaussian periods
 *
 *     b_j  =  sum_{k=0}^{p-2} zeta^{ sigma^{kp+j-1} }   ,   1 <= j <= p-1.
 *
 * Writing eta_i = sum_{k=0}^{p-2} zeta^{g^{kp+i} mod p^2} for i = 0..p-1 we
 * have b_j = eta_{j-1}; the orbit of <sigma^p> contains -1, so every eta_i is
 * real, and sum_{i=0}^{p-1} eta_i = mu(p^2) = 0 (which is why only p-1 of the
 * p periods appear in the basis). The p real embeddings are the restrictions
 * of sigma^t, t = 0..p-1, and sigma shifts the period index by one:
 *
 *     tau_t(b_0) = 1,     tau_t(b_j) = eta_{(j-1+t) mod p}.
 *
 * Hence for alpha = sum_j a_j b_j,
 *
 *     N(alpha) = prod_{t=0}^{p-1} (a_0 + sum_{j>=1} a_j eta_{(j-1+t) mod p}),
 *
 * an ordinary rational integer.
 *
 * ------------------------------------------------------------- pipeline ----
 * 1. ENUMERATE every vector in {-A,...,A}^p whose first nonzero coefficient is
 *    positive. Since p is odd, N(-alpha) = -N(alpha), so alpha and -alpha have the
 *    same |N| and fixing the first sign halves the work with no loss. The
 *    enumeration runs over subsets of {0..p-1} of every weight 1..p and, for
 *    each, over all nonzero coefficient patterns; the total is
 *      sum_w C(p,w) A (2A)^{w-1} = ((2A+1)^p - 1)/2,
 *    which the program checks against its own counter and reports as
 *    COVERAGE_OK. For A = 1 this is a complete search of the {-1,0,1} cube:
 *      p = 23:  (3^23-1)/2 = 47,071,589,413
 *      p = 29:  (3^29-1)/2 = 34,315,188,682,441
 *
 * 2. SCREEN in double precision against 4x the norm bound; any candidate with
 *    a nearly-vanishing embedding (|S_t| < 1e-6, where relative error could
 *    matter) is sent to exact verification regardless. Underflow of the
 *    running product can only shrink the estimate, i.e. only pass the screen.
 *    The screen therefore has no false negatives.
 *
 * 3. VERIFY EXACTLY. Norms are recomputed modulo ~62-bit primes q = 1 mod p^2
 *    (where an element h of order exactly p^2 exists: h = u^((q-1)/p^2) with
 *    h^p != 1) and reconstructed by CRT over a modulus that the program sizes
 *    at runtime to exceed twice the a-priori bound prod_t (w*max_i|eta_i| + 1)
 *    on |N|.  The reconstructed integer is exact.
 *
 * 4. CLASSIFY. |N| = 1 -> UNIT. |N| prime -> PRIME (decided by Miller-Rabin
 *    with the fixed bases 2..37, a proven deterministic test below
 *    318665857834031151167461 [Sorenson-Webster 2017]; above that the tag becomes PRP_NEEDS_CERT).
 *    Composite |N| <= --comp-bound is trial-divided by all primes < 2^16 and,
 *    if the remaining cofactor is 1 or a single prime, emitted as FACTORED
 *    with its full factorization. Those lines feed the quotient trick: if
 *    N(x) = P * q1 * q2 * ... with every q_i already known to have a
 *    principal prime above it, then dividing x by the corresponding
 *    conjugates yields an algebraic integer of norm P, so P joins S too.
 *
 * Every line also records whether |N| really is totally split, i.e. whether
 * |N|^(p-1) = 1 (mod p^2); that is printed rather than assumed. The ramified
 * prime p itself must be excluded from S and is tagged split=0 automatically.
 *
 * ---------------------------------------------------------------- build ----
 *   mpicc -O3 -march=native -funroll-loops -fopenmp \
 *         -o bp1_norm_search bp1_norm_search.c -lgmp -lm
 *   gcc -DNOMPI -O3 -march=native -funroll-loops -fopenmp \
 *         -o bp1_norm_search bp1_norm_search.c -lgmp -lm     (no MPI)
 *
 * ---------------------------------------------------------------- usage ----
 *   ./bp1_norm_search --p 23 --bound 1000000000000 --out /scratch/.../b23
 *   ./bp1_norm_search --selftest 13          (reproduces Miller's norms 19,23)
 *
 *   --p N            the prime p (odd; 13, 23, 29 tested)
 *   --weights a-b    restrict to these weights (default 1-p = everything)
 *   --bound X        keep prime norms with |N| <= X       (default 10^12)
 *   --comp-bound X   also emit factorizations of composite |N| <= X
 *                    (default = --bound; use 0 to disable)
 *   --shard k/n      process only slice k of n of the support range
 *   --out PREFIX     output goes to PREFIX_shard%03d_rank%05d.txt
 * ==========================================================================*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <gmp.h>
#include <omp.h>

#ifndef NOMPI
#include <mpi.h>
#endif

#define MAXP       31                 /* p <= 31: tables stay small          */
#define MAXPRIMES  12
#define TINY       1e-6               /* route near-zero embeddings to exact */
#define SPLIMIT    65536              /* trial division bound for factoring  */

static const uint64_t MR_BASES[12] = {2,3,5,7,11,13,17,19,23,29,31,37};
#define MR_DET_LIMIT_STR "318665857834031151167461"   /* psi_12 */

/* ------------------------- modular arithmetic ---------------------------- */
static inline uint64_t addm(uint64_t a, uint64_t b, uint64_t p){
    uint64_t s = a + b; return (s >= p) ? s - p : s;
}
static inline uint64_t mulm(uint64_t a, uint64_t b, uint64_t p){
    return (uint64_t)((unsigned __int128)a * b % p);
}
static uint64_t powm(uint64_t a, uint64_t e, uint64_t p){
    uint64_t r = 1 % p;
    while (e){ if (e & 1) r = mulm(r,a,p); a = mulm(a,a,p); e >>= 1; }
    return r;
}
static int is_prime_u64(uint64_t n){
    if (n < 2) return 0;
    for (int i = 0; i < 12; i++){
        if (n == MR_BASES[i]) return 1;
        if (n % MR_BASES[i] == 0) return 0;
    }
    uint64_t d = n - 1; int r = 0;
    while (!(d & 1)){ d >>= 1; r++; }
    for (int i = 0; i < 12; i++){
        uint64_t x = powm(MR_BASES[i] % n, d, n);
        if (x == 1 || x == n-1) continue;
        int ok = 0;
        for (int j = 1; j < r; j++){ x = mulm(x,x,n); if (x == n-1){ ok = 1; break; } }
        if (!ok) return 0;
    }
    return 1;
}

/* ------------------------------ globals ---------------------------------- */
static int      P;                    /* the prime p; degree = P            */
static int      CA = 1;               /* coefficients range over {-CA..CA}   */
static int      VAL[64];              /* nonzero coefficient values, in order*/
static int      NVAL;                 /* = 2*CA                              */
static int      PP;                   /* p^2                                */
static int      GROOT;                /* primitive root mod p^2             */
static uint64_t BIN[MAXP+1][MAXP+1];
static double   ETA[MAXP];            /* eta_0 .. eta_{p-1}                 */
static double   Ecol[MAXP][MAXP];     /* Ecol[j][t] = tau_t(b_j)            */
static double   ETAMAX;               /* max_i |eta_i|                      */
static uint64_t PR[MAXPRIMES];
static int      NPR;
static uint64_t Epcol[MAXPRIMES][MAXP][MAXP];
static uint32_t SMALLP[7000]; static int NSMALL;   /* primes < 2^16         */

static void build_binomials(void){
    for (int n = 0; n <= MAXP; n++){
        BIN[n][0] = 1;
        for (int k = 1; k <= MAXP; k++)
            BIN[n][k] = (k > n) ? 0 : BIN[n-1][k-1] + BIN[n-1][k];
    }
}
static void unrank_comb(uint64_t r, int w, int *c){
    int x = 0;
    for (int i = 0; i < w; i++){
        while (BIN[P-1-x][w-1-i] <= r){ r -= BIN[P-1-x][w-1-i]; x++; }
        c[i] = x++;
    }
}
static int next_comb(int *c, int w){
    int i = w - 1;
    while (i >= 0 && c[i] == P - w + i) i--;
    if (i < 0) return 0;
    c[i]++;
    for (int j = i+1; j < w; j++) c[j] = c[j-1] + 1;
    return 1;
}
static void small_primes(void){
    static char comp[SPLIMIT];
    NSMALL = 0;
    for (int i = 2; i < SPLIMIT; i++){
        if (!comp[i]){
            SMALLP[NSMALL++] = (uint32_t)i;
            for (long j = (long)i*i; j < SPLIMIT; j += i) comp[j] = 1;
        }
    }
}

/* smallest primitive root mod p^2 (order exactly p(p-1)) */
static int find_primroot(void){
    uint64_t m = PP, ord = (uint64_t)P*(P-1);
    for (uint64_t g = 2; g < 1000; g++){
        if (powm(g, ord/P, m) == 1) continue;           /* order | p-1      */
        int ok = 1;
        for (int q = 2; q <= P-1; q++){                  /* prime factors of p-1 */
            if ((P-1) % q) continue;
            int isp = 1;
            for (int d = 2; d*d <= q; d++) if (q % d == 0){ isp = 0; break; }
            if (!isp) continue;
            if (powm(g, ord/q, m) == 1){ ok = 0; break; }
        }
        if (ok) return (int)g;
    }
    fprintf(stderr, "FATAL: no primitive root found mod %d\n", PP);
    exit(1);
}

static void gen_primes(int need){
    if (need > MAXPRIMES){
        fprintf(stderr, "FATAL: need %d CRT primes > %d\n", need, MAXPRIMES); exit(1);
    }
    uint64_t start = (uint64_t)1 << 62;
    uint64_t q = start + ((PP - (start % PP)) % PP) + 1;   /* q = 1 mod p^2 */
    while (q % PP != 1) q++;
    if (!(q & 1)) q += PP;
    int cnt = 0;
    while (cnt < need){
        if (is_prime_u64(q)) PR[cnt++] = q;
        q += 2ULL*PP;                     /* keeps parity and residue class */
    }
    NPR = need;
}

static void build_tables(void){
    /* real periods, long double accumulation */
    long double eta[MAXP];
    for (int i = 0; i < P; i++){
        long double s = 0.0L;
        uint64_t e = powm(GROOT, i, PP);            /* g^i                  */
        uint64_t step = powm(GROOT, P, PP);         /* g^p                  */
        for (int k = 0; k < P-1; k++){
            s += cosl(2.0L*(long double)M_PI*(long double)e/(long double)PP);
            e = (e * step) % PP;
        }
        eta[i] = s;
    }
    ETAMAX = 0.0;
    for (int i = 0; i < P; i++){
        ETA[i] = (double)eta[i];
        if (fabs(ETA[i]) > ETAMAX) ETAMAX = fabs(ETA[i]);
    }
    for (int t = 0; t < P; t++){
        Ecol[0][t] = 1.0;
        for (int j = 1; j < P; j++) Ecol[j][t] = ETA[(j-1+t) % P];
    }
    for (int ip = 0; ip < NPR; ip++){
        uint64_t q = PR[ip], h = 0;
        for (uint64_t u = 2; ; u++){
            h = powm(u, (q-1)/PP, q);
            if (powm(h, P, q) != 1) break;   /* order exactly p^2, not p     */
        }
        uint64_t em[MAXP];
        uint64_t step = powm(GROOT, P, PP);
        for (int i = 0; i < P; i++){
            uint64_t acc = 0, e = powm(GROOT, i, PP);
            for (int k = 0; k < P-1; k++){
                acc = addm(acc, powm(h, e, q), q);
                e = (e * step) % PP;
            }
            em[i] = acc;
        }
        for (int t = 0; t < P; t++){
            Epcol[ip][0][t] = 1;
            for (int j = 1; j < P; j++) Epcol[ip][j][t] = em[(j-1+t) % P];
        }
    }
}
static int crt_primes_needed(int w){
    double bits = P * log2((double)CA * (double)w * ETAMAX + 1.0) + 4.0;
    int np = (int)(bits / 61.0) + 1;
    return np < 2 ? 2 : np;
}

/* --------------------------- verification -------------------------------- */
typedef struct { mpz_t bound, cbound, mrlim, N, A, M, t1, t2, base, x, d, cof; } ver_t;

static void ver_init(ver_t *V, const mpz_t bound, const mpz_t cbound){
    mpz_init_set(V->bound, bound);
    mpz_init_set(V->cbound, cbound);
    mpz_init_set_str(V->mrlim, MR_DET_LIMIT_STR, 10);
    mpz_inits(V->N, V->A, V->M, V->t1, V->t2, V->base, V->x, V->d, V->cof, NULL);
}
static void ver_clear(ver_t *V){
    mpz_clears(V->bound, V->cbound, V->mrlim, V->N, V->A, V->M, V->t1, V->t2,
               V->base, V->x, V->d, V->cof, NULL);
}
static int mpz_isprime_det(ver_t *V, const mpz_t n){
    if (mpz_cmp_ui(n, 2) < 0) return 0;
    for (int i = 0; i < 12; i++){
        if (mpz_cmp_ui(n, MR_BASES[i]) == 0) return 1;
        if (mpz_divisible_ui_p(n, MR_BASES[i])) return 0;
    }
    mpz_sub_ui(V->d, n, 1);
    unsigned long r = mpz_scan1(V->d, 0);
    mpz_tdiv_q_2exp(V->d, V->d, r);
    mpz_sub_ui(V->t1, n, 1);
    for (int i = 0; i < 12; i++){
        mpz_set_ui(V->base, MR_BASES[i]);
        mpz_powm(V->x, V->base, V->d, n);
        if (mpz_cmp_ui(V->x,1) == 0 || mpz_cmp(V->x, V->t1) == 0) continue;
        int ok = 0;
        for (unsigned long j = 1; j < r; j++){
            mpz_mul(V->x, V->x, V->x); mpz_mod(V->x, V->x, n);
            if (mpz_cmp(V->x, V->t1) == 0){ ok = 1; break; }
        }
        if (!ok) return 0;
    }
    return 1;
}
/* |N|^(p-1) == 1 mod p^2 ?  (totally split test, computed not assumed) */
static int is_split(ver_t *V, const mpz_t a){
    mpz_set_ui(V->t1, (unsigned long)PP);
    unsigned long am = mpz_fdiv_ui(a, (unsigned long)PP);
    if (am % P == 0) return 0;
    return powm(am, (uint64_t)(P-1), (uint64_t)PP) == 1;
}

static void emit(FILE *out, const char *tag, const mpz_t N, int split, int w,
                 const int *c, const int8_t *s, const char *extra)
{
    char *ns = mpz_get_str(NULL, 10, N);
    #pragma omp critical(hitlog)
    {
        fprintf(out, "%s p=%d w=%d N=%s split=%d support=[", tag, P, w, ns, split);
        for (int i = 0; i < w; i++) fprintf(out, "%d%s", c[i], i+1<w?",":"");
        fprintf(out, "] signs=[");
        for (int i = 0; i < w; i++) fprintf(out, "%+d%s", (int)s[i], i+1<w?",":"");
        if (extra && *extra) fprintf(out, "] factors=%s\n", extra);
        else                 fprintf(out, "]\n");
        fflush(out);
    }
    free(ns);
}

static void exact_check(ver_t *V, int w, int np, const int *c, const int8_t *s,
                        FILE *out, uint64_t *nhits)
{
    uint64_t res[MAXPRIMES];
    for (int i = 0; i < np; i++){
        uint64_t q = PR[i], acc = 1;
        for (int t = 0; t < P; t++){
            uint64_t sum = 0;
            for (int k = 0; k < w; k++){
                uint64_t v = mulm(Epcol[i][c[k]][t], (uint64_t)abs(s[k]), q);
                sum = addm(sum, (s[k] > 0) ? v : q - v, q);
            }
            acc = mulm(acc, sum, q);
        }
        res[i] = acc;
    }
    mpz_set_ui(V->N, (unsigned long)res[0]);
    mpz_set_ui(V->M, (unsigned long)PR[0]);
    for (int i = 1; i < np; i++){
        mpz_set_ui(V->t1, (unsigned long)PR[i]);
        mpz_mod(V->t2, V->N, V->t1);
        uint64_t nmod = mpz_get_ui(V->t2);
        uint64_t diff = addm(res[i] % PR[i], PR[i] - nmod % PR[i], PR[i]);
        mpz_mod(V->t2, V->M, V->t1);
        uint64_t minv = powm(mpz_get_ui(V->t2), PR[i]-2, PR[i]);
        mpz_addmul_ui(V->N, V->M, (unsigned long)mulm(diff, minv, PR[i]));
        mpz_mul_ui(V->M, V->M, (unsigned long)PR[i]);
    }
    mpz_tdiv_q_2exp(V->t1, V->M, 1);
    if (mpz_cmp(V->N, V->t1) > 0) mpz_sub(V->N, V->N, V->M);
    mpz_abs(V->A, V->N);

    if (mpz_cmp_ui(V->A, 1) == 0){ emit(out,"UNIT",V->N,0,w,c,s,NULL); (*nhits)++; return; }

    int split = is_split(V, V->A);

    if (mpz_cmp(V->A, V->bound) <= 0){
        if (mpz_cmp(V->A, V->mrlim) < 0){
            if (mpz_isprime_det(V, V->A)){
                emit(out,"PRIME",V->N,split,w,c,s,NULL); (*nhits)++; return;
            }
        } else if (mpz_probab_prime_p(V->A, 64)){
            emit(out,"PRP_NEEDS_CERT",V->N,split,w,c,s,NULL); (*nhits)++; return;
        }
    }
    /* composite: factor for the quotient trick */
    if (mpz_sgn(V->cbound) > 0 && mpz_cmp(V->A, V->cbound) <= 0){
        char fs[512]; int fl = 0; fs[0] = 0;
        mpz_set(V->cof, V->A);
        for (int i = 0; i < NSMALL; i++){
            unsigned long q = SMALLP[i];
            while (mpz_divisible_ui_p(V->cof, q)){
                mpz_divexact_ui(V->cof, V->cof, q);
                fl += snprintf(fs+fl, sizeof(fs)-fl, "%s%lu", fl?"*":"", q);
            }
            if (mpz_cmp_ui(V->cof, 1) == 0) break;
        }
        if (mpz_cmp_ui(V->cof, 1) != 0){
            int pr = (mpz_cmp(V->cof, V->mrlim) < 0) ? mpz_isprime_det(V, V->cof)
                                                     : mpz_probab_prime_p(V->cof, 64);
            if (!pr) return;                 /* cofactor not prime: unusable */
            char *cs = mpz_get_str(NULL, 10, V->cof);
            fl += snprintf(fs+fl, sizeof(fs)-fl, "%s%s", fl?"*":"", cs);
            free(cs);
        }
        emit(out, "FACTORED", V->N, split, w, c, s, fs);
        (*nhits)++;
    }
}

/* ------------------------------ search ----------------------------------- */
static uint64_t search_block(int w, int np, uint64_t lo, uint64_t hi,
                             double screen, ver_t *V, FILE *out, uint64_t *nh)
{
    if (lo >= hi) return 0;
    int c[MAXP]; int8_t s[MAXP]; int idx[MAXP]; double S[MAXP];
    uint64_t done = 0;
    unrank_comb(lo, w, c);
    for (uint64_t ix = lo; ix < hi; ix++){
        /* first nonzero coefficient runs over 1..CA (positive: alpha, -alpha
           have the same |N| since p is odd), the rest over all of
           {-CA..-1,1..CA}.  Odometer with incremental update of S.          */
        for (int a0 = 1; a0 <= CA; a0++){
            s[0] = (int8_t)a0;
            for (int k = 1; k < w; k++){ idx[k] = 0; s[k] = (int8_t)VAL[0]; }
            for (int t = 0; t < P; t++) S[t] = 0.0;
            for (int k = 0; k < w; k++){
                double v = (double)s[k];
                for (int t = 0; t < P; t++) S[t] += v*Ecol[c[k]][t];
            }
            for (;;){
                double prod = 1.0; int tiny = 0;
                for (int t = 0; t < P; t++){
                    double v = S[t];
                    if (fabs(v) < TINY) tiny = 1;
                    prod *= v;
                }
                if (tiny || fabs(prod) < screen) exact_check(V, w, np, c, s, out, nh);
                done++;
                /* advance odometer over positions w-1 .. 1 */
                int k = w - 1;
                while (k >= 1){
                    int old = s[k];
                    idx[k]++;
                    if (idx[k] < NVAL){
                        s[k] = (int8_t)VAL[idx[k]];
                        double d = (double)(s[k] - old);
                        for (int t = 0; t < P; t++) S[t] += d*Ecol[c[k]][t];
                        break;
                    }
                    idx[k] = 0;
                    s[k] = (int8_t)VAL[0];
                    double d = (double)(s[k] - old);
                    for (int t = 0; t < P; t++) S[t] += d*Ecol[c[k]][t];
                    k--;
                }
                if (k < 1) break;
            }
        }
        next_comb(c, w);
    }
    return done;
}

/* ----------------------------- self test --------------------------------- */
static int selftest(void){
    if (P != 13){
        fprintf(stderr, "FATAL: --selftest only applies to p=13 (the reference\n"
                        "       elements of norm 19 and 23 are Miller's for B_{13,1}).\n");
        return 1;
    }
    /* Miller, Class numbers in cyclotomic Z_p-extensions, Prop. 2.3:
       for p = 13, b_0-b_1-b_2-b_3-b_6-b_7-b_8 has norm 19 and
       b_0-b_1-b_2-b_4+b_12 has norm 23.                                    */
    int c1[7] = {0,1,2,3,6,7,8}; int8_t s1[7] = {1,-1,-1,-1,-1,-1,-1};
    int c2[5] = {0,1,2,4,12};   int8_t s2[5] = {1,-1,-1,-1,1};
    struct { int w; int *c; int8_t *s; long want; } T[2] =
        {{7,c1,s1,19},{5,c2,s2,23}};
    int bad = 0;
    mpz_t b, cb; mpz_init_set_ui(b, 1000000); mpz_init_set_ui(cb, 0);
    for (int i = 0; i < 2; i++){
        ver_t V; ver_init(&V, b, cb);
        uint64_t res[MAXPRIMES];
        int np = crt_primes_needed(T[i].w);
        for (int ip = 0; ip < np; ip++){
            uint64_t q = PR[ip], acc = 1;
            for (int t = 0; t < P; t++){
                uint64_t sum = 0;
                for (int k = 0; k < T[i].w; k++){
                    uint64_t v = mulm(Epcol[ip][T[i].c[k]][t], (uint64_t)abs(T[i].s[k]), q);
                    sum = addm(sum, (T[i].s[k] > 0) ? v : q - v, q);
                }
                acc = mulm(acc, sum, q);
            }
            res[ip] = acc;
        }
        mpz_set_ui(V.N, (unsigned long)res[0]); mpz_set_ui(V.M,(unsigned long)PR[0]);
        for (int ip = 1; ip < np; ip++){
            mpz_set_ui(V.t1,(unsigned long)PR[ip]);
            mpz_mod(V.t2, V.N, V.t1);
            uint64_t nmod = mpz_get_ui(V.t2);
            uint64_t diff = addm(res[ip]%PR[ip], PR[ip]-nmod%PR[ip], PR[ip]);
            mpz_mod(V.t2, V.M, V.t1);
            uint64_t minv = powm(mpz_get_ui(V.t2), PR[ip]-2, PR[ip]);
            mpz_addmul_ui(V.N, V.M, (unsigned long)mulm(diff,minv,PR[ip]));
            mpz_mul_ui(V.M, V.M, (unsigned long)PR[ip]);
        }
        mpz_tdiv_q_2exp(V.t1, V.M, 1);
        if (mpz_cmp(V.N, V.t1) > 0) mpz_sub(V.N, V.N, V.M);
        mpz_abs(V.A, V.N);
        long got = mpz_get_si(V.A);
        int split = is_split(&V, V.A);
        printf("selftest p=13: |N| = %ld (expected %ld) split=%d  %s\n",
               got, T[i].want, split, got == T[i].want ? "OK" : "FAIL");
        if (got != T[i].want) bad = 1;
        ver_clear(&V);
    }
    mpz_clears(b, cb, NULL);
    printf("selftest %s\n", bad ? "FAILED" : "PASSED");
    return bad;
}

/* ------------------------------- driver ---------------------------------- */
int main(int argc, char **argv){
    int rank = 0, size = 1;
#ifndef NOMPI
    int prov; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    const char *out_prefix = "bp1_hits";
    const char *bound_str = "1000000000000";        /* 10^12 */
    const char *cbound_str = NULL;
    int wlo = -1, whi = -1, dosel = 0;
    long shard_k = 0, shard_n = 1;
    P = 0;
    for (int i = 1; i < argc; i++){
        if (!strcmp(argv[i],"--p") && i+1 < argc) P = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--selftest") && i+1 < argc){ P = atoi(argv[++i]); dosel = 1; }
        else if (!strcmp(argv[i],"--bound") && i+1 < argc) bound_str = argv[++i];
        else if (!strcmp(argv[i],"--comp-bound") && i+1 < argc) cbound_str = argv[++i];
        else if (!strcmp(argv[i],"--out") && i+1 < argc) out_prefix = argv[++i];
        else if (!strcmp(argv[i],"--coeff") && i+1 < argc) CA = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--weights") && i+1 < argc){
            if (sscanf(argv[++i], "%d-%d", &wlo, &whi) != 2){ wlo = whi = atoi(argv[i]); }
        }
        else if (!strcmp(argv[i],"--shard") && i+1 < argc){
            if (sscanf(argv[++i], "%ld/%ld", &shard_k, &shard_n) != 2 ||
                shard_n < 1 || shard_k < 0 || shard_k >= shard_n){
                if (!rank) fprintf(stderr,"FATAL: --shard must be k/n, 0<=k<n\n");
                return 1;
            }
        }
        else {
            if (!rank) fprintf(stderr,"unknown arg %s\n", argv[i]);
            return 1;
        }
    }
    if (P < 3 || P > MAXP || !(P & 1)){
        if (!rank) fprintf(stderr,"FATAL: --p must be an odd prime <= %d\n", MAXP);
        return 1;
    }
    if (CA < 1 || CA > 9){
        if (!rank) fprintf(stderr,"FATAL: --coeff must be 1..9\n");
        return 1;
    }
    NVAL = 2*CA;
    { int n = 0;
      for (int v = 1; v <= CA; v++){ VAL[n++] = v; VAL[n++] = -v; } }
    PP = P*P;
    if (wlo < 0){ wlo = 1; whi = P; }
    if (wlo < 1 || whi > P || wlo > whi){
        if (!rank) fprintf(stderr,"FATAL: weights outside 1-%d\n", P);
        return 1;
    }

    mpz_t bound, cbound; mpz_init(bound); mpz_init(cbound);
    if (mpz_set_str(bound, bound_str, 10) || mpz_sgn(bound) <= 0){
        if (!rank) fprintf(stderr,"FATAL: bad --bound\n");
        return 1;
    }
    if (cbound_str){
        if (mpz_set_str(cbound, cbound_str, 10)){
            if (!rank) fprintf(stderr,"FATAL: bad --comp-bound\n");
            return 1;
        }
    }
    else mpz_set(cbound, bound);

    build_binomials();
    small_primes();
    GROOT = find_primroot();
    gen_primes(MAXPRIMES);
    build_tables();

    if (dosel){
        int bad = selftest();
#ifndef NOMPI
        MPI_Finalize();
#endif
        return bad;
    }

    double screen = 4.0 * mpz_get_d(bound);
    if (mpz_sgn(cbound) > 0){
        double cs = 4.0 * mpz_get_d(cbound);
        if (cs > screen) screen = cs;
    }

    char fname[4096];
    snprintf(fname, sizeof fname, "%s_shard%03ld_rank%05d.txt", out_prefix, shard_k, rank);
    FILE *out = fopen(fname, "a");
    if (!out){ fprintf(stderr,"FATAL: cannot open %s\n", fname); return 1; }

    if (!rank){
        printf("# B_{%d,1}: degree %d, conductor %d, primitive root %d\n", P, P, PP, GROOT);
        printf("# max|eta_i| = %.4f   CRT primes available %d, first %llu\n",
               ETAMAX, NPR, (unsigned long long)PR[0]);
        printf("# ranks=%d threads/rank=%d weights=%d-%d bound=%s comp-bound=%s shard=%ld/%ld\n",
               size, omp_get_max_threads(), wlo, whi, bound_str,
               cbound_str?cbound_str:bound_str, shard_k, shard_n);
        printf("# coefficients in {-%d..%d}; full box ((2A+1)^%d-1)/2 = ", CA, CA, P);
        { unsigned __int128 tot = 0;
          for (int w = 1; w <= P; w++){
              unsigned __int128 per = CA;
              for (int k = 1; k < w; k++) per *= (unsigned)NVAL;
              tot += (unsigned __int128)BIN[P][w]*per;
          }
          if (tot >> 64) printf("%.6g (exceeds 2^64: use --shard)\n", (double)tot);
          else printf("%llu\n", (unsigned long long)(uint64_t)tot); }
        fflush(stdout);
    }

    for (int w = wlo; w <= whi; w++){
        int np = crt_primes_needed(w);
        if (np > NPR){
            if (!rank) fprintf(stderr,"FATAL: need %d CRT primes, have %d\n", np, NPR);
            return 1;
        }
        uint64_t M  = BIN[P][w];
        uint64_t slo = (uint64_t)((unsigned __int128)M * shard_k / shard_n);
        uint64_t shi = (uint64_t)((unsigned __int128)M * (shard_k+1) / shard_n);
        uint64_t MS = shi - slo;
        double t0 = omp_get_wtime();
        uint64_t lcount = 0, lhits = 0;
        #pragma omp parallel reduction(+:lcount,lhits)
        {
            int T = omp_get_num_threads(), tid = omp_get_thread_num();
            uint64_t G = (uint64_t)size*T, lane = (uint64_t)rank*T + tid;
            uint64_t lo = slo + (uint64_t)((unsigned __int128)MS * lane / G);
            uint64_t hi = slo + (uint64_t)((unsigned __int128)MS * (lane+1) / G);
            ver_t V; ver_init(&V, bound, cbound);
            uint64_t h = 0;
            lcount = search_block(w, np, lo, hi, screen, &V, out, &h);
            lhits = h;
            ver_clear(&V);
        }
        uint64_t gcount = lcount, ghits = lhits;
#ifndef NOMPI
        MPI_Reduce(&lcount,&gcount,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&lhits ,&ghits ,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
#endif
        if (!rank){
            unsigned __int128 per = CA;
            for (int k = 1; k < w; k++) per *= (unsigned)NVAL;
            unsigned __int128 expect128 = (unsigned __int128)MS*per;
            uint64_t expect = (uint64_t)expect128;
            double dt = omp_get_wtime()-t0;
            printf("p=%d w=%2d np=%d candidates=%llu expected=%llu %s hits=%llu time=%.1fs\n",
                   P, w, np, (unsigned long long)gcount, (unsigned long long)expect,
                   (expect128>>64)==0 && gcount==expect ? "COVERAGE_OK" : "COVERAGE_MISMATCH",
                   (unsigned long long)ghits, dt);
            fflush(stdout);
        }
#ifndef NOMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    fclose(out);
    mpz_clears(bound, cbound, NULL);
#ifndef NOMPI
    MPI_Finalize();
#endif
    return 0;
}
