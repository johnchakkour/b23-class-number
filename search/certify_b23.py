#!/usr/bin/env python3
"""
certify_b23.py -- build the certificate for h(B_{p,1}) = 1 from the output of
bp1_norm_search.

    python3 certify_b23.py PREFIX --out b23.certificate.txt [--p 23] [--cutoff 50000000]
    python3 certify_b23.py --selftest

PREFIX is the --out prefix given to bp1_norm_search; the hits are read from
PREFIX_shard*_rank*.txt.  Only the Python standard library is needed.

1. Every line is parsed strictly: it must be complete and well formed, its
   listed factors must be prime and multiply back to |N|.
2. A totally split prime q is certified directly by an element of norm +-q
   (kind P), or of norm q times powers of p and of inert primes (kind Q).
3. The quotient trick is applied in rounds: an element whose norm contains q
   exactly once, and otherwise only p, inert primes and split primes already
   certified, certifies q (kind Q, with those split primes as dependencies).
4. The certificate keeps every certified prime q <= cutoff, plus the primes
   their rows depend on.

The output depends only on the SET of lines in the hit files, not on their
order or on how the search was divided among processes: whenever several
elements can certify the same prime in the same round, the one with the fewest
nonzero coefficients is taken, ties broken by a fixed order on coefficient
vectors.  The certificate's norms are not checked here; that is the job of
verify/verify_certificate.py, which recomputes all of them exactly.
"""
import glob, hashlib, math, os, re, sys, tempfile

# ------------------------------------------------------------------ primality
MRB = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
MR_LIMIT = 318665857834031151167461     # MR with the bases MRB is proven below this

def isprime(n, _cache={}):
    v = _cache.get(n)
    if v is not None: return v
    if n < 2: return False
    for b in MRB:
        if n % b == 0: return n == b
    if n >= MR_LIMIT: raise ValueError("too large for the deterministic test: %d" % n)
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    v = True
    for b in MRB:
        x = pow(b, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            v = False; break
    _cache[n] = v
    return v

def primroot(p):
    """smallest primitive root mod p^2 -- the rule used by bp1_norm_search.c"""
    m, order = p * p, p * (p - 1)
    qs = [q for q in range(2, p + 1) if order % q == 0 and isprime(q)]
    return next(g for g in range(2, m) if math.gcd(g, p) == 1
                and all(pow(g, order // q, m) != 1 for q in qs))

# ------------------------------------------------------------------- parsing
LINE = re.compile(
    r"^(UNIT|PRIME|PRP_NEEDS_CERT|FACTORED) p=(\d+) w=(\d+) N=(-?\d+) split=[01] "
    r"support=\[([0-9,]*)\] signs=\[([-+0-9,]*)\](?: factors=([0-9]+(?:\*[0-9]+)*))?$")

def parse_vec(sup, sgn, w, p):
    try:
        s = [int(x) for x in sup.split(",")] if sup else []
        a = [int(x) for x in sgn.split(",")] if sgn else []
    except ValueError:
        return None
    if not (len(s) == len(a) == w and 1 <= w <= p): return None
    if any(not 0 <= x < p for x in s) or any(s[i] >= s[i + 1] for i in range(w - 1)): return None
    if any(x == 0 or abs(x) > 9 for x in a) or a[0] <= 0: return None
    return s, a

def canonical_key(s, a, p):
    """fewest nonzero coefficients first, then a fixed order on the vectors"""
    code = 0
    for i, x in zip(s, a): code += (2 * x - 1 if x > 0 else -2 * x) * 19 ** i
    return len(s) * 19 ** p + code

# ---------------------------------------------------------------------- main
def certify(prefix, out, p=23, cutoff=50_000_000, log=print):
    m = p * p
    split = lambda q: q % p != 0 and pow(q, p - 1, m) == 1
    files = sorted(glob.glob(prefix + "_shard*_rank*.txt"))
    if not files: sys.exit("no files match %s_shard*_rank*.txt" % prefix)
    log("B_{%d,1}: reading %d hit files" % (p, len(files)))

    rejects, nlines = {}, 0
    def rej(why): rejects[why] = rejects.get(why, 0) + 1
    direct = {}         # q -> (rank, key, file, offset); rank 0 = kind P, 1 = kind Q
    pending = []        # (split factors with exponents, key, file, offset)
    for fid, fn in enumerate(files):
        off = 0
        with open(fn, "rb") as fh:
            for raw in fh:
                here = off; off += len(raw); nlines += 1
                if not raw.endswith(b"\n"): rej("incomplete line"); continue
                mt = LINE.match(raw[:-1].decode("ascii", "replace"))
                if not mt: rej("malformed line"); continue
                tag, pp, w, N, sup, sgn, fac = mt.groups()
                if int(pp) != p: rej("wrong p"); continue
                v = parse_vec(sup, sgn, int(w), p)
                if v is None: rej("malformed coefficient vector"); continue
                if tag == "UNIT": continue
                if tag == "PRP_NEEDS_CERT": rej("norm beyond the proven primality range"); continue
                A, key = abs(int(N)), canonical_key(*v, p)
                if tag == "PRIME":
                    if fac is not None or not isprime(A): rej("PRIME line whose norm is not prime"); continue
                    if split(A):
                        cand = (0, key, fid, here)
                        if A not in direct or cand < direct[A]: direct[A] = cand
                    continue
                if fac is None: rej("FACTORED line without factors"); continue
                fs = [int(x) for x in fac.split("*")]
                if math.prod(fs) != A: rej("factors do not multiply to |N|"); continue
                if not all(isprime(x) for x in fs): rej("a listed factor is not prime"); continue
                ex = {}
                for x in fs:
                    if split(x): ex[x] = ex.get(x, 0) + 1
                if 1 not in ex.values(): continue           # cannot certify anything
                if len(ex) == 1:
                    q = next(iter(ex))
                    cand = (1, key, fid, here)
                    if q not in direct or cand < direct[q]: direct[q] = cand
                else:
                    pending.append((tuple(sorted(ex.items())), key, fid, here))
    log("lines read: %d" % nlines)
    for why, k in sorted(rejects.items()): log("   rejected: %9d  %s" % (k, why))
    known = {q: ("PQ"[r], (), fid, off) for q, (r, key, fid, off) in direct.items()}
    log("primes certified directly: %d" % len(known))

    rnd = 0
    while pending:
        rnd += 1
        best, rest = {}, []
        for ex, key, fid, off in pending:
            unknown = [(q, e) for q, e in ex if q not in known]
            if not unknown: continue
            if len(unknown) == 1 and unknown[0][1] == 1:
                q = unknown[0][0]
                if q not in best or key < best[q][0]:
                    best[q] = (key, tuple(r for r, _ in ex if r != q), fid, off)
            else:
                rest.append((ex, key, fid, off))
        if not best: break
        for q, (key, deps, fid, off) in best.items(): known[q] = ("Q", deps, fid, off)
        pending = rest
        log("quotient trick, round %d: +%d primes (%d in all)" % (rnd, len(best), len(known)))
    del pending

    chosen = sorted(q for q in known if q <= cutoff)
    need, stack = set(chosen), list(chosen)
    while stack:
        for r in known[stack.pop()][1]:
            if r not in need: need.add(r); stack.append(r)
    extra = len(need) - len(chosen)
    log("certificate: the %d certified primes <= %d, and %d larger primes they depend on"
        % (len(chosen), cutoff, extra))

    g = primroot(p)
    handles = [open(fn, "rb") for fn in files]
    rows = []
    for q in sorted(need):
        kind, deps, fid, off = known[q]
        handles[fid].seek(off)
        tag, _, w, N, sup, sgn, fac = LINE.match(handles[fid].readline().decode().rstrip("\n")).groups()
        rows.append("%d %s N=%s support=[%s] signs=[%s]%s%s" % (
            q, kind, N, sup, sgn, " factors=" + fac if kind == "Q" else "",
            " divide_out=" + ",".join(map(str, sorted(deps))) if deps else ""))
    for fh in handles: fh.close()

    text = "\n".join([
        "# Certificate: the class number of B_{%d,1} is 1.  Check it with verify/verify_certificate.py." % p,
        "# basis: b_0 = 1, b_j = sum_{h in H} zeta^{g^(j-1) h}, zeta = exp(2 pi i/%d), g = %d,"
        " H = {h : h^%d = 1 mod %d}" % (m, g, p - 1, m),
        "# row: q kind N=norm support=[...] signs=[...] [factors=...] [divide_out=...];"
        " the element is sum_i signs[i] * b_{support[i]}",
        "# kind P: N = +-q.  kind Q: |N| = product of the factors; the factors other than q are %d,"
        " inert primes, and the divide_out primes, each certified in its own row." % p,
        "# rows: the %d totally split primes q <= %d certified by the search, and %d larger primes"
        " they depend on." % (len(chosen), cutoff, extra),
    ] + rows) + "\n"
    with open(out, "w") as fo: fo.write(text)
    log("written to %s\nSHA-256 %s" % (out, hashlib.sha256(text.encode()).hexdigest()))
    return text

# ----------------------------------------------------------------- self test
def selftest():
    ok = True
    def check(label, good):
        nonlocal ok
        ok &= good
        print("%-60s %s" % (label, "OK" if good else "FAIL"))
    line = "FACTORED p=23 w=2 N=-94 split=0 support=[0,3] signs=[+1,-1] factors=2*47"
    def accepted(text):
        mt = LINE.match(text)
        if not mt: return False
        _, _, w, N, sup, sgn, fac = mt.groups()
        return parse_vec(sup, sgn, int(w), 23) is not None and math.prod(
            int(x) for x in fac.split("*")) == abs(int(N))
    check("parser accepts a good line", accepted(line))
    check("parser rejects a truncated line", not accepted(line[:-2]))
    check("parser rejects factors that do not multiply to |N|", not accepted(line.replace("2*47", "2*43")))
    check("parser rejects an unsorted support", not accepted(line.replace("[0,3]", "[3,0]")))
    check("parser rejects a negative first coefficient", not accepted(line.replace("[+1,-1]", "[-1,-1]")))
    check("smallest primitive root mod 529 is 5, mod 169 is 2", primroot(23) == 5 and primroot(13) == 2)

    # A miniature search output, built from rows of the real certificate: 2179 is
    # certified directly (norm 23^3 * 2179), 263 by the quotient trick, and so on.
    hits = [
        "PRIME p=23 w=14 N=-607298377879 split=1 support=[1,2,3,6,8,10,11,12,13,15,16,19,20,21] signs=[+1,-1,-1,+1,-1,-1,+1,+1,-1,+1,+1,+1,+1,-1]",
        "FACTORED p=23 w=9 N=159719473382177 split=0 support=[0,1,3,5,7,12,15,17,18] signs=[+1,+1,-1,+1,+1,+1,+1,-1,+1] factors=263*607298377879",
        "FACTORED p=23 w=8 N=26511893 split=1 support=[1,2,3,4,14,17,20,22] signs=[+1,-1,+1,+1,-1,-1,-1,+1] factors=23*23*23*2179",
        "FACTORED p=23 w=9 N=26511893 split=1 support=[1,2,3,4,14,17,20,21,22] signs=[+1,-1,+1,+1,-1,-1,-1,+1,+1] factors=23*23*23*2179",
        "FACTORED p=23 w=4 N=-25668533354753 split=0 support=[0,1,8,16] signs=[+1,+1,+1,+1] factors=647*659*2371*25391",
        "UNIT p=23 w=1 N=1 split=0 support=[0] signs=[+1]",
    ]
    with tempfile.TemporaryDirectory() as d:
        outs = []
        for order, nfiles in ((hits, 1), (hits[::-1], 3)):
            for f in glob.glob(os.path.join(d, "t*")): os.remove(f)
            for i, h in enumerate(order):
                with open(os.path.join(d, "t_shard000_rank%05d.txt" % (i % nfiles)), "a") as fo:
                    fo.write(h + "\n")
            outs.append(certify(os.path.join(d, "t"), os.path.join(d, "cert.txt"), log=lambda *a: None))
    rows = [l for l in outs[0].split("\n") if l and not l.startswith("#")]
    check("output does not depend on line order or file layout", outs[0] == outs[1])
    check("certifies 263, 2179 and 607298377879, and nothing else",
          [int(l.split()[0]) for l in rows] == [263, 2179, 607298377879])
    check("263 uses the quotient trick with 607298377879 divided out",
          rows[0].startswith("263 Q") and rows[0].endswith("divide_out=607298377879"))
    check("2179 uses the element with fewer nonzero coefficients", "support=[1,2,3,4,14,17,20,22]" in rows[1])
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]: sys.exit(selftest())
    opts = {"--p": "23", "--cutoff": "50000000", "--out": None}
    pos, i = [], 0
    while i < len(args):
        if args[i] in opts and i + 1 < len(args): opts[args[i]] = args[i + 1]; i += 2
        else: pos.append(args[i]); i += 1
    if len(pos) != 1 or not opts["--out"]: sys.exit(__doc__)
    certify(pos[0], opts["--out"], int(opts["--p"]), int(float(opts["--cutoff"])))
