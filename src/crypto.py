"""
Cryptographic primitives of BPPFL (Asad & Otoum 2025 [3]); real implementations, not mocks.

  * Threshold Paillier (TPE): Paillier cryptosystem (g = n+1) whose decryption key d (d = 0 mod lambda, d = 1 mod n) is Shamir-shared
    among n participants with threshold t (trusted dealer = the paper's Trusted Authority). Decryption shares c_i = c^(2 Delta s_i) mod N^2
    are combined with integer Lagrange coefficients (Damgard-Jurik / Fouque-Poupard-Stern style). Any t shares decrypt; t-1 do not.
    Several model parameters are packed into one plaintext (fixed point, 36-bit slots) so that one ciphertext carries ~55 parameters.
  * Threshold Schnorr signature (TSA): the signing key x is Shamir-shared t-of-n over Z_q; the t cooperating signers produce
    partial signatures s_i = k_i + e * lambda_i * x_i; the combined signature (e, s) verifies as an ordinary Schnorr signature under y = g^x.
  Simulation notes: a non-cryptographic seeded PRNG is used for reproducibility (use os.urandom in production); ordinary (not safe)
  primes are used for Paillier since the dealer is trusted; no network. Requires gmpy2 for speed (falls back to Python ints).
"""
import hashlib, json, math, os, random
try:
    import gmpy2
    from gmpy2 import mpz, powmod, invert, next_prime, is_prime
except ImportError:                                           # slow fallback
    mpz = int; powmod = pow; invert = lambda a, m: pow(a, -1, m)
    def is_prime(n, k=25):
        if n < 2: return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
            if n % p == 0: return n == p
        d, s = n - 1, 0
        while d % 2 == 0: d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1); x = pow(a, d, n)
            if x in (1, n - 1): continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
        return True
    def next_prime(n):
        n += 1 + (n % 2 == 0)
        while not is_prime(n): n += 2
        return n


def _rand_prime(bits, rng):
    return next_prime(mpz(rng.getrandbits(bits)) | (mpz(1) << (bits - 1)) | 1)


def _shamir(secret, t, n, modulus, rng):
    coef = [secret] + [mpz(rng.getrandbits(modulus.bit_length() + 64)) % modulus for _ in range(t - 1)]
    return {i: sum(c * mpz(i) ** k for k, c in enumerate(coef)) % modulus for i in range(1, n + 1)}


# ----------------------------------------------------------------------------------------- threshold Paillier
class ThresholdPaillier:
    SLOT, SCALE, OFFSET_BITS = 36, 20, 32          # 36-bit slots (room for sums of <=16 clients), 2^-20 resolution, 32-bit signed range

    def __init__(self, n_parts=10, t=7, prime_bits=1024, seed=0):
        rng = random.Random(seed)
        p = _rand_prime(prime_bits, rng); q = _rand_prime(prime_bits, rng)
        while q == p: q = _rand_prime(prime_bits, rng)
        self.n = p * q; self.n2 = self.n ** 2
        lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        d = lam * invert(lam, self.n)                                  # d = 0 mod lam, d = 1 mod n
        self.N, self.t, self.delta = n_parts, t, math.factorial(n_parts)
        self.shares = _shamir(d % (self.n * lam), t, n_parts, self.n * lam, rng)
        self.rng = rng
        self.per_ct = (self.n.bit_length() - 2) // self.SLOT           # parameters per ciphertext
        self.offset = 1 << (self.OFFSET_BITS - 1)
        self.clipped = 0

    # fixed-point packing
    def encode(self, v):
        lim = (1 << (self.OFFSET_BITS - 1)) - 1
        x = [int(round(a * (1 << self.SCALE))) for a in v]
        self.clipped += sum(abs(a) > lim for a in x)
        x = [min(max(a, -lim), lim) + self.offset for a in x]
        return [sum(int(x[i + j]) << (self.SLOT * j) for j in range(min(self.per_ct, len(x) - i))) for i in range(0, len(x), self.per_ct)]

    def decode_sum(self, pts, length, count):
        out = []
        for k, m in enumerate(pts):
            for j in range(min(self.per_ct, length - k * self.per_ct)):
                out.append(((int(m) >> (self.SLOT * j)) & ((1 << self.SLOT) - 1)) - count * self.offset)
        return [a / (1 << self.SCALE) for a in out]

    def encrypt_ints(self, ms):
        res = []
        for m in ms:
            r = mpz(self.rng.getrandbits(self.n.bit_length() - 1)) | 1
            res.append((1 + mpz(m) * self.n) * powmod(r, self.n, self.n2) % self.n2)   # (1+n)^m r^n mod n^2
        return res

    def encrypt(self, v): return self.encrypt_ints(self.encode(v))

    def add(self, cts):                                                  # homomorphic sum = product of ciphertexts (Eq. 16)
        agg = [mpz(1)] * len(cts[0])
        for c in cts: agg = [a * b % self.n2 for a, b in zip(agg, c)]
        return agg

    def partial_decrypt(self, i, c): return powmod(c, 2 * self.delta * self.shares[i], self.n2)       # Eq. (17)

    def combine(self, shares_by_i, c_count=None):                       # Eq. (18); shares_by_i: {i: share}
        S = sorted(shares_by_i)
        assert len(S) >= self.t, "need at least t shares"
        S = S[:self.t]; acc = mpz(1)
        for i in S:
            num = self.delta
            den = 1
            for j in S:
                if j != i: num *= j; den *= (j - i)
            mu = num // den                                              # integer: Delta * prod j/(j-i)
            base = shares_by_i[i]
            acc = acc * (powmod(base, 2 * mu, self.n2) if mu >= 0 else powmod(invert(base, self.n2), -2 * mu, self.n2)) % self.n2
        L = (acc - 1) // self.n
        return L * invert(4 * self.delta ** 2 % self.n, self.n) % self.n

    def decrypt_threshold(self, cts, parts):
        """parts: ids of cooperating participants (>= t). Returns the plaintext integers."""
        return [self.combine({i: self.partial_decrypt(i, c) for i in parts}) for c in cts]


# ----------------------------------------------------------------------------------------- threshold Schnorr
_GROUP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "schnorr_group.json")


def schnorr_group(seed=1):
    """2048-bit prime p with a 256-bit prime-order-q subgroup generated by g (generated once, cached)."""
    if os.path.exists(_GROUP):
        d = json.load(open(_GROUP)); return mpz(d["p"]), mpz(d["q"]), mpz(d["g"])
    rng = random.Random(seed); q = _rand_prime(256, rng)
    while True:
        k = mpz(rng.getrandbits(1792)) | (mpz(1) << 1791); k += k % 2
        p = k * q + 1
        if is_prime(p): break
    h = mpz(2)
    while powmod(h, k, p) == 1: h += 1
    g = powmod(h, k, p)
    os.makedirs(os.path.dirname(_GROUP), exist_ok=True)
    json.dump(dict(p=str(p), q=str(q), g=str(g)), open(_GROUP, "w"))
    return p, q, g


def _H(*parts, q):
    h = hashlib.sha256("|".join(str(x) for x in parts).encode()).digest()
    return int.from_bytes(h, "big") % int(q)


class ThresholdSchnorr:
    def __init__(self, n_parts=10, t=7, seed=2):
        self.p, self.q, self.g = schnorr_group(); self.n, self.t = n_parts, t
        rng = random.Random(seed); self.rng = rng
        x = mpz(rng.getrandbits(300)) % self.q
        self.shares = _shamir(x, t, n_parts, self.q, rng); self.y = powmod(self.g, x, self.p)

    def _lagrange(self, S, i):
        num, den = 1, 1
        for j in S:
            if j != i: num = num * j % int(self.q); den = den * (j - i) % int(self.q)
        return num * int(invert(den, self.q)) % int(self.q)

    def sign(self, msg, signers):
        S = sorted(signers)[:self.t]; ks = {i: mpz(self.rng.getrandbits(300)) % self.q for i in S}
        R = mpz(1)
        for i in S: R = R * powmod(self.g, ks[i], self.p) % self.p
        e = _H(R, self.y, msg, q=self.q)
        s = sum(int(ks[i]) + e * self._lagrange(S, i) * int(self.shares[i]) for i in S) % int(self.q)      # partial sigs combined (Eq. 21)
        return (e, s)

    def verify(self, msg, sig):                                                                              # Eq. (22)
        e, s = sig
        R = powmod(self.g, mpz(s), self.p) * invert(powmod(self.y, mpz(e), self.p), self.p) % self.p
        return _H(R, self.y, msg, q=self.q) == e
