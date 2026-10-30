"""
Permissioned blockchain with smart contracts and PBFT-style consensus (in-process simulation).

What is implemented (and what is not)
  * Blocks are linked by hashes:  hash = SHA-256(JSON({index, timestamp, content, prev_hash}, sorted keys))
    (the block-hash rule described in Bhardwaj & Sumangali 2025, Eq. 17, and used for update hashes by
    Otoum et al. 2024).
  * N = 3f+1 validator nodes; a block is committed only after the three PBFT phases
    (pre-prepare, prepare, commit) with 2f+1 matching votes (Castro & Liskov, OSDI 1999).
    Every validator re-computes the block hash, checks the link to its own copy of the chain and
    runs the smart-contract rule before it votes. Each validator keeps its own copy of the chain.
  * Smart contract (FLContract): model initialisation on chain, update validation
    (hash check + quality gate), aggregation records, and an ERC-20-like reward token.
  * Chain verification re-computes every hash on every validator copy (tamper evidence).
  Not implemented: a networked blockchain (peer-to-peer messaging, view changes, Byzantine validators).
  Validators are honest in the simulation; a Byzantine-validator test is provided in tests/.
"""
import hashlib, json, time


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_array(a) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


class Block:
    def __init__(self, index, timestamp, content, prev_hash):
        self.index, self.timestamp, self.content, self.prev_hash = index, timestamp, content, prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        return sha256_json({"index": self.index, "timestamp": self.timestamp,
                            "content": self.content, "prev_hash": self.prev_hash})


class FLContract:
    """Smart-contract logic executed by every validator before it votes."""
    def __init__(self, quality_gate=None):
        self.quality_gate = quality_gate          # callable(content) -> bool, or None
        self.balances = {}                        # ERC-20-like reward token
        self.initialised = False

    def validate(self, content):
        kind = content.get("type")
        if kind == "init":
            return "model_hash" in content and not self.initialised
        if kind == "update":                      # hash of the submitted update must equal the claimed one
            ok = content.get("claimed_hash") == content.get("update_hash")
            if ok and self.quality_gate is not None: ok = bool(self.quality_gate(content))
            return ok
        if kind in ("aggregate", "explanation", "reward"):
            return "round" in content
        return False

    def apply(self, content):
        if content["type"] == "init": self.initialised = True
        if content["type"] == "reward":
            for cid, amt in content["rewards"].items():
                self.balances[cid] = self.balances.get(cid, 0.0) + amt


class Validator:
    def __init__(self, vid, contract_factory):
        self.vid = vid
        self.chain = [Block(0, 0, {"type": "genesis"}, "0" * 64)]
        self.contract = contract_factory()

    def check(self, block):
        last = self.chain[-1]
        return (block.index == last.index + 1 and block.prev_hash == last.hash
                and block.hash == block.compute_hash() and self.contract.validate(block.content))

    def commit(self, block):
        self.chain.append(block); self.contract.apply(block.content)

    def verify_chain(self):
        for i, b in enumerate(self.chain):
            if b.hash != b.compute_hash(): return False
            if i and b.prev_hash != self.chain[i - 1].hash: return False
        return True


class PermissionedChain:
    def __init__(self, n_validators=4, quality_gate=None):
        assert n_validators >= 4 and (n_validators - 1) % 3 == 0, "PBFT needs N = 3f+1"
        self.n, self.f = n_validators, (n_validators - 1) // 3
        self.validators = [Validator(i, lambda: FLContract(quality_gate)) for i in range(n_validators)]
        self.messages = 0                         # PBFT message counter (cost metric)
        self.committed = 0
        self.rejected = 0

    @property
    def chain(self): return self.validators[0].chain

    def submit(self, content, byzantine=()):
        """Leader (validator 0) proposes; PBFT three-phase commit. Returns True if committed."""
        leader = self.validators[0]
        block = Block(len(leader.chain), self.committed + self.rejected + 1, content, leader.chain[-1].hash)
        self.messages += self.n - 1                                   # pre-prepare
        votes = 0
        for v in self.validators:
            ok = v.check(block) and v.vid not in byzantine            # byzantine validators vote against
            votes += ok
        self.messages += (self.n - 1) * self.n * 2                    # prepare + commit all-to-all
        if votes >= 2 * self.f + 1:
            for v in self.validators: v.commit(block)
            self.committed += 1
            return True
        self.rejected += 1
        return False

    def verify(self):
        same = all(v.chain[-1].hash == self.validators[0].chain[-1].hash for v in self.validators)
        return same and all(v.verify_chain() for v in self.validators)

    def balances(self): return dict(self.validators[0].contract.balances)


class PoSChain(PermissionedChain):
    """Proof-of-Stake chain used by BPPFL (Asad & Otoum 2025, Sec. 3.3.2): the block proposer is drawn with probability
    P_i = s_i / sum_j s_j (Eq. 8); every node validates the block (hash link + smart-contract rule, e.g. the Schnorr signature check)
    and attests; the block is final when the attesting stake is >= 2/3 of the total stake. Stake grows with rewards (incentive contract).
    Simulation: in-process nodes, no fork choice / slashing."""
    def __init__(self, n_nodes=10, quality_gate=None, seed=0):
        self.n, self.f = n_nodes, (n_nodes - 1) // 3
        self.validators = [Validator(i, lambda: FLContract(quality_gate)) for i in range(n_nodes)]
        self.stake = {i: 1.0 for i in range(n_nodes)}
        self.messages = 0; self.committed = 0; self.rejected = 0
        import random; self._rng = random.Random(seed)

    def submit(self, content, byzantine=()):
        tot = sum(self.stake.values())
        r = self._rng.random() * tot; acc = 0.0; prop = 0
        for i, s in self.stake.items():
            acc += s
            if r <= acc: prop = i; break
        leader = self.validators[prop]
        block = Block(len(leader.chain), self.committed + self.rejected + 1, content, leader.chain[-1].hash)
        self.messages += self.n - 1                                     # proposal broadcast
        att = sum(self.stake[v.vid] for v in self.validators if v.check(block) and v.vid not in byzantine)
        self.messages += 2 * (self.n - 1)                               # attestations + finality broadcast
        if att >= 2 / 3 * tot:
            for v in self.validators: v.commit(block)
            self.committed += 1
            if content.get("type") == "reward":
                for cid, amt in content["rewards"].items(): self.stake[int(cid)] += amt
            return True
        self.rejected += 1
        return False
