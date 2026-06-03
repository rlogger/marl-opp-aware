"""Semantic Scholar literature review: place this project next to the SOTA.

Centered on the lab's actual thesis -- MODEL-BASED multi-agent RL (multi-agent
MuZero-style baselines), sample efficiency, and handling ADAPTIVE opponents under
limited interaction -- plus the representation gap the team diagnosed (the VAE
encodes map-location, not rotation/reflection-invariant strategy), and the JEPA /
belief-planning angles this project adds.

Hits the S2 graph API (rate-limited to 1 req/sec), filters obvious off-topic noise
(LLM-agent frameworks, networking/IoT apps), citation-ranks each family, and saves
full records (with abstracts) for annotation.

Output: logs/lit_review_s2.json  (raw), plus a printed digest.
"""
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/search"
KEY = os.environ.get("S2_API_KEY", "")
FIELDS = ("title,year,venue,citationCount,influentialCitationCount,authors,"
          "externalIds,abstract")
HEADERS = {"x-api-key": KEY} if KEY else {}

# Families chosen to surface the model-based-MARL baselines the lab cares about,
# the adaptive-opponent / non-stationarity line, the representation-gap line, and
# the JEPA / planning angles this project adds.
QUERIES = [
    # --- model-based MARL: the lab's core baseline family ---
    ("mbarl:multi-agent-muzero", "multi-agent MuZero planning learned model reinforcement"),
    ("mbarl:world-model-marl", "model-based multi-agent reinforcement learning world model scalable"),
    ("mbarl:muzero", "mastering atari go chess shogi planning with a learned model"),
    ("mbarl:efficientzero", "sample efficient model-based reinforcement learning mastering atari limited data"),
    ("mbarl:dreamer", "dream to control world models latent imagination behaviours"),
    ("mbarl:model-based-opp", "model-based opponent modeling planning reinforcement learning"),
    ("mbarl:centralized-model", "centralized world model multi-agent reinforcement learning communication"),
    # --- opponent modeling / theory of mind / adaptivity / non-stationarity ---
    ("opp:deep-om", "opponent modeling deep reinforcement learning"),
    ("opp:tom", "machine theory of mind agent modeling behaviour meta-learning"),
    ("opp:survey", "autonomous agents modelling other agents comprehensive survey"),
    ("opp:nonstationary", "dealing with non-stationarity multi-agent reinforcement learning opponents"),
    ("opp:continual-adapt", "continuous adaptation via meta-learning nonstationary competitive environments"),
    ("opp:bayes-type", "Bayesian opponent type inference uncertainty reinforcement learning"),
    ("opp:recursive", "probabilistic recursive reasoning level-k multi-agent reinforcement learning"),
    ("opp:robust-ensemble", "robust opponent modeling adversarial ensemble reinforcement learning"),
    # --- representation: the 'strategy not position' gap ---
    ("rep:vae-om", "variational autoencoder opponent modeling multi-agent local information"),
    ("rep:equivariant", "equivariant symmetry MDP homomorphic networks reinforcement learning"),
    ("rep:trajectory", "trajectory representation learning behaviour embedding policy"),
    ("rep:skill-discovery", "unsupervised skill discovery diversity is all you need latent reinforcement learning"),
    ("rep:contrastive", "contrastive learning state representation reinforcement learning"),
    ("rep:self-predictive", "self-predictive representations reinforcement learning momentum target"),
    # --- JEPA / predictive SSL for control ---
    ("jepa:core", "joint embedding predictive architecture self-supervised representation"),
    ("jepa:control", "joint embedding predictive world model planning control navigation latent"),
    # --- planning over beliefs / tree search ---
    ("plan:mcts-marl", "Monte Carlo tree search multi-agent planning learned model"),
    ("plan:ipomdp", "interactive POMDP belief space planning partially observable multi-agent"),
    ("plan:intent-plan", "intention prediction belief update planning interactive autonomous agents"),
]

# drop obviously off-topic hits (the 'multi-agent' keyword pulls LLM/IoT spam)
DENY = [
    "large language model", " llm", "llm ", "llm-", "gpt-", "chatbot", "agentic ai",
    "malware", "traffic light", "traffic signal", "satellite", "double auction",
    "cloud-native", "cloud resource", "packet routing", "jamming", "ehr", "ecg",
    "collider", "software design", "refactoring", "portfolio", "stock market",
    "recommendation", "wireless", "edge computing", "iot network", "air combat",
    "spiking neural", "is sora", "social network structures",
]


def search(query, limit=25):
    url = API + "?" + urllib.parse.urlencode(
        {"query": query, "limit": limit, "fields": FIELDS})
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 504) and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}", "body": e.read().decode()[:200]}
        except Exception as e:  # noqa
            if attempt < 3:
                time.sleep(2)
                continue
            return {"error": str(e)}
    return {"error": "exhausted retries"}


def authors(p, n=3):
    a = [x["name"] for x in (p.get("authors") or [])]
    return (", ".join(a[:n]) + (" et al." if len(a) > n else "")) if a else "?"


def ident(p):
    ext = p.get("externalIds") or {}
    if "ArXiv" in ext:
        return f"arXiv:{ext['ArXiv']}"
    return ext.get("DOI", "")


def off_topic(p):
    blob = (p.get("title", "") + " " + (p.get("abstract") or "")).lower()
    return any(k in blob for k in DENY)


def main():
    if not KEY:
        print("!! S2_API_KEY not set in environment; aborting.")
        return
    out = {}
    for label, q in QUERIES:
        res = search(q)
        if "error" in res:
            print(f"\n### {label}\n  ERROR: {res['error']}  {res.get('body','')}")
            out[label] = {"query": q, "error": res["error"]}
            time.sleep(1.2)
            continue
        papers = [p for p in (res.get("data") or [])
                  if p.get("title") and not off_topic(p)]
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        out[label] = {"query": q, "papers": papers}
        print(f"\n### {label}   (query: {q!r})")
        for p in papers[:7]:
            cc = p.get("citationCount") or 0
            yr = p.get("year") or "----"
            print(f"  [{cc:>6}] {yr}  {authors(p, 1):<16} {p['title'][:84]}  {ident(p)}")
        time.sleep(1.2)  # respect 1 req/sec

    os.makedirs("logs", exist_ok=True)
    with open("logs/lit_review_s2.json", "w") as f:
        json.dump(out, f, indent=2)
    n = sum(len(v.get("papers", [])) for v in out.values())
    print(f"\nsaved logs/lit_review_s2.json  ({n} papers across {len(QUERIES)} families)")


if __name__ == "__main__":
    main()
