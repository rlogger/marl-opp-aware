"""Backfill the families that hit HTTP 429 in the main sweep, plus targeted
by-name queries for the papers most relevant to the 'strategy not position' gap
(equivariance / MDP homomorphic nets) and JEPA-for-control. Merges into the
existing logs/lit_review_s2.json. Slow spacing (2.5s) to respect the rate limit.
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

QUERIES = [
    ("opp:survey", "autonomous agents modelling other agents comprehensive survey open problems"),
    ("rep:equivariant", "MDP homomorphic networks symmetry equivariant deep reinforcement learning"),
    ("rep:equivariant-marl", "multi-agent MDP homomorphic networks symmetry equivariant policy"),
    ("rep:trajectory", "trajectory representation learning behaviour embedding contrastive policy"),
    ("jepa:control", "joint embedding predictive world model planning navigation control"),
    ("jepa:dino-wm", "DINO world model latent planning model predictive control self-supervised"),
    ("opp:lemol", "learning to model opponent learning anticipation multi-agent"),
    ("mbarl:mazero", "multi-agent MuZero tree search learned model cooperative MAZero"),
]
DENY = ["large language model", " llm", "malware", "traffic light", "satellite",
        "auction", "packet routing", "jamming", "wireless", "edge computing",
        "volt", "distribution network", "spectrum sharing"]


def search(query, limit=25):
    url = API + "?" + urllib.parse.urlencode(
        {"query": query, "limit": limit, "fields": FIELDS})
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 504) and attempt < 4:
                time.sleep(4 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}"}
        except Exception as e:  # noqa
            if attempt < 4:
                time.sleep(3)
                continue
            return {"error": str(e)}
    return {"error": "exhausted"}


def off_topic(p):
    blob = (p.get("title", "") + " " + (p.get("abstract") or "")).lower()
    return any(k in blob for k in DENY)


def auth1(p):
    a = p.get("authors") or []
    return a[0]["name"] if a else "?"


def ident(p):
    ext = p.get("externalIds") or {}
    return f"arXiv:{ext['ArXiv']}" if "ArXiv" in ext else ext.get("DOI", "")


def main():
    path = "logs/lit_review_s2.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for label, q in QUERIES:
        time.sleep(2.5)
        res = search(q)
        if "error" in res:
            print(f"### {label}: ERROR {res['error']}")
            continue
        papers = [p for p in (res.get("data") or [])
                  if p.get("title") and not off_topic(p)]
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        out[label] = {"query": q, "papers": papers}
        print(f"\n### {label}   (query: {q!r})")
        for p in papers[:7]:
            print(f"  [{p.get('citationCount') or 0:>6}] {p.get('year') or '----'}  "
                  f"{auth1(p):<20} {p['title'][:78]}  {ident(p)}")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    n = sum(len(v.get("papers", [])) for v in out.values())
    print(f"\nmerged -> {path}  ({n} papers, {len(out)} families)")


if __name__ == "__main__":
    main()
