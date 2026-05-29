"""Mine a seed bank from the existing K-BrowseComp curation + failure-mode runs.

Reads from ./seed_material/ and writes back into the same directory (treated as
the canonical home for everything Stage 2 and Stage 3 consume):

- seed_material/seed_bank.json
                              Labeled source-class taxonomy plus per-class statistics:
                              URL count, distinct question count, failure-mode distribution
                              (joint across both solver models, plus per-model breakdown),
                              category mix, sample URLs.
- seed_material/seed_url_index.json
                              Flat per-URL index (host, path, class, question_id, category,
                              failure_mode per model). For downstream seed sampling.
- seed_material/failure_mode_exemplars.json
                              Per failure mode, 4-6 curated failed trajectories (digest only)
                              for in-context conditioning of the proposer.
- seed_material/seed_material_summary.md
                              Human-readable summary of class coverage, top hosts, and
                              fallback hosts the rule table did not catch.

Run:
    python automated_data_gen/mine_seed_bank.py
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
SEED_MATERIAL = ROOT / "seed_material"

SEED_QUESTIONS = SEED_MATERIAL / "seed_questions.json"
SEED_BANK_OUT = SEED_MATERIAL / "seed_bank.json"
SEED_URL_INDEX_OUT = SEED_MATERIAL / "seed_url_index.json"
FAILURE_MODE_EXEMPLARS_OUT = SEED_MATERIAL / "failure_mode_exemplars.json"
SUMMARY_OUT = SEED_MATERIAL / "seed_material_summary.md"
CLASSIFICATION = {
    "gemini-3.1-flash-lite-preview": SEED_MATERIAL / "classification_gemini.jsonl",
    "gpt-5.4-mini": SEED_MATERIAL / "classification_gpt5.jsonl",
}
TRAJECTORIES = {
    "gemini-3.1-flash-lite-preview": SEED_MATERIAL / "trajectories_gemini",
    "gpt-5.4-mini": SEED_MATERIAL / "trajectories_gpt5",
}

FAILURE_MODE_NAMES = {
    "F1": "첫 검색어를 좁힐 수 없음 / 사후 검증형 문제",
    "F3": "비인접 도메인 hopping",
    "F4": "semi-structured parsing 실패",
    "F6": "희소 엔티티 정규화 실패",
    "F7": "조건 누적 / constraint tracking 실패",
    "F8": "중간 계산 / 절차형 reasoning 실패",
    "F9": "검색 결과 선택 실패",
    "F10": "iframe / 동적 페이지 / 특정 페이지 진입 실패",
}


# ---------------------------------------------------------------------------
# Source-class rule table.
#
# Order matters: first match wins. Each rule has:
#   - predicate(host, path_segments_decoded, raw_url) -> bool
#   - class_name
#   - prior_failure_modes: human hypothesis of which failure modes this class tends
#     to produce. Empirical failure-mode distribution comes from data (the joins
#     with classification.jsonl below); this prior is for human review only.
#   - retrieval_hypothesis: one-line rationale to record in the report.
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    name: str
    predicate: Callable[[str, list[str], str], bool]
    prior_failure_modes: list[str]
    retrieval_hypothesis: str


def _has_segment(parts: list[str], prefix: str) -> bool:
    return any(p.startswith(prefix) for p in parts)


RULES: list[Rule] = [
    Rule(
        name="namu_wiki_subsection",
        predicate=lambda h, ps, _u: h == "namu.wiki" and len(ps) > 2 and ps[0] == "w",
        prior_failure_modes=["F4", "F7"],
        retrieval_hypothesis="namu.wiki sub-pages: deep tables/sub-lists not surfaced in SERP snippets",
    ),
    Rule(
        name="namu_wiki_main",
        predicate=lambda h, ps, _u: h == "namu.wiki" and ps[:1] == ["w"],
        prior_failure_modes=["F4", "F7", "F6"],
        retrieval_hypothesis="namu.wiki main articles: dense semi-structured content with sub-sections",
    ),
    Rule(
        name="wikipedia_ko_category",
        predicate=lambda h, ps, _u: h == "ko.wikipedia.org" and any(p.startswith("분류:") for p in ps),
        prior_failure_modes=["F1"],
        retrieval_hypothesis="Wikipedia 분류 pages: force candidate enumeration; no obvious starting entity",
    ),
    Rule(
        name="wikipedia_ko_article",
        predicate=lambda h, ps, _u: h == "ko.wikipedia.org",
        prior_failure_modes=["F3", "F7"],
        retrieval_hypothesis="Korean Wikipedia articles: cross-domain hops via inline links",
    ),
    Rule(
        name="wikipedia_en_article",
        predicate=lambda h, ps, _u: h == "en.wikipedia.org",
        prior_failure_modes=["F3", "F6"],
        retrieval_hypothesis="English Wikipedia: rare/transliterated Korean entities and cross-language hops",
    ),
    Rule(
        name="naver_search",
        predicate=lambda h, _ps, _u: h == "search.naver.com",
        prior_failure_modes=["F9"],
        retrieval_hypothesis="Naver SERPs: deep result-selection from many similar hits",
    ),
    Rule(
        name="naver_blog",
        predicate=lambda h, _ps, _u: h in {"blog.naver.com", "m.blog.naver.com"},
        prior_failure_modes=["F9", "F4"],
        retrieval_hypothesis="Naver Blog: long-form user posts; answer in a single line buried in prose",
    ),
    Rule(
        name="naver_news",
        predicate=lambda h, _ps, _u: h in {"news.naver.com", "n.news.naver.com"},
        prior_failure_modes=["F9"],
        retrieval_hypothesis="Naver News redirects: stable URLs to news articles",
    ),
    Rule(
        name="naver_movie",
        predicate=lambda h, _ps, _u: h == "movie.naver.com",
        prior_failure_modes=["F4"],
        retrieval_hypothesis="Naver movie detail pages: table cells, ratings, cast rows",
    ),
    Rule(
        name="naver_encyclopedia",
        predicate=lambda h, _ps, _u: h == "terms.naver.com",
        prior_failure_modes=["F4", "F6"],
        retrieval_hypothesis="Naver Knowledge Encyclopedia: structured but sparse on niche entries",
    ),
    Rule(
        name="naver_other",
        predicate=lambda h, _ps, _u: h.endswith("naver.com"),
        prior_failure_modes=["F9"],
        retrieval_hypothesis="Other Naver subdomains (sports/maps/etc.); often dynamic UIs",
    ),
    Rule(
        name="daum",
        predicate=lambda h, _ps, _u: h.endswith("daum.net"),
        prior_failure_modes=["F9"],
        retrieval_hypothesis="Daum portal pages (news / aggregator)",
    ),
    Rule(
        name="nate",
        predicate=lambda h, _ps, _u: h.endswith("nate.com"),
        prior_failure_modes=["F9"],
        retrieval_hypothesis="Nate news aggregator",
    ),
    Rule(
        name="youtube_video",
        predicate=lambda h, _ps, _u: h in {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"},
        prior_failure_modes=["F10"],
        retrieval_hypothesis="YouTube: media-only content; the answer often lives in video/audio not text",
    ),
    Rule(
        name="instagram",
        predicate=lambda h, _ps, _u: h.endswith("instagram.com"),
        prior_failure_modes=["F10"],
        retrieval_hypothesis="Instagram: login/dynamic walls; SERP rarely shows the relevant detail",
    ),
    Rule(
        name="encyclopedia_korean_culture",
        predicate=lambda h, _ps, _u: h == "encykorea.aks.ac.kr",
        prior_failure_modes=["F6", "F4"],
        retrieval_hypothesis="한국민족문화대백과사전: niche cultural/historical entries; rare-entity normalization",
    ),
    Rule(
        name="species_inventory",
        predicate=lambda h, _ps, _u: h == "species.nibr.go.kr",
        prior_failure_modes=["F4", "F6"],
        retrieval_hypothesis="Korean species inventory: scientific-name tables; row-level lookups",
    ),
    Rule(
        name="government_notice",
        predicate=lambda h, _ps, _u: h.endswith(".go.kr"),
        prior_failure_modes=["F4", "F10"],
        retrieval_hypothesis="*.go.kr notices/공고: heavy tables, dynamic listing, 별표/별첨",
    ),
    Rule(
        name="academic_korean",
        predicate=lambda h, _ps, _u: h.endswith(".ac.kr"),
        prior_failure_modes=["F4", "F10"],
        retrieval_hypothesis="*.ac.kr universities/academies; academic catalogs and notices",
    ),
    Rule(
        name="organization_korean",
        predicate=lambda h, _ps, _u: h.endswith(".or.kr"),
        prior_failure_modes=["F4"],
        retrieval_hypothesis="*.or.kr associations/foundations: rosters, awards, records pages",
    ),
    Rule(
        name="music_streaming",
        predicate=lambda h, _ps, _u: h in {
            "music.bugs.co.kr", "www.bugs.co.kr",
            "www.melon.com", "vibe.naver.com",
            "www.genie.co.kr", "tidal.com",
        },
        prior_failure_modes=["F4"],
        retrieval_hypothesis="Streaming services: tracklist tables, release-date rows, credits",
    ),
    Rule(
        name="bookstore",
        predicate=lambda h, _ps, _u: h in {
            "product.kyobobook.co.kr", "www.kyobobook.co.kr", "store.kyobobook.co.kr",
            "www.yes24.com", "m.yes24.com",
            "www.aladin.co.kr",
            "kupress.com",
        },
        prior_failure_modes=["F4"],
        retrieval_hypothesis="Bookstores/publishers: book metadata rows, ISBN, author/translator credits",
    ),
    Rule(
        name="theatre_ticket",
        predicate=lambda h, _ps, _u: h in {
            "www.themusical.co.kr", "themusical.co.kr",
            "ticket.interpark.com", "tickets.interpark.com", "www.interpark.com",
            "ticket.melon.com", "tickets.naver.com",
        },
        prior_failure_modes=["F4"],
        retrieval_hypothesis="Theatre/musical detail pages: cast rotation tables, venue listings",
    ),
    Rule(
        name="arxiv_paper",
        predicate=lambda h, _ps, _u: h == "arxiv.org",
        prior_failure_modes=["F6", "F3"],
        retrieval_hypothesis="arXiv papers: rare-entity authors / metadata at the abstract page",
    ),
    Rule(
        name="news_legacy",
        predicate=lambda h, _ps, _u: any(h.endswith(d) for d in [
            "chosun.com", "donga.com", "joongang.co.kr", "joins.com",
            "hankyung.com", "hankookilbo.com", "khan.co.kr",
            "yna.co.kr", "ytn.co.kr", "mbn.co.kr", "sbs.co.kr",
            "mbc.co.kr", "kbs.co.kr", "imbc.com", "ohmynews.com",
            "newsis.com", "hani.co.kr", "edaily.co.kr", "mt.co.kr",
            "kmib.co.kr", "munhwa.com", "newstapa.org", "pressian.com",
            "asiae.co.kr", "fnnews.com", "moneys.co.kr",
            "news1.kr", "koreatimes.co.kr", "etnews.com", "mk.co.kr",
            "cine21.com", "soompi.com", "koreaherald.com", "joongdo.co.kr",
            "biz.heraldcorp.com", "heraldcorp.com",
        ]),
        prior_failure_modes=["F9", "F4"],
        retrieval_hypothesis="Legacy Korean news outlets: article pages; date/quote/source extraction",
    ),
    Rule(
        name="government_portal",
        predicate=lambda h, _ps, _u: h in {"www.korea.kr", "korea.kr"},
        prior_failure_modes=["F4", "F9"],
        retrieval_hypothesis="korea.kr: cross-ministry portal; press releases and policy briefings",
    ),
    Rule(
        name="naver_shortlink",
        predicate=lambda h, _ps, _u: h == "naver.me",
        prior_failure_modes=["F9", "F10"],
        retrieval_hypothesis="naver.me short URLs: opaque redirects; harder to extract structured data",
    ),
    Rule(
        name="brunch_blog",
        predicate=lambda h, _ps, _u: h == "brunch.co.kr",
        prior_failure_modes=["F9", "F4"],
        retrieval_hypothesis="Kakao Brunch long-form blog posts; answer often inside prose",
    ),
    Rule(
        name="dbpia_academic",
        predicate=lambda h, _ps, _u: h == "www.dbpia.co.kr",
        prior_failure_modes=["F4", "F6"],
        retrieval_hypothesis="DBpia: Korean academic article DB; metadata-locked PDFs",
    ),
    Rule(
        name="sports_records",
        predicate=lambda h, _ps, _u: h in {
            "www.koreabaseball.com", "koreabaseball.com",
            "statiz.sporki.com", "statiz.co.kr",
            "www.olympics.com", "olympics.com",
            "www.mlb.com", "mlb.com",
            "www.fiba.basketball", "fiba.basketball",
            "www.k-league.com", "k-league.com", "kleague.com",
            "www.leagueoflegends.com", "esports.op.gg",
        },
        prior_failure_modes=["F4", "F8"],
        retrieval_hypothesis="Sports/league record sites: deep stat tables, season-by-season rows",
    ),
    Rule(
        name="search_aggregator",
        predicate=lambda h, _ps, _u: h in {
            "www.google.com", "google.com",
            "scholar.google.com",
        },
        prior_failure_modes=["F9", "F1"],
        retrieval_hypothesis="Search engine result pages cited directly: solver must replicate the query",
    ),
    Rule(
        name="github_repo",
        predicate=lambda h, _ps, _u: h in {"github.com", "raw.githubusercontent.com"},
        prior_failure_modes=["F6", "F4"],
        retrieval_hypothesis="GitHub: code/data file contents; rare-entity normalization",
    ),
    Rule(
        name="global_news",
        predicate=lambda h, _ps, _u: any(h.endswith(d) for d in [
            "bbc.com", "bbc.co.uk", "reuters.com", "nytimes.com",
            "cnn.com", "theguardian.com", "wsj.com",
        ]),
        prior_failure_modes=["F3", "F9"],
        retrieval_hypothesis="Global news outlets: cross-language hops from Korean topics",
    ),
    Rule(
        name="social_misc",
        predicate=lambda h, _ps, _u: h in {
            "www.facebook.com", "facebook.com",
            "twitter.com", "x.com",
            "www.threads.net", "threads.net",
            "luma.com",
            "www.linkedin.com", "linkedin.com",
            "pf.kakao.com",
        },
        prior_failure_modes=["F10"],
        retrieval_hypothesis="Social platforms: dynamic / login walls; SERP rarely surfaces details",
    ),
    Rule(
        name="app_store",
        predicate=lambda h, _ps, _u: h in {"apps.apple.com", "play.google.com"},
        prior_failure_modes=["F4"],
        retrieval_hypothesis="App store pages: structured app metadata (developer, version, rating)",
    ),
    Rule(
        name="ebs_education",
        predicate=lambda h, _ps, _u: h in {"www.ebsi.co.kr", "ebsi.co.kr", "www.ebs.co.kr", "ebs.co.kr"},
        prior_failure_modes=["F4"],
        retrieval_hypothesis="EBS materials: 수능/모의고사 archive, broadcast schedules",
    ),
    Rule(
        name="financial_card",
        predicate=lambda h, _ps, _u: h in {
            "www.samsungcard.com", "samsungcard.com",
            "www.hyundaicard.com", "hyundaicard.com",
            "www.shinhancard.com", "www.kbcard.com", "www.lottecard.com",
            "www.bccard.com",
        },
        prior_failure_modes=["F4", "F10"],
        retrieval_hypothesis="Credit card product pages: benefit rows, partner-store tables",
    ),
    Rule(
        name="korean_research_institute",
        predicate=lambda h, _ps, _u: h.endswith(".re.kr"),
        prior_failure_modes=["F4", "F6"],
        retrieval_hypothesis="*.re.kr Korean research institutes: publications, project rosters",
    ),
]


def classify_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "unparseable"
    host = (parsed.netloc or "").lower()
    raw_path = urllib.parse.unquote(parsed.path or "")
    parts = [p for p in raw_path.split("/") if p]
    for rule in RULES:
        try:
            if rule.predicate(host, parts, url):
                return rule.name
        except Exception:
            continue
    if not host:
        return "unparseable"
    return f"unclassified:{host}"


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def load_seed_questions() -> list[dict[str, Any]]:
    return json.loads(SEED_QUESTIONS.read_text(encoding="utf-8"))


def load_classifications() -> dict[str, dict[str, dict[str, Any]]]:
    """Returns model_name -> normalized_question_text -> classification record."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model, path in CLASSIFICATION.items():
        rows: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = norm_text(row.get("question", ""))
                if key:
                    rows[key] = row
        out[model] = rows
    return out


# ---------------------------------------------------------------------------
# Trajectory parsing (handles both Gemini and OpenAI Responses message shapes).
# ---------------------------------------------------------------------------


def parse_trajectory(convo: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of search steps: [{query, result_count, top_3: [{title, url}]}]."""
    steps: list[dict[str, Any]] = []
    pending_query: str | None = None
    messages = convo.get("messages", []) or []

    def consume_result(payload: Any) -> None:
        if pending_query is None:
            return
        try:
            if isinstance(payload, str):
                payload = json.loads(payload)
            results = payload.get("results") if isinstance(payload, dict) else []
            top3 = [
                {"title": (r.get("title") or "")[:200], "url": (r.get("url") or "")[:300]}
                for r in (results or [])[:3]
            ]
            steps.append({
                "query": pending_query,
                "result_count": len(results or []),
                "top_3": top3,
            })
        except Exception:
            steps.append({"query": pending_query, "result_count": 0, "top_3": []})

    for msg in messages:
        # OpenAI Responses flat schema
        msg_type = msg.get("type")
        if msg_type == "function_call" and msg.get("name") == "search_web":
            try:
                args = json.loads(msg.get("arguments") or "{}")
                pending_query = (args.get("query") or "")[:400]
            except Exception:
                pending_query = None
            continue
        if msg_type == "function_call_output":
            consume_result(msg.get("output"))
            pending_query = None
            continue

        # Anthropic / unified-block schema embedded in content list
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "search_web":
                    pending_query = ((block.get("input") or {}).get("query") or "")[:400]
                if block.get("type") == "tool_result":
                    consume_result(block.get("content"))
                    pending_query = None

        # Gemini nested-content schema
        if isinstance(content, dict):
            parts = content.get("parts") or []
            for p in parts:
                fc = p.get("function_call") if isinstance(p, dict) else None
                if fc and fc.get("name") == "search_web":
                    args = fc.get("args") or {}
                    pending_query = (args.get("query") or "")[:400]
                fr = p.get("function_response") if isinstance(p, dict) else None
                if fr:
                    consume_result(fr.get("response"))
                    pending_query = None

    return steps[:8]  # cap to keep exemplars compact


# ---------------------------------------------------------------------------
# Exemplar curation.
# ---------------------------------------------------------------------------


def curate_exemplars(
    classifications: dict[str, dict[str, dict[str, Any]]],
    per_mode: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """Pick high-confidence failed trajectories per failure mode, stratified across solver models.

    Per model, sort by classification_confidence and take the top half of per_mode. This
    guarantees a mix even when one model's confidences are systematically lower.
    """

    exemplars: dict[str, list[dict[str, Any]]] = {fm: [] for fm in FAILURE_MODE_NAMES}
    n_models = len(TRAJECTORIES)
    quota_per_model = max(1, per_mode // n_models)

    for fm in FAILURE_MODE_NAMES:
        per_model_candidates: dict[str, list[tuple[float, dict[str, Any]]]] = {m: [] for m in TRAJECTORIES}
        for model, traj_dir in TRAJECTORIES.items():
            p = traj_dir / f"{fm}.json"
            if not p.exists():
                continue
            try:
                rows = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            cls_by_question = classifications.get(model, {})
            for row in rows:
                if row.get("failure_mode") != fm:
                    continue
                datum = row.get("datum") or {}
                question = norm_text(datum.get("problem", ""))
                conf = 0.0
                cls = cls_by_question.get(question)
                if cls:
                    try:
                        conf = float(cls.get("classification_confidence") or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                per_model_candidates[model].append((conf, row))

        picked: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        for model, cands in per_model_candidates.items():
            cands.sort(key=lambda t: t[0], reverse=True)
            taken = 0
            for conf, row in cands:
                if taken >= quota_per_model:
                    break
                datum = row.get("datum") or {}
                q = norm_text(datum.get("problem", ""))
                if q in seen_questions:
                    continue
                seen_questions.add(q)
                picked.append({
                    "solver_model": model,
                    "classification_confidence": conf,
                    "datum_id": datum.get("id"),
                    "question": datum.get("problem"),
                    "gold_answer": datum.get("answer"),
                    "model_response": row.get("model_response"),
                    "primary_failure_mode": row.get("failure_mode"),
                    "failure_mode_reason": row.get("failure_mode_reason"),
                    "search_trajectory": parse_trajectory(row.get("convo") or {}),
                })
                taken += 1
        exemplars[fm] = picked

    return exemplars


# ---------------------------------------------------------------------------
# Mining the seed bank.
# ---------------------------------------------------------------------------


@dataclass
class ClassStats:
    name: str
    url_count: int = 0
    question_ids: set[str] = field(default_factory=set)
    categories: Counter = field(default_factory=Counter)
    sample_urls: list[str] = field(default_factory=list)
    failure_mode_by_model: dict[str, Counter] = field(default_factory=dict)
    failure_mode_joint: Counter = field(default_factory=Counter)
    hosts: Counter = field(default_factory=Counter)
    prior_failure_modes: list[str] = field(default_factory=list)
    retrieval_hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prior_failure_modes": self.prior_failure_modes,
            "retrieval_hypothesis": self.retrieval_hypothesis,
            "url_count": self.url_count,
            "distinct_question_count": len(self.question_ids),
            "categories": dict(self.categories.most_common()),
            "hosts": dict(self.hosts.most_common(10)),
            "failure_mode_joint": dict(self.failure_mode_joint.most_common()),
            "failure_mode_by_model": {m: dict(c.most_common()) for m, c in self.failure_mode_by_model.items()},
            "sample_urls": self.sample_urls[:8],
        }


def mine() -> dict[str, Any]:
    seed_questions = load_seed_questions()
    classifications = load_classifications()

    rule_by_name = {r.name: r for r in RULES}

    classes: dict[str, ClassStats] = {}
    flat_index: list[dict[str, Any]] = []
    unclassified_hosts = Counter()

    for q_idx, q in enumerate(seed_questions):
        problem = norm_text(q.get("problem", ""))
        question_id = f"seed_{q_idx:04d}"
        category = q.get("category", "")
        per_model_fm: dict[str, str | None] = {}
        for model, idx in classifications.items():
            cls = idx.get(problem)
            per_model_fm[model] = (cls or {}).get("primary_failure_mode") or None

        for step in q.get("expected_chain", []) or []:
            for url in step.get("sources", []) or []:
                source_class = classify_url(url)
                cs = classes.get(source_class)
                if cs is None:
                    rule = rule_by_name.get(source_class)
                    cs = ClassStats(
                        name=source_class,
                        prior_failure_modes=rule.prior_failure_modes if rule else [],
                        retrieval_hypothesis=rule.retrieval_hypothesis if rule else "",
                    )
                    classes[source_class] = cs
                cs.url_count += 1
                cs.question_ids.add(question_id)
                cs.categories[category] += 1
                if url not in cs.sample_urls and len(cs.sample_urls) < 16:
                    cs.sample_urls.append(url)
                try:
                    host = urllib.parse.urlparse(url).netloc.lower()
                except Exception:
                    host = ""
                if host:
                    cs.hosts[host] += 1
                if source_class.startswith("unclassified:"):
                    unclassified_hosts[host or source_class] += 1

                fm_tags = []
                for model, fm in per_model_fm.items():
                    cs.failure_mode_by_model.setdefault(model, Counter())
                    if fm:
                        cs.failure_mode_by_model[model][fm] += 1
                        fm_tags.append(fm)
                if fm_tags:
                    cs.failure_mode_joint[tuple(sorted(set(fm_tags)))] += 1
                    for fm in set(fm_tags):
                        cs.failure_mode_joint[fm] += 1

                flat_index.append({
                    "question_id": question_id,
                    "category": category,
                    "url": url,
                    "host": host,
                    "source_class": source_class,
                    "failure_modes": per_model_fm,
                })

    chain_archetypes = _compute_chain_archetypes(seed_questions, classifications)

    return {
        "classes": classes,
        "flat_index": flat_index,
        "unclassified_hosts": unclassified_hosts,
        "seed_question_count": len(seed_questions),
        "chain_archetypes": chain_archetypes,
    }


def _compute_chain_archetypes(
    seed_questions: list[dict[str, Any]],
    classifications: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Group the 300 problems into chain archetypes keyed by the source_class
    of each problem's FINAL expected_chain URL.

    Each archetype is the natural unit for seed_expansion: "find more chains
    whose answer lives on a page of this kind, with the same retrieval
    pathology." This sidesteps the per-URL singleton explosion (242 singletons
    in the URL-atom view collapse into ~33% 1-chain archetypes here).

    Each chain entry carries everything seed_expansion + problem_generation
    need for provenance — question_id, problem text, answer, the full URL
    chain, per-model failure modes — so downstream joins back to
    seed_url_index.json / seed_questions.json become unnecessary.
    """
    archetypes: dict[str, dict[str, Any]] = {}

    for q_idx, q in enumerate(seed_questions):
        question_id = f"seed_{q_idx:04d}"
        problem = norm_text(q.get("problem", ""))
        answer = q.get("answer", "")
        category = q.get("category", "")

        per_model_fm: dict[str, str | None] = {}
        for model, idx in classifications.items():
            cls = idx.get(problem)
            per_model_fm[model] = (cls or {}).get("primary_failure_mode") or None

        chain_urls: list[str] = []
        for step in q.get("expected_chain", []) or []:
            for url in step.get("sources", []) or []:
                chain_urls.append(url)
        if not chain_urls:
            continue  # no chain → can't bucket

        chain_classes = [classify_url(u) for u in chain_urls]
        chain_hosts = []
        for u in chain_urls:
            try:
                chain_hosts.append(urllib.parse.urlparse(u).netloc.lower())
            except Exception:
                chain_hosts.append("")
        final_url = chain_urls[-1]
        final_url_class = chain_classes[-1]

        target_fms_union = sorted({fm for fm in per_model_fm.values() if fm})

        chain_entry = {
            "question_id": question_id,
            "problem": problem,
            "answer": answer,
            "category": category,
            "chain_length": len(chain_urls),
            "chain_classes": chain_classes,
            "chain_urls": chain_urls,
            "chain_hosts": chain_hosts,
            "final_url": final_url,
            "final_url_class": final_url_class,
            "target_failure_modes_by_model": per_model_fm,
            "target_failure_modes": target_fms_union,
        }

        arch = archetypes.get(final_url_class)
        if arch is None:
            arch = {
                "name": final_url_class,
                "chain_count": 0,
                "categories": Counter(),
                "hosts": Counter(),
                "failure_mode_distribution": Counter(),
                "failure_mode_by_model": {m: Counter() for m in classifications},
                "chains": [],
            }
            archetypes[final_url_class] = arch
        arch["chain_count"] += 1
        arch["chains"].append(chain_entry)
        arch["categories"][category] += 1
        for h in chain_hosts:
            if h:
                arch["hosts"][h] += 1
        for fm in target_fms_union:
            arch["failure_mode_distribution"][fm] += 1
        for model, fm in per_model_fm.items():
            if fm:
                arch["failure_mode_by_model"][model][fm] += 1

    return archetypes


def _counter_keys_to_str(c: Counter) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in c.most_common():
        if isinstance(k, tuple):
            out["+".join(k)] = v
        else:
            out[str(k)] = v
    return out


def write_outputs(mined: dict[str, Any], exemplars: dict[str, list[dict[str, Any]]]) -> None:
    SEED_MATERIAL.mkdir(parents=True, exist_ok=True)

    seed_bank: dict[str, Any] = {
        "schema_version": 1,
        "description": (
            "Source-class taxonomy for K-BrowseComp seed pages. Each class lists hand-coded "
            "retrieval-pathology priors (for human review) and empirical failure-mode "
            "propensities computed by joining the 300 hand-curated questions with two "
            "frontier-model failure-mode classifications (Gemini-3.1-flash-lite-preview, "
            "GPT-5.4-mini)."
        ),
        "failure_mode_names": FAILURE_MODE_NAMES,
        "seed_question_count": mined["seed_question_count"],
        "total_url_count": sum(c.url_count for c in mined["classes"].values()),
        "classes": [],
    }
    # Replace tuple keys in failure_mode_joint with plus-joined strings before dumping.
    for cs in sorted(mined["classes"].values(), key=lambda c: c.url_count, reverse=True):
        d = cs.to_dict()
        d["failure_mode_joint"] = _counter_keys_to_str(cs.failure_mode_joint)
        seed_bank["classes"].append(d)

    # Chain-level view: 300 chains bucketed by final-URL's source_class.
    # This is the natural iteration unit for seed_expansion — see
    # _compute_chain_archetypes() docstring.
    seed_bank["chain_archetypes"] = []
    for arch in sorted(
        mined["chain_archetypes"].values(), key=lambda a: a["chain_count"], reverse=True
    ):
        arch_out = {
            "name": arch["name"],
            "chain_count": arch["chain_count"],
            "categories": _counter_keys_to_str(arch["categories"]),
            "failure_mode_distribution": _counter_keys_to_str(arch["failure_mode_distribution"]),
            "failure_mode_by_model": {
                m: _counter_keys_to_str(c)
                for m, c in arch["failure_mode_by_model"].items()
            },
            "hosts": _counter_keys_to_str(arch["hosts"]),
            "chains": arch["chains"],
        }
        seed_bank["chain_archetypes"].append(arch_out)
    seed_bank["chain_count"] = sum(a["chain_count"] for a in seed_bank["chain_archetypes"])

    SEED_BANK_OUT.write_text(
        json.dumps(seed_bank, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    SEED_URL_INDEX_OUT.write_text(
        json.dumps(mined["flat_index"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    FAILURE_MODE_EXEMPLARS_OUT.write_text(
        json.dumps(
            {
                "failure_mode_names": FAILURE_MODE_NAMES,
                "exemplars": exemplars,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    write_report(mined, exemplars)


def write_report(mined: dict[str, Any], exemplars: dict[str, list[dict[str, Any]]]) -> None:
    lines: list[str] = []
    classes = sorted(mined["classes"].values(), key=lambda c: c.url_count, reverse=True)
    total_urls = sum(c.url_count for c in classes)

    lines.append("# K-BrowseComp Seed Material Summary")
    lines.append("")
    lines.append(f"- Seed questions: {mined['seed_question_count']}")
    lines.append(f"- Total source URLs: {total_urls}")
    lines.append(f"- Source classes detected: {len(classes)}")
    classified_total = sum(c.url_count for c in classes if not c.name.startswith("unclassified:"))
    lines.append(
        f"- Classified URLs: {classified_total}/{total_urls} "
        f"({(classified_total / total_urls * 100):.1f}% if denom > 0)"
    )
    unclassified_total = sum(c.url_count for c in classes if c.name.startswith("unclassified:"))
    lines.append(f"- Unclassified URLs (long tail): {unclassified_total}")
    lines.append("")
    lines.append("## Classes by URL count")
    lines.append("")
    lines.append("| Class | URLs | Q's | Top categories | Top failure modes (joint) | Hypothesis |")
    lines.append("|---|---:|---:|---|---|---|")
    for c in classes:
        if c.name.startswith("unclassified:"):
            continue
        top_cats = ", ".join(f"{k} ({v})" for k, v in c.categories.most_common(3))
        # Joint failure modes already aggregate by single tag too; show top single tags.
        top_fms = ", ".join(
            f"{k} ({v})"
            for k, v in c.failure_mode_joint.most_common(20)
            if not isinstance(k, tuple)
        )[:120]
        lines.append(
            f"| `{c.name}` | {c.url_count} | {len(c.question_ids)} | {top_cats} | {top_fms} | {c.retrieval_hypothesis} |"
        )
    lines.append("")

    lines.append("## Unclassified long-tail hosts (top 30)")
    lines.append("")
    lines.append("These hosts fell through the rule table. Add rules for any that look load-bearing.")
    lines.append("")
    top_unclassified = mined["unclassified_hosts"].most_common(30)
    if top_unclassified:
        lines.append("| Host | URLs |")
        lines.append("|---|---:|")
        for h, n in top_unclassified:
            lines.append(f"| {h} | {n} |")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Failure-mode exemplars (counts only; full digests in failure_mode_exemplars.json)")
    lines.append("")
    for fm, examples in exemplars.items():
        models = Counter(e["solver_model"] for e in examples)
        lines.append(f"- **{fm}** {FAILURE_MODE_NAMES[fm]} — {len(examples)} exemplars ({dict(models)})")
    lines.append("")

    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mined = mine()
    exemplars = curate_exemplars(load_classifications())
    write_outputs(mined, exemplars)
    print("Wrote:")
    for p in (SEED_BANK_OUT, SEED_URL_INDEX_OUT, FAILURE_MODE_EXEMPLARS_OUT, SUMMARY_OUT):
        if p.exists():
            print(f"  {p.relative_to(ROOT)}: {p.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
