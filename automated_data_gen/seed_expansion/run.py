"""
seed_expansion/run.py — Agent-driven seed-bank expansion.

For each source class in seed_material/seed_bank.json (or a filtered subset),
spawn a headless Claude Code subprocess with internet access. Each agent reads
a class-specific brief (task_spec.md), uses WebSearch + WebFetch to discover
intra-domain (more URLs from the same hosts) and inter-domain (sibling hosts
with the same structural archetype) candidate URLs, and writes structured JSON
into its workspace. After all per-class runs complete, the orchestrator
aggregates the per-class outputs into:

  - seed_expansion/intra_domain_candidates.json
  - seed_expansion/inter_domain_candidates.json

Usage:
  python3 run.py                            # run all classes (sequential)
  python3 run.py --classes species_inventory,encyclopedia_korean_culture
  python3 run.py --skip-classes namu_wiki_main,news_legacy
  python3 run.py --target-intra 20 --target-inter 20
  python3 run.py --parallel 4               # up to 4 agents concurrent
  python3 run.py --aggregate-only           # skip running, just aggregate work/
  python3 run.py --dry-run                  # build briefs but don't spawn claude
"""

import argparse
import concurrent.futures as futures
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 7200))  # 2 hours per class
REQUIRED_CLAUDE_VERSION = os.environ.get("CLAUDE_VERSION", "2.1.138")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
EFFORT = os.environ.get("CLAUDE_EFFORT", "max")

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SEED_MATERIAL = PARENT / "seed_material"
SEED_BANK = SEED_MATERIAL / "seed_bank.json"
WORK_DIR = ROOT / "work"
TEMPLATE = ROOT / "task_spec_template.md"

OUT_INTRA = ROOT / "intra_domain_candidates.json"
OUT_INTER = ROOT / "inter_domain_candidates.json"

# Allow `--dangerously-skip-permissions` inside containers running as root.
os.environ.setdefault("IS_SANDBOX", "1")


# ---------------------------------------------------------------------------
# Environment / preflight.
# ---------------------------------------------------------------------------


def check_auth() -> None:
    """Soft-check Claude Code auth.

    On macOS, `claude login` stores OAuth credentials in the system Keychain
    rather than `~/.claude/.credentials.json`, so the file check is unreliable.
    We warn rather than exit and let the subprocess surface real auth errors.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if (Path.home() / ".claude" / ".credentials.json").exists():
        return
    if sys.platform == "darwin" and (Path.home() / ".claude").exists():
        print(
            "[run] Warning: no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json, "
            "but ~/.claude/ exists. Assuming macOS Keychain OAuth.",
            flush=True,
        )
        return
    sys.exit(
        "[run] Claude Code is not authenticated. Either:\n"
        "  - set ANTHROPIC_API_KEY=<your-api-key>, or\n"
        "  - run `claude login` once to authenticate via OAuth."
    )


def check_claude_version() -> None:
    result = subprocess.run(
        [CLAUDE_BIN, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"[run] `{CLAUDE_BIN} --version` failed: {result.stderr.strip()}")
    match = re.search(r"\d+\.\d+\.\d+", result.stdout)
    if not match:
        sys.exit(f"[run] Could not parse claude version from: {result.stdout.strip()!r}")
    version = match.group(0)
    if version != REQUIRED_CLAUDE_VERSION:
        print(
            f"[run] Warning: claude {version} found, expected {REQUIRED_CLAUDE_VERSION}. "
            "Set CLAUDE_VERSION env var to silence.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Subprocess plumbing (lifted from the original runner).
# ---------------------------------------------------------------------------


def _kill_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _short(s: str, n: int = 200) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."


def _tool_summary(name: str, inp: dict[str, Any]) -> str:
    if name == "Bash":
        return _short(inp.get("command", ""))
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return inp.get("file_path", "?")
    if name in ("Grep", "Glob"):
        return f"pattern={inp.get('pattern', '?')}"
    if name in ("Task", "Agent"):
        return inp.get("description", "?")
    if name == "WebFetch":
        return inp.get("url", "?")
    if name == "WebSearch":
        return inp.get("query", "?")
    return _short(json.dumps(inp))


def _format_event(evt: dict[str, Any], prefix: str = "") -> str | None:
    t = evt.get("type")
    if t == "system":
        sub = evt.get("subtype", "system")
        if sub == "init":
            return f"{prefix}[init] session={evt.get('session_id', '?')[:8]}"
        return f"{prefix}[{sub}]"
    if t == "assistant":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                txt = block.get("text", "").strip()
                if txt:
                    parts.append(f"{prefix}{txt}")
            elif bt == "tool_use":
                name = block.get("name", "?")
                parts.append(f"{prefix}→ {name}: {_tool_summary(name, block.get('input', {}))}")
        return "\n".join(parts) if parts else None
    if t == "user":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                marker = "✗" if block.get("is_error") else "✓"
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                parts.append(f"{prefix}  {marker} {_short(str(content).strip())}")
        return "\n".join(parts) if parts else None
    if t == "result":
        success = evt.get("subtype") == "success"
        cost = evt.get("total_cost_usd")
        cost_str = f" cost=${cost:.4f}" if cost else ""
        return f"{prefix}[result] success={success}{cost_str}"
    return None


def run_claude(args: list[str], cwd: Path, label: str, timeout: int = CLAUDE_TIMEOUT) -> int:
    """Run a single claude subprocess, pretty-printing stream-json events live."""
    prefix = f"[{label}] " if label else ""
    proc = subprocess.Popen(
        [CLAUDE_BIN] + args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        _kill_process_group(proc.pid)

    timer = threading.Timer(timeout, _on_timeout)
    timer.start()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            try:
                pretty = _format_event(json.loads(line), prefix=prefix)
            except json.JSONDecodeError:
                pretty = prefix + line
            if pretty:
                print(pretty, flush=True)
        proc.wait()
        if timed_out.is_set():
            print(f"{prefix}[run] Timeout after {timeout}s; killed process group.", flush=True)
        return proc.returncode
    finally:
        timer.cancel()
        _kill_process_group(proc.pid)


# ---------------------------------------------------------------------------
# Class workspace setup.
# ---------------------------------------------------------------------------


FAILURE_MODE_NAMES = {
    "F1": "첫 검색어를 좁힐 수 없음 / 사후 검증형 문제 — starting-entity ambiguity",
    "F3": "비인접 도메인 hopping — cross-domain chain retrieval",
    "F4": "semi-structured parsing 실패 — extracting from tables / notice fields / sub-sections",
    "F6": "희소 엔티티 정규화 실패 — rare-entity normalization (Korean variant names, transliteration)",
    "F7": "조건 누적 / constraint tracking 실패 — accumulating multiple constraints",
    "F8": "중간 계산 / 절차형 reasoning 실패 — intermediate computation / multi-step arithmetic",
    "F9": "검색 결과 선택 실패 — choosing the right SERP result among many",
    "F10": "iframe / 동적 페이지 / 특정 페이지 진입 실패 — dynamic / login-walled / iframe content",
}


def load_seed_bank() -> dict[str, Any]:
    return json.loads(SEED_BANK.read_text(encoding="utf-8"))


def _build_url_class_lookup(sb: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map URL-class name → URL-class entry, so chain archetypes can inherit
    `retrieval_hypothesis` + `prior_failure_modes` (hand-coded in the RULES
    table) by matching archetype name to URL-class name."""
    return {c["name"]: c for c in sb.get("classes", [])}


def _archetype_to_klass(arch: dict[str, Any], url_classes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert a chain archetype into the `klass`-shaped dict that render_brief
    expects. Each archetype name matches a URL-class name (since the archetype
    key IS the final URL's source_class), so we can inherit the named-rule
    priors when available. New chain-specific data is preserved under
    additional fields for richer brief context."""
    matching_url_class = url_classes.get(arch["name"], {})
    # Collect a sample of distinct final URLs across the archetype's chains
    # to serve as the legacy `sample_urls` block.
    sample_final_urls: list[str] = []
    seen: set[str] = set()
    for ch in arch.get("chains", []):
        u = ch.get("final_url")
        if u and u not in seen:
            seen.add(u)
            sample_final_urls.append(u)
        if len(sample_final_urls) >= 16:
            break
    return {
        "name": arch["name"],
        "url_count": arch["chain_count"],          # repurposed: "size of input set"
        "distinct_question_count": arch["chain_count"],
        "hosts": arch.get("hosts", {}),
        "categories": arch.get("categories", {}),
        "failure_mode_joint": arch.get("failure_mode_distribution", {}),
        "failure_mode_by_model": arch.get("failure_mode_by_model", {}),
        "sample_urls": sample_final_urls,
        # Inherited hand-coded priors (may be absent for archetypes whose name
        # is a singleton-host bucket like "unclassified:example.com").
        "retrieval_hypothesis": matching_url_class.get("retrieval_hypothesis", ""),
        "prior_failure_modes": matching_url_class.get("prior_failure_modes", []),
        # Chain-level data for downstream provenance / brief enrichment.
        "chains": arch.get("chains", []),
        "chain_count": arch["chain_count"],
    }


def select_classes(
    sb: dict[str, Any],
    include: list[str] | None,
    skip: list[str] | None,
    min_urls: int,
) -> list[dict[str, Any]]:
    """Pick chain archetypes (final-URL-class buckets) to run seed_expansion on.

    The `min_urls` arg name is kept for CLI compatibility but it now applies
    to `chain_count`. Archetype unclassified-host buckets are skipped — they
    came from a single URL with no named rule, providing no archetype signal.
    """
    url_classes = _build_url_class_lookup(sb)
    archetypes_src = sb.get("chain_archetypes") or []
    classes = []
    for arch in archetypes_src:
        name = arch["name"]
        if name.startswith("unclassified:") or name == "unparseable":
            continue
        if include and name not in include:
            continue
        if skip and name in skip:
            continue
        if arch["chain_count"] < min_urls:
            continue
        classes.append(_archetype_to_klass(arch, url_classes))
    return classes


def _compute_fm_quotas(
    klass: dict[str, Any],
    total: int,
    k: int = 3,
) -> list[tuple[str, int]]:
    """Allocate `total` candidates across the top-K *empirical* failure modes
    for this archetype, with prior-FM fallback.

    Empirical FMs come from joining the seed_bank with the two solver-model
    classifications — what FMs actually fired on this archetype's chains.
    Priors are the hand-coded hypothesis (kept as backup for archetypes whose
    chain pool is too small to have meaningful empirical signal).

    Splits 20 across 3 FMs as 7+7+6 (and 20 across 2 as 10+10, 20 across 1
    as 20). Returns list of (fm, count) tuples; counts always sum to `total`.
    """
    # failure_mode_joint here holds single-tag counts under str keys (we
    # filter out the joint pairs like "F4+F7" which are tuple-joined strings).
    empirical = klass.get("failure_mode_joint", {}) or {}
    single = {k: v for k, v in empirical.items() if "+" not in k}
    if single:
        ranked = sorted(single.items(), key=lambda x: -x[1])
        chosen = [fm for fm, _ in ranked[:k]]
    else:
        priors = klass.get("prior_failure_modes", []) or []
        chosen = priors[:k] if priors else ["F4"]  # ultimate fallback
    n = len(chosen)
    base = total // n
    rem = total % n
    return [(fm, base + (1 if i < rem else 0)) for i, fm in enumerate(chosen)]


def _final_url_hosts(klass: dict[str, Any]) -> dict[str, int]:
    """Count distinct hosts among the FINAL URLs of chains in this archetype.

    The archetype's `hosts` field aggregates hosts across ALL chain URLs
    including intermediate hops, which over-counts. For the intra-rule
    decision we only care about where the answers live — the final URL.
    """
    import urllib.parse
    from collections import Counter
    counts: Counter[str] = Counter()
    for ch in klass.get("chains", []):
        try:
            h = urllib.parse.urlparse(ch.get("final_url", "")).netloc.lower()
        except Exception:
            continue
        if h:
            counts[h] += 1
    return dict(counts)


def _render_intra_host_rule(klass: dict[str, Any]) -> str:
    """Return the intra-axis instructions appropriate for this archetype's
    final-URL host distribution.

    - Multi-host archetypes (news_legacy, government_notice, academic_korean,
      etc.): keep the original 'find more hosts of the same kind' rule.
    - Single-host archetypes (namu_wiki_main, ko.wikipedia.org, youtube_video,
      etc., 18 of 30): redefine intra as 'deeper coverage of the same host
      with new URLs not yet in the seed bank,' since there are no good
      cross-host siblings to find.
    """
    final_hosts = _final_url_hosts(klass)
    if len(final_hosts) <= 1:
        host = next(iter(final_hosts), "(no host detected)")
        return (
            f"This archetype is **single-host**: every chain in the seed bank "
            f"ends on `{host}`. For intra entries, find more pages on "
            f"**`{host}` itself** that are NOT already in the seed bank "
            f"(see `class_context.json` `sample_urls` and the eight reference "
            f"URLs above). Use the host's category index, sitemap, in-site "
            f"search, or list pages to discover unseen URL paths.\n\n"
            f"Each candidate's `website_url`:\n"
            f"- MUST be on `{host}` (or a near-identical subdomain — "
            f"e.g. `m.{host}` if it serves the same content)\n"
            f"- MUST be a fresh URL not in any seed-bank chain\n"
            f"- MUST exhibit the same retrieval pathology (same top "
            f"empirical FMs as the seed class)\n"
            f"- MUST satisfy the Korean-relevance, structural-density, and "
            f"snippet-poverty criteria above\n\n"
            f"Goal: deeper coverage of `{host}` so Stage 3 has a wide pool "
            f"of `{host}` pages to draft problems against. Cross-host "
            f"expansion happens via the inter axis below."
        )
    sample_hosts = sorted(final_hosts, key=lambda h: -final_hosts[h])[:5]
    return (
        f"This archetype is **multi-host**: chains in the seed bank end on "
        f"{len(final_hosts)} distinct hosts (top: {', '.join(f'`{h}`' for h in sample_hosts)}…). "
        f"For intra entries, find Korean websites that serve **the same "
        f"kind of content** as the seeds, on **hosts NOT in this class's "
        f"seed bank** (see `class_context.json` `hosts`). Same retrieval "
        f"pathology, same archetype-type, new host.\n\n"
        f"Strategies:\n"
        f"- Use `WebSearch` for Korean queries naming the archetype + "
        f"alternative providers.\n"
        f"- For each candidate host, fetch the root or sitemap first to "
        f"confirm the host exists and is structurally similar.\n"
        f"- Then enumerate detail pages within each host (3-6 per host is "
        f"a good cadence — see diversity rules below)."
    )


def _render_fm_quota_block(
    intra_quotas: list[tuple[str, int]],
    inter_quotas: list[tuple[str, int]],
) -> str:
    intra_map = dict(intra_quotas)
    inter_map = dict(inter_quotas)
    all_fms = sorted({fm for fm, _ in intra_quotas + inter_quotas})
    lines = [
        "| Failure mode | Intra target | Inter target | Description |",
        "|---|---:|---:|---|",
    ]
    for fm in all_fms:
        lines.append(
            f"| **{fm}** | {intra_map.get(fm, 0)} | {inter_map.get(fm, 0)} | "
            f"{FAILURE_MODE_NAMES.get(fm, '')} |"
        )
    return "\n".join(lines)


def render_brief(klass: dict[str, Any], target_intra: int, target_inter: int) -> tuple[str, dict[str, Any]]:
    """Return (task_spec.md content, class_context.json dict) for one class."""
    template = TEMPLATE.read_text(encoding="utf-8")
    intra_quotas = _compute_fm_quotas(klass, target_intra)
    inter_quotas = _compute_fm_quotas(klass, target_inter)
    fm_quota_block = _render_fm_quota_block(intra_quotas, inter_quotas)
    intra_host_rule = _render_intra_host_rule(klass)

    top_hosts = list(klass.get("hosts", {}).items())[:8]
    top_categories = list(klass.get("categories", {}).items())[:5]

    # Empirical failure-mode distribution: single-tag counts only (not joint pairs).
    fm_single = {
        k: v
        for k, v in klass.get("failure_mode_joint", {}).items()
        if "+" not in k
    }
    top_fms = sorted(fm_single.items(), key=lambda kv: kv[1], reverse=True)[:5]

    fm_blocks = []
    for fm_code, _ct in top_fms:
        fm_blocks.append(f"- **{fm_code}** — {FAILURE_MODE_NAMES.get(fm_code, '')}")
    fm_descriptions = "\n".join(fm_blocks) if fm_blocks else "(no failure-mode signal in seed data)"

    hosts_block = "\n".join(f"- `{h}` ({n} URLs)" for h, n in top_hosts) or "(none)"
    cats_block = "\n".join(f"- {c} ({n})" for c, n in top_categories) or "(none)"
    fm_hist_block = ", ".join(f"{k}={v}" for k, v in top_fms) or "(none)"
    # Sample of representative chains (preferred) — each line annotates the
    # final URL with its source question_id and target failure modes, so the
    # agent sees both the URL pattern AND the retrieval pathology to match.
    chains = klass.get("chains", [])[:8]
    if chains:
        samples_block_lines = []
        for i, ch in enumerate(chains):
            fms = ",".join(ch.get("target_failure_modes", [])) or "(no FM)"
            samples_block_lines.append(
                f"{i + 1}. {ch.get('final_url', '')}  "
                f"[{ch.get('question_id', '?')}; target_FM={fms}; "
                f"chain_len={ch.get('chain_length', '?')}]"
            )
        samples_block = "\n".join(samples_block_lines)
    else:
        samples = klass.get("sample_urls", [])[:8]
        samples_block = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(samples)) or "(none)"

    rendered = template.format(
        class_name=klass["name"],
        retrieval_hypothesis=klass.get("retrieval_hypothesis", ""),
        prior_failure_modes=", ".join(klass.get("prior_failure_modes", [])) or "(none)",
        url_count=klass["url_count"],
        distinct_question_count=klass["distinct_question_count"],
        hosts_block=hosts_block,
        categories_block=cats_block,
        fm_descriptions=fm_descriptions,
        fm_histogram=fm_hist_block,
        fm_quota_block=fm_quota_block,
        intra_host_rule=intra_host_rule,
        samples_block=samples_block,
        target_intra=target_intra,
        target_inter=target_inter,
    )

    ctx = {
        "class_name": klass["name"],
        "retrieval_hypothesis": klass.get("retrieval_hypothesis", ""),
        "prior_failure_modes": klass.get("prior_failure_modes", []),
        "url_count": klass["url_count"],
        "distinct_question_count": klass["distinct_question_count"],
        "hosts": klass.get("hosts", {}),
        "categories": klass.get("categories", {}),
        "failure_mode_joint": klass.get("failure_mode_joint", {}),
        "failure_mode_by_model": klass.get("failure_mode_by_model", {}),
        "sample_urls": klass.get("sample_urls", []),
        "failure_mode_names": FAILURE_MODE_NAMES,
        "target_intra": target_intra,
        "target_inter": target_inter,
        "fm_quotas_intra": dict(intra_quotas),
        "fm_quotas_inter": dict(inter_quotas),
    }
    return rendered, ctx


def setup_class_workspace(klass: dict[str, Any], run_id: str,
                          target_intra: int, target_inter: int) -> Path:
    ws = WORK_DIR / klass["name"] / run_id
    ws.mkdir(parents=True, exist_ok=True)

    rendered, ctx = render_brief(klass, target_intra, target_inter)
    (ws / "task_spec.md").write_text(rendered, encoding="utf-8")
    (ws / "class_context.json").write_text(
        json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ws


# ---------------------------------------------------------------------------
# Per-class runner.
# ---------------------------------------------------------------------------


def class_has_prior_run(class_name: str) -> bool:
    """True if any prior run for this class produced valid intra.json + inter.json."""
    class_dir = WORK_DIR / class_name
    if not class_dir.is_dir():
        return False
    for run_dir in class_dir.iterdir():
        if not run_dir.is_dir():
            continue
        intra = run_dir / "intra.json"
        inter = run_dir / "inter.json"
        if not (intra.exists() and inter.exists()):
            continue
        try:
            json.loads(intra.read_text(encoding="utf-8"))
            json.loads(inter.read_text(encoding="utf-8"))
            return True
        except Exception:
            continue
    return False


def run_one_class(klass: dict[str, Any], run_id: str, target_intra: int, target_inter: int,
                  dry_run: bool, resume: bool) -> tuple[str, int]:
    label = klass["name"]

    if resume and class_has_prior_run(label):
        print(f"[{label}] skip (a prior run already produced valid intra+inter)", flush=True)
        return label, 0

    ws = setup_class_workspace(klass, run_id, target_intra, target_inter)

    if dry_run:
        print(f"[{label}] dry-run: brief written to {ws / 'task_spec.md'}", flush=True)
        return label, 0

    session_id = str(uuid.uuid4())
    claude_args = [
        "-p",
        "Read task_spec.md and complete the task it describes. "
        "Write final outputs to ./intra.json and ./inter.json in this directory.",
        "--output-format", "stream-json",
        "--verbose",
        "--model", MODEL,
        "--effort", EFFORT,
        "--session-id", session_id,
        "--dangerously-skip-permissions",
    ]
    print(f"[{label}] start run_id={run_id} session={session_id[:8]}", flush=True)
    rc = run_claude(claude_args, cwd=ws, label=label)
    print(f"[{label}] done rc={rc}", flush=True)
    return label, rc


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _load_json_safe(p: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def aggregate() -> tuple[int, int]:
    """Walk work/<class>/<run_id>/{intra,inter}.json. Merge across all run_ids per
    class, deduplicating by website_url (later run_ids win on conflict). Each
    output entry is annotated with `source_class` and `run_id` for traceability.

    Preserves entries already present in the top-level intra/inter JSONs from
    prior runs whose work/ has since been cleaned. Without this, running a
    new class with an empty work/ would silently wipe earlier classes' results.
    Conflict policy:
      existing entry < fresh work/ entry (new data always wins on URL match)
    """
    intra_by_url: dict[str, dict[str, Any]] = {}
    inter_by_url: dict[str, dict[str, Any]] = {}

    # Seed from any existing top-level aggregates so we never silently shrink.
    for path, sink in ((OUT_INTRA, intra_by_url), (OUT_INTER, inter_by_url)):
        if not path.exists():
            continue
        for e in _load_json_safe(path):
            url = e.get("website_url")
            if isinstance(url, str) and url:
                sink[url] = e
    seeded_intra, seeded_inter = len(intra_by_url), len(inter_by_url)

    if WORK_DIR.exists():
        for class_dir in sorted(WORK_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            cn = class_dir.name
            run_dirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
            for run_dir in run_dirs:
                rid = run_dir.name
                for kind, sink in (("intra", intra_by_url), ("inter", inter_by_url)):
                    p = run_dir / f"{kind}.json"
                    if not p.exists():
                        continue
                    for e in _load_json_safe(p):
                        url = e.get("website_url")
                        if not isinstance(url, str) or not url:
                            continue
                        e.setdefault("source_class", cn)
                        e["run_id"] = rid
                        sink[url] = e  # later run_ids overwrite earlier ones
    else:
        print("[aggregate] no work/ directory; preserving existing top-level entries", flush=True)

    intra_all = list(intra_by_url.values())
    inter_all = list(inter_by_url.values())
    OUT_INTRA.write_text(json.dumps(intra_all, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_INTER.write_text(json.dumps(inter_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[aggregate] wrote {OUT_INTRA.name} ({len(intra_all)} unique entries; "
        f"{seeded_intra} preserved + {len(intra_all) - seeded_intra} new/updated) and "
        f"{OUT_INTER.name} ({len(inter_all)} unique entries; "
        f"{seeded_inter} preserved + {len(inter_all) - seeded_inter} new/updated)",
        flush=True,
    )
    return len(intra_all), len(inter_all)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run agent-driven seed-bank expansion across source classes."
    )
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names to include (default: all).")
    ap.add_argument("--skip-classes", default=None,
                    help="Comma-separated class names to exclude.")
    ap.add_argument("--min-urls", type=int, default=0,
                    help=("Skip archetypes with fewer than this many chains "
                          "(arg name kept for CLI back-compat; now applied to chain_count)."))
    ap.add_argument("--target-intra", type=int, default=15,
                    help="Per-class target for intra-domain entries.")
    ap.add_argument("--target-inter", type=int, default=15,
                    help="Per-class target for inter-domain entries.")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Max concurrent class subprocesses. Default 1.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip classes that already have at least one valid prior run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Render briefs to work/<class>/<run_id>/task_spec.md but do not launch claude.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip all subprocesses, just aggregate existing work/ outputs.")
    ap.add_argument("--run-id", default=None,
                    help="Identifier for this run. Defaults to current UTC timestamp. "
                         "Each (class, run_id) gets its own isolated subdir; pass the "
                         "same value across multiple invocations to share a directory.")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    if not args.dry_run:
        check_claude_version()
        check_auth()

    if not SEED_BANK.exists():
        sys.exit(f"[run] seed_bank.json not found at {SEED_BANK}. Run mine_seed_bank.py first.")
    if not TEMPLATE.exists():
        sys.exit(f"[run] task_spec_template.md not found at {TEMPLATE}.")

    sb = load_seed_bank()
    classes = select_classes(
        sb,
        include=parse_csv(args.classes),
        skip=parse_csv(args.skip_classes),
        min_urls=args.min_urls,
    )
    if not classes:
        sys.exit("[run] No classes matched the filters.")

    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[run] Claude:    {CLAUDE_BIN} ({REQUIRED_CLAUDE_VERSION} expected)", flush=True)
    print(f"[run] Model:     {MODEL} (effort={EFFORT})", flush=True)
    print(f"[run] Run ID:    {run_id}", flush=True)
    print(f"[run] Classes:   {len(classes)} ({', '.join(c['name'] for c in classes[:6])}"
          + (", ..." if len(classes) > 6 else "")
          + ")", flush=True)
    print(f"[run] Targets:   intra={args.target_intra} inter={args.target_inter} per class", flush=True)
    print(f"[run] Parallel:  {args.parallel}", flush=True)
    print(f"[run] Workspace: {WORK_DIR} (subdir per <class>/<run_id>)", flush=True)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, int]] = []
    if args.parallel <= 1:
        for k in classes:
            results.append(run_one_class(k, run_id, args.target_intra, args.target_inter,
                                         args.dry_run, args.resume))
    else:
        with futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [
                ex.submit(run_one_class, k, run_id, args.target_intra, args.target_inter,
                          args.dry_run, args.resume)
                for k in classes
            ]
            for f in futures.as_completed(futs):
                results.append(f.result())

    print("\n[run] Per-class results:", flush=True)
    for label, rc in sorted(results):
        print(f"  {label}: rc={rc}", flush=True)

    if not args.dry_run:
        aggregate()


if __name__ == "__main__":
    main()
