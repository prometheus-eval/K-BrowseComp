# Seed-bank expansion — source class: `{class_name}`

You are expanding the seed bank for **K-BrowseComp**, a Korean web-browsing
benchmark in the style of OpenAI's BrowseComp. Your job is to discover more
candidate URLs that share the **retrieval pathology** of this source class —
not just URLs that *look* similar on the surface.

You have **internet access** via `WebSearch` and `WebFetch`. Use them aggressively.

A deterministic post-hoc validator will re-check every URL you submit. If you
report `http_status: 200` for a URL that returns 404, or you submit a snippet
that isn't actually present on the page, that entry will be silently dropped
and the run will be flagged. There is no upside to inflating counts with
unverified entries — quality is checked, not trusted.

---

## Class context

- **Retrieval hypothesis:** {retrieval_hypothesis}
- **Prior failure-mode hypotheses (hand-coded):** {prior_failure_modes}
- **Seed bank size for this class:** {url_count} URLs across
  {distinct_question_count} hand-curated K-BrowseComp questions.
- **Empirical failure-mode histogram (single-tag counts):** {fm_histogram}

### Hosts already in the seed bank (with counts)
{hosts_block}

### Question categories where this class appears
{categories_block}

### Top failure modes observed empirically for this class
{fm_descriptions}

### Per-FM quotas for THIS run (you must honor these)

You will produce **{target_intra}** intra entries and **{target_inter}**
inter entries. They must be distributed across failure modes as shown.
Each candidate's `target_failure_modes` field must include the FM bucket
it's filling (you may add a second FM if the page genuinely exhibits two,
but the primary slot must match the table).

{fm_quota_block}

### Eight reference seed URLs (read these to ground your sense of the class)
{samples_block}

---

## What you are looking for

A "good" candidate URL for this class has these structural properties:

1. **Structural density.** The page carries the load-bearing information
   inside structured elements — HTML tables, `<dl>` lists, multi-row notice
   fields, sub-sections, year-by-year stat tables, taxonomy rows, tracklists,
   cast rotations, 별표/별첨, etc. The answer to a hypothetical K-BrowseComp
   question lives in one cell or row, not in a paragraph.

2. **Snippet poverty.** The identifying detail is **not** visible in a
   standard Google/Naver SERP snippet for the page. The solver has to actually
   land on the page and inspect it.

3. **Title non-leakage.** The page title does not give away the answer or the
   most identifying detail.

4. **Korean relevance — REQUIRED, not optional.** K-BrowseComp asks
   Korean-language questions about Korean entities, culture, institutions,
   companies, history, regional content, government, etc. Every candidate
   URL must satisfy **both**:

   (a) **The page itself** carries Korean-relevant content. Korean-language
       text is sufficient. English- or other-language pages count only if
       the *subject* is a Korean entity (a Korean person, place, company,
       cultural item, institution, scientific specimen from Korea, etc.).

   (b) **The access route** is a question a Korean speaker would ask. The
       indirect descriptors a solver would chain through must read as a
       Korean-context question — e.g., "한국의 ___ 중에서 ___ 인 것은?",
       "국립___관 데이터베이스에서 ___ 의 ___ 는?", "한국 영화 ___ 의
       ___ 배우는?". If the only natural Korean-language question that
       reaches this page would feel forced or unnatural, **drop it**.

   Concrete examples:
   - ✅ `https://ko.wikipedia.org/wiki/박찬욱` — Korean Wikipedia, Korean
     director. Page is Korean; route is Korean.
   - ✅ `https://en.wikipedia.org/wiki/Park_Chan-wook` — English Wikipedia
     but subject is a Korean director. Page is English (4a satisfied via
     "subject is Korean"); a Korean question about 박찬욱 naturally reaches
     this page (4b satisfied).
   - ✅ `https://arxiv.org/abs/2310.12345` — arXiv paper, but authors are
     Korean researchers from KAIST. (Both 4a and 4b satisfied.)
   - ❌ `https://en.wikipedia.org/wiki/Eiffel_Tower` — English page about
     a French monument. Fails 4a (subject not Korean) AND 4b (no natural
     Korean-context question reaches this).
   - ❌ `https://www.bbc.com/news/uk-politics-12345` — UK domestic politics
     article. Fails both.
   - ❌ `https://github.com/some-american-startup/repo` — code repo with
     no Korean connection. Fails both.

5. **Reachable as ordinary web text.** Not an embedded PDF/image/video, not
   behind login walls (unless the class explicitly hosts such pages, e.g.,
   `instagram` or `dbpia_academic`).

---

## Two discovery axes

Both axes require **the same retrieval pathology** as the seed class (same
top empirical failure modes). They differ on **content scope**:

- **Intra** — same archetype/content kind as the seeds.
- **Inter** — different content, same pathology.

### A. Intra-archetype expansion (target: **{target_intra}** entries)

{intra_host_rule}

### B. Inter-archetype expansion (target: **{target_inter}** entries)

Find Korean websites that exhibit **the same retrieval pathology** but serve
**different content** than the seed class. The goal is to expand the seed
pool into off-the-beaten-path corners of the Korean web — places a
search-augmented solver doesn't know how to query — so the downstream
problem-generation stage can produce harder problems.

For a class like `species_inventory` (F4 = semi-structured parsing), inter
candidates exhibit the same row-level-lookup difficulty but talk about
something else entirely. Examples to inspire (not exhaustive):

- 국가유산청 / 문화재청 detail pages (cultural property metadata rows)
- 식약처 식품·의약품 등록 검색 결과 (product registration rows)
- 한국향토문화전자대전 (regional culture encyclopedia row tables)
- 국가법령정보센터 별표 / 별지 (regulatory annexes with row tables)
- KOSIS 통계표 (statistical tables, often JS-rendered)
- 국립국악원 / 국립한글박물관 collection catalogs
- 국가전자도서관 specialized catalog detail pages

The same logic applies to other classes — let the failure modes drive what
"similar pathology" means.

Strategies:
- Reason from the failure mode forwards: "F4 means row-table extraction
  difficulty — where else in the Korean web do row-tables hide answers in
  cells that don't appear in SERP snippets?"
- Use `WebSearch` to find Korean institutional sites you don't know about.
- Confirm structural similarity by fetching a sample page from each host
  before adding multiple URLs from it.

### Diversity requirements (HARD RULES — apply to BOTH intra and inter)

Each output file (`intra.json` and `inter.json` independently) must satisfy:

1. **No host overlap with seed bank.** Every `website_url`'s host must NOT
   appear in this class's host list (`class_context.json`'s `hosts` keys).
2. **Each entry distinct.** No two entries within the same file may share a
   `website_url`. The `website_url` must also not equal any URL in the seed
   bank (including the entry's own `source`).
3. **Minimum host diversity.** Span at least **`max(5, ceil(target / 5))`**
   distinct hosts in the file (target = `{target_intra}` for intra,
   `{target_inter}` for inter).
4. **Per-host cap.** No more than **`ceil(target / 3)`** entries from any
   single host. If one host produces easy verifications, stop at the cap and
   go find another host.
5. **Push through blocks before abandoning a host.** If a candidate host
   blocks `WebFetch`, try `Bash` + `curl` (see Step 3 of the verification
   protocol below) before giving up on it.

---

## Verification protocol (MANDATORY)

This is the highest-priority rule. For **every** URL you put in the output:

### Step 1 — Probe with `WebFetch`

Call `WebFetch` on the URL. Inspect what it returned:
- **HTTP status** (200 / 301 / 404 / 500 / connection error)
- Whether the page body contains **load-bearing content** that identifies the
  page (a 학명 cell, a 분포 row, a specific row value), or only **boilerplate**
  (site title / nav chrome / generic landing-page text)
  - Good load-bearing: `"학명: Elaphe schrenckii  국명: 구렁이  분포: 한반도 전역  멸종위기 등급: II급"`
  - Boilerplate (not load-bearing): `"한반도의 생물다양성 | 국립생물자원관"`

### Step 2 — Reject anything that fails

**Drop** the URL (do not include in output) if:
- HTTP status is **not** in {{200, 301, 302}}
- `WebFetch` returns ECONNREFUSED / DNS error / connection timeout AND the
  curl probe in Step 3 also fails
- The page renders 200 with **only** boilerplate AND you cannot verify the
  page is a real detail page via Step 3 or 4
- **The Korean-relevance gate (criterion #4 above) fails.** Specifically,
  drop if neither (a) the page contains Korean-language load-bearing
  content NOR (b) the page's subject is a Korean entity/place/company/
  cultural-or-historical item that a Korean-language question would
  naturally ask about. Do NOT include candidates that satisfy the
  structural criteria (table-heavy, snippet-poor) but fail the Korean-
  relevance test — they belong in some other benchmark, not K-BrowseComp.

### Step 3 — `curl` probe (use for bot-blocked hosts)

Many Korean gov sites block Claude Code's `WebFetch` but accept a regular
browser user-agent. Use `Bash` to run a real HTTP probe before giving up:

```bash
curl -sSL -m 15 \
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36' \
  -o $(pwd)/_probe_page.html -w '%{{http_code}}' '<URL>'
```

- If `curl` returns 200 and `$(pwd)/_probe_page.html` contains load-bearing Korean
  content, record `verification.method = "curl"`, the actual HTTP status, and
  a 100–300 char snippet copied from the HTML.
- If `curl` returns 4xx/5xx, **drop the URL** — it doesn't exist.
- If `curl` also fails (ECONNREFUSED, timeout, certificate error), fall
  through to Step 4.

### Step 4 — `WebSearch` fallback (last resort)

Use only when both `WebFetch` and `curl` failed AT THE NETWORK LEVEL, OR when
`WebFetch` returned 200 with boilerplate only AND `curl` is also returning
the same JS-rendered shell (i.e., the host runs an SPA you genuinely cannot
read server-side).

1. Run `WebSearch` with the exact URL, **or** a precise phrase that should
   appear on the page.
2. The SERP must show that URL **with a snippet that contains load-bearing
   content** — actual row values from the page, not just the site title.
3. Record `verification.method = "websearch"`, `verification.http_status = -1`,
   and `verification.evidence` = the load-bearing SERP snippet (must include
   distinguishing detail, not just `"<site> | <section>"`).

**Never use the WebSearch fallback to bypass a 404.** A 4xx/5xx URL from
either WebFetch or curl is invalid, full stop.

**Never accept a SERP snippet that only shows the site title** — that's
indistinguishable from a 404 page that Google indexed. The snippet must
contain something that proves the page has actual detail content.

### Step 5 — Do not pattern-guess IDs

Never produce a URL by extrapolating an ID format from another page (e.g.,
"the seed uses `ktsn=120000064254`, so `ktsn=120000064255` probably exists").
Always come from a real index, a real link, or a real search result. Numeric
IDs are exactly where hallucinations come from.

---

## Output schema (strict — validator will enforce)

Write final results to two files in **this directory**:

- `./intra.json` — JSON array of intra-domain candidates
- `./inter.json` — JSON array of inter-domain candidates

Each entry has **exactly** these fields:

```json
{{
  "source": "<an existing seed URL anchoring this candidate>",
  "website_url": "<the new URL you discovered and verified>",
  "website_summary": "<one sentence: what kind of page this is>",
  "why_this_is_promising": "<one or two sentences: which failure mode(s) this targets and why the page structure would defeat a search-augmented solver>",
  "target_failure_modes": ["F4", "F6"],
  "verification": {{
    "method": "webfetch" | "curl" | "websearch",
    "http_status": 200,
    "evidence": "<100-300 char load-bearing snippet from the page or SERP>"
  }}
}}
```

`target_failure_modes` is a non-empty array of failure-mode codes from
`{{F1, F3, F4, F6, F7, F8, F9, F10}}` (see `class_context.json`
`failure_mode_names` for definitions). These should match the FMs you
mentioned in `why_this_is_promising`. **For both intra and inter, the first
FM listed in this array IS the bucket this candidate fills against the
per-FM quota table above.** Across all your intra entries combined, the
distribution of first-listed FMs must match the **Intra target** column;
same for inter and the **Inter target** column. You may include a second
FM if the page genuinely exhibits two pathologies.

`method` values:
- `"webfetch"` — Step 1 succeeded; `http_status` is the HTTP code observed.
- `"curl"` — Step 3 succeeded after WebFetch was blocked; `http_status` is the
  HTTP code from `curl -w '%{{http_code}}'`.
- `"websearch"` — Step 4 last-resort fallback; `http_status = -1`; `evidence`
  is the load-bearing SERP snippet (not a site-title-only snippet).

### Rules for the `source` field

- For **both intra and inter** entries: `source` must be an existing seed URL
  in this class (any URL listed in `class_context.json` `sample_urls` or
  anywhere in this class's seed set). It records which seed your candidate is
  anchored to. The seed URL and the new `website_url` must not be equal, and
  their hosts must differ (since the new URL's host is NOT in the seed bank).

### Rules for `why_this_is_promising`

- Be specific about which failure mode (F1/F3/F4/F6/F7/F8/F9/F10) the page
  would exercise.
- Tie the rationale to the page's *structure*, not just its topic.
- **Include a one-clause justification of Korean relevance** — either
  "page is Korean-language" or "subject is the Korean entity ___, which a
  Korean question would naturally ask about." If you can't write that
  clause honestly, drop the candidate.

### Example of a well-formed entry (intra — same archetype, different host)

```json
{{
  "source": "https://species.nibr.go.kr/home/mainHome.do?contCd=009002&cont_link=009&ktsn=120000064254&pageMode=view&subMenu=009002",
  "website_url": "https://www.mbris.kr/pub/marine/tsearch/tsearchDetail.do?spcTxnId=270000005487",
  "website_summary": "MBRIS (국립해양생물자원관) marine-species detail page for 가시복 (Diodon holocanthus); structured row-table with 학명/명명자/taxonomy chain/분포/표본 보유기관 cells.",
  "why_this_is_promising": "F4. Same row-level lookup pathology as the seed: 학명·명명자 and 분포 cells live in stacked label-value rows; SERP snippet only shows the portal header. The solver must hit the structured detail page to read the binomial-with-author and the holding-institution row.",
  "target_failure_modes": ["F4"],
  "verification": {{
    "method": "curl",
    "http_status": 200,
    "evidence": "<p class=\"tit\">가시복<span class=\"stxt\"><em>Diodon holocanthus</em> Linnaeus, 1758</span></p> Animalia동물계 > Chordata척삭동물문 > Teleostei > Tetraodontiformes복어목 > Diodontidae가시복과 > Diodon가시복속"
  }}
}}
```

### Example of a well-formed entry (inter — same pathology, different topic)

```json
{{
  "source": "https://species.nibr.go.kr/home/mainHome.do?contCd=009002&cont_link=009&ktsn=120000064254&pageMode=view&subMenu=009002",
  "website_url": "https://muju.grandculture.net/muju/toc/GC06500281",
  "website_summary": "디지털무주문화대전 entry for 반딧불이 (firefly) in the Muju regional cultural encyclopedia; tabular layout with 학명, 생물학적 분류 rows, and a paragraph enumerating species (Hotaria unmunsana, Luciola lateralis, etc.).",
  "why_this_is_promising": "F4 + F6. Same row-level extraction pathology as species_inventory but on a regional cultural encyclopedia (different topic). The 학명 row cell carries the Latin binomial; species-list paragraph forces rare-entity disambiguation across multiple Hotaria/Luciola entries that don't appear in SERP snippets.",
  "target_failure_modes": ["F4", "F6"],
  "verification": {{
    "method": "curl",
    "http_status": 200,
    "evidence": "<th>학명</th><td>Luciola cruciata</td> ... 우리나라에는 5속 6종이 서식 — 운문산반딧불이[Hotaria unmunsana], 파파리반딧불이[Hotaria papariensis], 애반딧불이[Luciola lateralis], 늦반딧불이[Pyrocoelia rufa]."
  }}
}}
```

---

## Useful files in this workspace

- `./task_spec.md` — this file.
- `./class_context.json` — machine-readable class metadata (hosts, sample URLs,
  failure-mode histogram, failure-mode name glossary).

You may read those at any time. Do not edit them.

---

## Style and budget guidance

- Use search and fetch aggressively. Spending 60–100 `WebSearch` / `WebFetch`
  calls on one class is fine — the seed expansion is amortized across
  hundreds of downstream generation runs.
- Quality > quantity. {target_intra} verified intra entries beats
  {target_intra}×2 unverified guesses.
- Write JSON outputs incrementally if helpful, but the final state of
  `./intra.json` and `./inter.json` is what matters.

When you are confident the two output files contain only **verified** entries
with **load-bearing** evidence snippets, you are done.
