#!/usr/bin/env bash
#
# run_all.sh — Three-tier seed-bank expansion to size the seed pool for ~700
# downstream K-BrowseComp problems.
#
# Rationale:
#   - Target 700 generated problems in Stage 3 (problem_generation/).
#   - Assuming ~1.3 problems per accepted seed and ~70% defeat-loop acceptance,
#     we need ~800-900 candidate seed URLs from Stage 2.
#   - Tier 1 = sparse but high-strategic-value classes (F4/F6/F8): boost a lot.
#   - Tier 2 = already well-covered classes (the core of the existing 300):
#     moderate expansion, mostly intra-domain since they have many seed hosts.
#   - Tier 3 = niche / limited-utility classes (F10/F9-dominant): light coverage.
#   - Skip: SERP-only and shortlink classes; too few seeds in 2 others.
#
# Cost note:
#   claude-opus-4-7 at max effort over ~30 classes with aggressive WebSearch +
#   WebFetch is expensive. Run the warm-up first (commented below), inspect a
#   handful of outputs in work/, then uncomment the three tier invocations.
#
# Usage:
#   bash automated_data_gen/seed_expansion/run_all.sh
#
# Override knobs:
#   PARALLEL=4   bash run_all.sh           # higher concurrency (interleaved logs)
#   RESUME=1     bash run_all.sh           # skip classes whose intra+inter exist
#

set -euo pipefail
cd "$(dirname "$0")"

PARALLEL="${PARALLEL:-1}"
EXTRA_FLAGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--resume)
fi

# -----------------------------------------------------------------------------
# Warm-up (uncomment to sanity-check the brief and agent quality on 2 classes
# before committing to the full run).
# -----------------------------------------------------------------------------
# echo "=== WARM-UP (2 sparse classes) ==="
# python3 run.py \
#   --classes species_inventory,encyclopedia_korean_culture \
#   --target-intra 10 --target-inter 10 \
#   --parallel "$PARALLEL" "${EXTRA_FLAGS[@]}"
# echo
# echo "Inspect work/species_inventory/{intra,inter}.json and"
# echo "work/encyclopedia_korean_culture/{intra,inter}.json before continuing."
# exit 0

# -----------------------------------------------------------------------------
# Tier 1 — sparse but strategically valuable. Heavy inter-domain expansion to
# build out under-represented archetypes (F4 / F6 / F8 dominant).
# Budget per class: intra=20, inter=25  (8 classes × 45 = 360 candidates)
# -----------------------------------------------------------------------------
echo "=== TIER 1: sparse high-value classes ==="
python3 run.py \
  --classes species_inventory,encyclopedia_korean_culture,sports_records,arxiv_paper,dbpia_academic,music_streaming,korean_research_institute,ebs_education \
  --target-intra 20 --target-inter 25 \
  --parallel "$PARALLEL" "${EXTRA_FLAGS[@]}"

# -----------------------------------------------------------------------------
# Tier 2 — already well-covered, moderate expansion. Mostly intra-domain since
# these have many seed hosts to walk from.
# Budget per class: intra=20, inter=15  (8 classes × 35 = 280 candidates)
# -----------------------------------------------------------------------------
echo "=== TIER 2: core classes ==="
python3 run.py \
  --classes namu_wiki_main,news_legacy,government_notice,academic_korean,wikipedia_ko_article,bookstore,namu_wiki_subsection,organization_korean \
  --target-intra 20 --target-inter 15 \
  --parallel "$PARALLEL" "${EXTRA_FLAGS[@]}"

# -----------------------------------------------------------------------------
# Tier 3 — niche / limited utility. Light coverage to keep mode diversity.
# Budget per class: intra=10, inter=10  (14 classes × 20 = 280 candidates)
# -----------------------------------------------------------------------------
echo "=== TIER 3: niche classes ==="
python3 run.py \
  --classes daum,naver_blog,naver_other,naver_encyclopedia,wikipedia_en_article,nate,brunch_blog,government_portal,financial_card,instagram,youtube_video,social_misc,github_repo,global_news \
  --target-intra 10 --target-inter 10 \
  --parallel "$PARALLEL" "${EXTRA_FLAGS[@]}"

# -----------------------------------------------------------------------------
# Final aggregation (re-runs even if Tier 3 succeeded, since run.py only
# aggregates per-invocation — this guarantees a fresh aggregate from all work/).
# -----------------------------------------------------------------------------
echo "=== AGGREGATE ==="
python3 run.py --aggregate-only

echo
echo "Done. Outputs:"
echo "  intra_domain_candidates.json"
echo "  inter_domain_candidates.json"
