#!/bin/bash
# Build the Kaggle upload bundle (Task C).
#
#   scripts/kaggle/make_kaggle_bundle.sh <git-ref> <out.zip>
#
# The bundle is ONE zip that the user uploads as a Kaggle Dataset; the notebook
# finds it under /kaggle/input/*/ and unpacks it into /kaggle/working.
#
# Layout inside the zip:
#
#   repo/                  git archive of <git-ref> (tracked files only, incl. assets/)
#   data/canonical/        canonical_ctle_phase_a_v2.npz  (schema 2, 20k specs)
#   data/zig/              the ZIG v2 forward surrogate
#   data/ctle_csv/         the CTLE history CSVs the dagger script concatenates
#   MANIFEST.txt           git sha + sha256 of every packed file
#
# The CTLE CSVs are needed even in canonical mode: the dagger script loads them
# at import time (dagger-nuance-distillation-kirchhoffnet.py, "csv_files") to
# rebind KN_INPUT_LOG_MIN / KN_INPUT_LOG_MAX (the student input scaling).
# Only the CSVs the script names are packed; the CSVs live behind a symlink
# (~/Documents/augmented-cvae-ctle), which is followed.
set -euo pipefail

usage() {
  echo "usage: $0 <git-ref> <out.zip>" >&2
  echo "  <git-ref>  a commit-ish (branch, tag or sha) of this repository" >&2
  echo "  <out.zip>  the zip to write" >&2
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

REF="$1"
OUT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --------------------------------------------------------------------------
# 1. the external data (all overridable; the defaults are the ssr0 paths)
# --------------------------------------------------------------------------
HOME_DIR="${HOME}"
CANONICAL="${CANONICAL:-${HOME_DIR}/Documents/ASPDAC_2026/kirchhoff_redesign/ideal/models/canonical_ctle_phase_a_v2.npz}"
ZIG_DIR="${ZIG_DIR:-${HOME_DIR}/Documents/improved-zig-nf-spline-pytorch-default-v2}"
DATA_DIR="${DATA_DIR:-${HOME_DIR}/Documents/augmented-cvae-ctle}"

MAX_MB="${MAX_MB:-100}"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$CANONICAL" ] || die "canonical dataset not found: $CANONICAL (override with CANONICAL=...)"
[ -d "$ZIG_DIR" ] || die "ZIG surrogate dir not found: $ZIG_DIR (override with ZIG_DIR=...)"
[ -d "$DATA_DIR" ] || die "CTLE CSV dir not found: $DATA_DIR (override with DATA_DIR=...)"

# --------------------------------------------------------------------------
# 2. resolve the ref and stage the tree
# --------------------------------------------------------------------------
git -C "$REPO" rev-parse --verify "${REF}^{commit}" >/dev/null 2>&1 \
  || die "git ref not found in ${REPO}: ${REF}"
SHA="$(git -C "$REPO" rev-parse "${REF}^{commit}")"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref "${REF}" 2>/dev/null || echo "${REF}")"
DESCRIBE="$(git -C "$REPO" describe --always --dirty "${SHA}" 2>/dev/null || echo "${SHA}")"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/kaggle_bundle.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "${STAGE}/repo" "${STAGE}/data/canonical" "${STAGE}/data/zig" "${STAGE}/data/ctle_csv"

echo "[bundle] git archive ${REF} (${SHA}) -> repo/"
git -C "$REPO" archive --format=tar "${SHA}" | tar -xf - -C "${STAGE}/repo"

# The bundle must carry the notebook itself, so a Kaggle run uses the same
# code as the archive it was built from.
if [ ! -f "${STAGE}/repo/notebooks/kaggle_hw_retrain.ipynb" ]; then
  echo "[bundle] WARNING: notebooks/kaggle_hw_retrain.ipynb is not in ${SHA}" >&2
fi

echo "[bundle] canonical dataset -> data/canonical/"
cp -L "$CANONICAL" "${STAGE}/data/canonical/canonical_ctle_phase_a_v2.npz"

echo "[bundle] ZIG surrogate -> data/zig/"
# -L: follow symlinks; the ZIG dir itself is a plain directory.
for f in "$ZIG_DIR"/*; do
  [ -f "$f" ] || continue
  cp -L "$f" "${STAGE}/data/zig/$(basename "$f")"
done

# The CSVs the dagger script actually concatenates.  The list is parsed out of
# the script itself (so it cannot drift), with a hard-coded fallback.
CSVS=()
while IFS= read -r name; do
  [ -n "$name" ] && CSVS+=("$name")
done < <(
  sed -n "/^csv_files = \[/,/^\]/p" "${REPO}/dagger-nuance-distillation-kirchhoffnet.py" 2>/dev/null \
    | grep -o "'[^']*\.csv'" | tr -d "'"
)
if [ "${#CSVS[@]}" -eq 0 ]; then
  CSVS=(dataset_log_part_1_may3.csv dataset_log_part_1_may4.csv dataset_log_part_2_may3.csv
        dataset_log_part_2_may4.csv dataset_log_part_3_may3.csv dataset_log_part_3_may4.csv
        dataset_log_part_4_may3.csv dataset_log_part_4_may4.csv dataset_log_part_5_may3.csv
        dataset_log_part_5_may4.csv dataset_log_part_6_may3.csv dataset_log_part_6_may4.csv
        dataset_log_may_7.csv dataset_log_march12.csv dataset_log_may_18.csv)
fi

echo "[bundle] CTLE CSVs -> data/ctle_csv/ (${#CSVS[@]} files)"
for name in "${CSVS[@]}"; do
  src="${DATA_DIR}/${name}"
  if [ ! -f "$src" ]; then
    echo "[bundle] WARNING: CSV missing (the dagger script skips it): $src" >&2
    continue
  fi
  cp -L "$src" "${STAGE}/data/ctle_csv/${name}"
done

# --------------------------------------------------------------------------
# 3. MANIFEST.txt: the git sha and the hash of every packed file
# --------------------------------------------------------------------------
echo "[bundle] writing MANIFEST.txt"
MANIFEST="${STAGE}/MANIFEST.txt"
{
  echo "# Kaggle bundle manifest"
  echo "generated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_ref: ${REF}"
  echo "git_branch: ${BRANCH}"
  echo "git_sha: ${SHA}"
  echo "git_describe: ${DESCRIBE}"
  echo "canonical_source: ${CANONICAL}"
  echo "zig_source: ${ZIG_DIR}"
  echo "ctle_csv_source: ${DATA_DIR}"
  echo "ctle_csv_files: ${#CSVS[@]}"
  echo "#"
  echo "# sha256 of every file in the bundle (paths are relative to the zip root):"
  ( cd "$STAGE" && find repo data -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum )
} > "$MANIFEST"

# --------------------------------------------------------------------------
# 4. zip it and check the size
# --------------------------------------------------------------------------
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
echo "[bundle] zipping -> ${OUT}"
( cd "$STAGE" && zip -q -r -X "$OUT" repo data MANIFEST.txt )

BYTES="$(stat -c %s "$OUT")"
MB="$(awk -v b="$BYTES" 'BEGIN { printf "%.1f", b / 1048576 }')"
echo "[bundle] size: ${BYTES} bytes (${MB} MB)"
if awk -v b="$BYTES" -v m="${MAX_MB}" 'BEGIN { exit !(b > m * 1048576) }'; then
  echo "[bundle] ERROR: the bundle is larger than ${MAX_MB} MB" >&2
  exit 1
fi

echo "[bundle] OK: ${OUT} (${MB} MB, git ${SHA})"
