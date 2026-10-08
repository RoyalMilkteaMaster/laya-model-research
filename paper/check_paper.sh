#!/usr/bin/env bash
# Mechanical checks for the paper.
#   bash paper/check_paper.sh            # static checks only
#   bash paper/check_paper.sh --compile  # also compile with tectonic into a temp dir
#                                         # (CHECK_OUTDIR=<dir> to choose the output dir;
#                                         # fails unless the References end on page PAGE_LIMIT
#                                         # (default 8) with that page filled and the Appendix
#                                         # starts on the next page, or on any Overfull \hbox)
# Revision R9: CVPR author kit format (cvpr.sty, ieeenat_fullname.bst) as in the reference paper;
# ruled (grid) tables, as in the reference paper (CHECK_GRID=0 skips that check). R9 round 2: an
# Appendix (A, B) follows the References from a new page and is outside the page budget.
#   bash paper/check_paper.sh --self-test # run the checks on temp copies with injected
#                                         # faults / Ticket 11 fixtures; never touches paper/
# Exit code 0 = all checks passed.
set -u
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PAPER_DIR" || exit 2
failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }
pass() { echo "PASS: $*"; }

# ---------------------------------------------------------------------------
# Self-test: every check must fail on a copy with the matching fault injected.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  root="$(mktemp -d)"
  copy() {  # fresh copy of the sources and generated tables/figures (never main.pdf)
    local d; d="$(mktemp -d "$root/case.XXXX")/paper"; mkdir -p "$d"
    cp -r main.tex sections refs.bib cvpr.sty ieeenat_fullname.bst zh-TW check_paper.sh "$d/"
    [ "${1:-}" = bare ] || for g in tables figures; do [ -d "$g" ] && cp -r "$g" "$d/"; done
    echo "$d"
  }
  run() { (cd "$1" && shift && bash check_paper.sh "$@") > "$root/last.txt" 2>&1; }
  expect_pass() { if run "$@"; then pass "self-test: $name"; else fail "self-test: $name"; tail -5 "$root/last.txt"; fi; }
  expect_fail() {  # expect_fail <copy> <required FAIL text> [checker args...]
    local dir="$1" want="$2"; shift 2
    if run "$dir" "$@"; then fail "self-test: $name (checker exited 0)"
    elif grep -qF "$want" "$root/last.txt"; then pass "self-test: $name"
    else fail "self-test: $name (missing '$want')"; tail -5 "$root/last.txt"; fi
  }

  name="pristine sources pass"; d="$(copy)"; expect_pass "$d"

  name="\\pending in any section fails"; d="$(copy)"
  printf '\n\\pending{LEFTOVER}\n' >> "$d/sections/experiments.tex"
  expect_fail "$d" "FAIL: placeholder in sections/experiments.tex"

  name="TODO(14) comment fails"; d="$(copy)"
  printf '\n%% TODO(14): 待 14 補上\n' >> "$d/sections/discussion.tex"
  expect_fail "$d" "FAIL: placeholder in sections/discussion.tex"

  name="placeholder in main.tex (author block) fails"; d="$(copy)"
  sed -i 's/^\\maketitle$/\\pending{author}\n\\maketitle/' "$d/main.tex"
  expect_fail "$d" "FAIL: placeholder in main.tex"

  name="zh-TW placeholder in any paragraph fails"; d="$(copy)"
  sed -i '0,/<!-- EN: §4.1 ¶1 -->/{/<!-- EN: §4.1 ¶1 -->/a\
**【待 14 補上】**
}' "$d/zh-TW/main.md"
  expect_fail "$d" "FAIL: zh-TW placeholder in paragraph §4.1 ¶1"

  name="missing generated figure fails"; d="$(copy)"
  rm -f "$d/figures/hops_distribution.pdf"
  expect_fail "$d" "FAIL: generated file missing: figures/hops_distribution.pdf"

  name="missing generated two-column table fails (\\widetableorplaceholder, revision R8)"; d="$(copy)"
  rm -f "$d/tables/efficiency_panels.tex"
  expect_fail "$d" "FAIL: generated file missing: tables/efficiency_panels.tex"

  name="missing main-results panel table fails (revision R8 hook)"; d="$(copy)"
  rm -f "$d/tables/main_results_panels.tex"
  expect_fail "$d" "FAIL: generated file missing: tables/main_results_panels.tex"

  name="missing error taxonomy table fails (revision R8 hook)"; d="$(copy)"
  rm -f "$d/tables/error_taxonomy.tex"
  expect_fail "$d" "FAIL: generated file missing: tables/error_taxonomy.tex"

  name="missing zh-TW marker fails"; d="$(copy)"
  sed -i '/<!-- EN: §3.4 ¶2 -->/d' "$d/zh-TW/main.md"
  expect_fail "$d" "FAIL: marker mismatch"

  name="appendix before the References fails (section order, R9 round 2)"; d="$(copy)"
  sed -i '/^\\clearpage$/d; /^\\appendix$/d; /^\\input{sections\/appendix}$/d' "$d/main.tex"
  sed -i 's/^\\bibliographystyle{ieeenat_fullname}$/\\appendix\n\\input{sections\/appendix}\n&/' "$d/main.tex"
  a="$(grep -n '^\\appendix$' "$d/main.tex" | cut -d: -f1)"; b="$(grep -n '^\\bibliography{' "$d/main.tex" | cut -d: -f1)"
  if [ -z "$a" ] || [ -z "$b" ] || [ "$a" -gt "$b" ]; then fail "self-test: $name (fault not injected)"
  else expect_fail "$d" "FAIL: section order"; fi

  name="main text citing an appendix table fails (R9 round 2)"; d="$(copy)"
  sed -i 's/Table~\\ref{tab:error_taxonomy} assigns/Table~\\ref{tab:error_taxonomy_by_dataset} assigns/' "$d/sections/experiments.tex"
  if ! grep -q 'tab:error_taxonomy_by_dataset} assigns' "$d/sections/experiments.tex"; then fail "self-test: $name (fault not injected)"
  else expect_fail "$d" "FAIL: cited float tab:error_taxonomy_by_dataset is not defined in the main text"; fi

  name="IEEEtran document class fails the CVPR format check (revision R9)"; d="$(copy)"
  sed -i 's/^\\documentclass\[10pt,twocolumn,letterpaper\]{article}$/\\documentclass[conference]{IEEEtran}/' "$d/main.tex"
  expect_fail "$d" "FAIL: CVPR format"

  name="booktabs rule in a generated table fails (revision R9: grid tables)"; d="$(copy)"
  sed -i '0,/^\\hline$/s//\\toprule/' "$d/tables/error_taxonomy.tex"
  expect_fail "$d" "FAIL: table style in tables/error_taxonomy.tex"

  name="\\ref to a table not defined in the main text fails (revision R8)"; d="$(copy)"
  sed -i 's/Table~\\ref{tab:error_taxonomy} assigns/Table~\\ref{tab:not_in_main_text} assigns/' "$d/sections/experiments.tex"
  expect_fail "$d" "FAIL: cited float tab:not_in_main_text is not defined in the main text"

  name="abstract above 200 words fails (revision R8)"; d="$(copy)"
  sed -i 's/^accuracy-efficiency trade-off\.$/accuracy-efficiency trade-off, which we state here in many more words than allowed./' "$d/sections/abstract.tex"
  if ! grep -q 'many more words' "$d/sections/abstract.tex"; then fail "self-test: $name (fault not injected)"
  else expect_fail "$d" "FAIL: abstract length"; fi

  name="abstract below 195 words fails (revision R8)"; d="$(copy)"
  sed -i 's/^Small language models (SLMs) often mismanage multi-hop question-answering loops: they stop$/Small language models (SLMs) often mismanage loops: they stop/' "$d/sections/abstract.tex"
  sed -i 's/^early, retrieve needlessly, or misuse tools\. Adaptive retrieval mostly reads control$/early. Adaptive retrieval mostly reads control/' "$d/sections/abstract.tex"
  if grep -q 'needlessly' "$d/sections/abstract.tex"; then fail "self-test: $name (fault not injected)"
  else expect_fail "$d" "FAIL: abstract length"; fi

  name="title above 12 words fails (revision R8)"; d="$(copy)"
  sed -i 's/^\\title{Calibrated Loop Control for/\\title{A Study of Calibrated External Loop Control for/' "$d/main.tex"
  expect_fail "$d" "FAIL: title length"

  name="extra appendix zh-TW marker fails (R9 round 2)"; d="$(copy)"
  printf '\n<!-- EN: §B ¶9 -->\n附錄。\n' >> "$d/zh-TW/main.md"
  expect_fail "$d" "FAIL: marker mismatch"

  name="missing appendix zh-TW marker fails (R9 round 2)"; d="$(copy)"
  sed -i '/<!-- EN: §A ¶2 -->/d' "$d/zh-TW/main.md"
  expect_fail "$d" "FAIL: marker mismatch"

  name="TeX paired quotes fail (typography)"; d="$(copy)"
  printf '\nSee ``quoted'"''"' text.\n' >> "$d/sections/discussion.tex"
  expect_fail "$d" "FAIL: typography in sections/discussion.tex"

  name="unknown citation key fails"; d="$(copy)"
  sed -i 's/\\cite{lewis2020rag}/\\cite{missing_reference_key}/' "$d/sections/introduction.tex"
  expect_fail "$d" "FAIL: cited key not in refs.bib: missing_reference_key"

  if command -v tectonic >/dev/null 2>&1; then
    name="compile without Ticket 11 files: every table/figure gets caption + label"
    d="$(copy bare)"; export CHECK_OUTDIR="$root/out-plain" CHECK_ALLOW_MISSING=1
    expect_pass "$d" --compile; unset CHECK_OUTDIR CHECK_ALLOW_MISSING
    fig_fixture="$root/out-plain/main.pdf"

    name="BST with TeX paired quotes around titles fails (main.bbl typography)"; d="$(copy)"
    sed -i 's/^    { title "t" change.case\$ }$/    { "``" title "t" change.case$ * "'"''"'" * }/' "$d/ieeenat_fullname.bst"
    if ! grep -qF '{ "``" title' "$d/ieeenat_fullname.bst"; then
      fail "self-test: $name (fault not injected)"
    else export CHECK_OUTDIR="$root/out-bst"; expect_fail "$d" "FAIL: paired TeX quotes in main.bbl" --compile; unset CHECK_OUTDIR; fi

    name="compile with bare-tabular + figure fixtures (Ticket 11 contract)"
    d="$(copy bare)"; mkdir -p "$d/tables" "$d/figures"; export CHECK_ALLOW_MISSING=1
    printf '%s\n' '\begin{tabular}{|l|r|}' '\hline' 'Model & Valid rate \\' '\hline' \
      'fixture-model & 0.00 \\' '\hline' '\end{tabular}' > "$d/tables/controller_probe.tex"
    cp "$fig_fixture" "$d/figures/latency_vs_size.pdf"
    out="$root/out-fixture"; export CHECK_OUTDIR="$out"; expect_pass "$d" --compile
    unset CHECK_OUTDIR CHECK_ALLOW_MISSING
    name="fixture table body is used"
    grep -q 'tables/controller_probe.tex' "$out/main.log" && pass "self-test: $name" || fail "self-test: $name"
    name="fixture table caption and number reach main.aux"
    if grep -q '\\newlabel{tab:controller_probe}' "$out/main.aux" \
       && grep -q 'contentsline {table}{\\numberline {[0-9]*}{\\ignorespaces Function-calling probe' "$out/main.aux"; then
      pass "self-test: $name"; grep -h 'tab:controller_probe\|Function-calling probe' "$out/main.aux"
    else fail "self-test: $name"; fi
    name="fixture figure is included with caption and label"
    if grep -q 'figures/latency_vs_size.pdf' "$out/main.log" && grep -q '\\newlabel{fig:latency_vs_size}' "$out/main.aux"; then
      pass "self-test: $name"; grep -h 'fig:latency_vs_size' "$out/main.aux"
    else fail "self-test: $name"; fi

    name="References running onto page 9 fail the page budget (R9 round 2)"; d="$(copy)"
    { printf '\n%%%% [§5 ¶5]\n'; for i in $(seq 1 60); do printf 'Filler sentence number %s pads the discussion beyond the page budget of the paper. ' "$i"; done; printf '\n'; } \
      >> "$d/sections/discussion.tex"
    { printf '\n<!-- EN: §5 ¶5 -->\n填充。\n'; } > "$root/zh_extra.md"
    awk -v f="$root/zh_extra.md" '/^### 5\.2\. 研究限制/{while((getline l < f) > 0) print l; print ""} {print}' "$d/zh-TW/main.md" > "$d/zh-TW/main.md.new" \
      && mv "$d/zh-TW/main.md.new" "$d/zh-TW/main.md"
    export CHECK_OUTDIR="$root/out-pages"; expect_fail "$d" "FAIL: page budget: References end on page 9" --compile; unset CHECK_OUTDIR

    # R9 round 2: the limitations paragraph (about 13 lines) emptied, so the References end early on page 8.
    name="unfilled page 8 fails (R9 round 2)"; d="$(copy)"
    awk '/^%% \[§5 ¶4\]/{print; skip=1; next} skip && /^[[:space:]]*$/{skip=0} !skip' "$d/sections/discussion.tex" > "$d/x" && mv "$d/x" "$d/sections/discussion.tex"
    if grep -q 'Thinking is disabled' "$d/sections/discussion.tex"; then fail "self-test: $name (fault not injected)"
    else export CHECK_OUTDIR="$root/out-unfilled"; expect_fail "$d" "FAIL: page 8 not filled" --compile; unset CHECK_OUTDIR; fi

    name="appendix pulled back onto page 8 fails (R9 round 2)"; d="$(copy)"
    awk '/^%% \[§5 ¶4\]/{print; skip=1; next} skip && /^[[:space:]]*$/{skip=0} !skip' "$d/sections/discussion.tex" > "$d/x" && mv "$d/x" "$d/sections/discussion.tex"
    sed -i '/^\\clearpage$/d' "$d/main.tex"
    if grep -q '^\\clearpage$' "$d/main.tex" || grep -q 'Thinking is disabled' "$d/sections/discussion.tex"; then fail "self-test: $name (fault not injected)"
    else export CHECK_OUTDIR="$root/out-apppage"; expect_fail "$d" "FAIL: appendix starts on page 8" --compile; unset CHECK_OUTDIR; fi

    name="Overfull \\hbox fails (revision R8)"; d="$(copy)"
    sed -i 's/^All 60 configurations were run once with 200 questions each/\\noindent\\rule{1.2\\columnwidth}{0.4pt}\\par\n&/' "$d/sections/experiments.tex"
    if ! grep -q 'rule{1.2' "$d/sections/experiments.tex"; then fail "self-test: $name (fault not injected)"
    else export CHECK_OUTDIR="$root/out-overfull"; expect_fail "$d" "FAIL: Overfull" --compile; unset CHECK_OUTDIR; fi
  else
    echo "NOTE: tectonic not found; compile self-tests skipped"
  fi
  echo "self-test failures: $failures (work dir: $root)"
  [ "$failures" -eq 0 ]; exit $?
fi

# Expand main.tex by following \input{sections/...} lines (recursively), in order.
expand() {
  local file="$1" line target
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" =~ ^[[:space:]]*\\input\{(sections/[A-Za-z_]+)\} ]]; then
      target="${BASH_REMATCH[1]}.tex"
      expand "$target"
    else
      printf '%s\n' "$line"
    fi
  done < "$file"
}
expanded="$(expand main.tex)"

# 1. Section order (revision R9: CVPR numbering "1. Introduction"; still checked on the \section{} names).
actual_order="$(printf '%s\n' "$expanded" | grep -v '^[[:space:]]*%' | sed -n \
  -e 's/.*\\begin{abstract}.*/Abstract/p' \
  -e 's/^\\section{\([^}]*\)}.*/\1/p' \
  -e 's/^\\appendix.*/Appendix/p' \
  -e 's/^[[:space:]]*\\bibliography{.*/References/p' | paste -sd '|')"
expected_order="Abstract|Introduction|Related Work|Method|Experimental Results|Discussion|Conclusion|References|Appendix|Implementation and Reproducibility Details|Supplementary Results"
echo "section order: $actual_order"
[ "$actual_order" = "$expected_order" ] && pass "section order" || fail "section order (expected $expected_order)"
# R9 round 2: the Appendix (A, B) follows the References (section order above) and starts a new page.
if printf '%s\n' "$expanded" | grep -v '^[[:space:]]*%' | grep -B1 '^\\appendix$' | head -1 | grep -q '^\\clearpage$'; then
  pass "appendix starts a new page after the References (\\clearpage before \\appendix)"
else fail "appendix: \\clearpage must precede \\appendix"; fi
# Revision R9: CVPR author kit, camera-ready with page numbers, as in the reference paper.
fmt_ok=1
for want in '^\\documentclass\[10pt,twocolumn,letterpaper\]{article}$' '^\\usepackage\[pagenumbers\]{cvpr}$' \
            '^[[:space:]]*\\bibliographystyle{ieeenat_fullname}$'; do
  grep -q "$want" main.tex || { fail "CVPR format: main.tex lacks $want"; fmt_ok=0; }
done
for f in cvpr.sty ieeenat_fullname.bst; do [ -f "$f" ] || { fail "CVPR format: $f missing"; fmt_ok=0; }; done
if grep -v '^[[:space:]]*%' main.tex sections/*.tex | grep -q 'IEEEtran\|IEEEkeywords\|ORCID'; then
  fail "CVPR format: IEEEtran / keywords / ORCID left"; fmt_ok=0
fi
[ "$fmt_ok" -eq 1 ] && pass "CVPR format (article 10pt twocolumn letterpaper, cvpr [pagenumbers], ieeenat_fullname)"
# Every \ref{tab:...} / \ref{fig:...} in the main text (before \appendix) targets a float defined in the
# main text; references inside the Appendix may also target the Appendix tables (R9 round 2).
uncommented="$(printf '%s\n' "$expanded" | grep -v '^[[:space:]]*%')"
main_part="$(printf '%s\n' "$uncommented" | sed '/^\\appendix$/,$d')"
unplaced=0
for part in main all; do
  if [ "$part" = main ]; then src="$main_part"; defs="$main_part"; where="the main text"
  else src="$(printf '%s\n' "$uncommented" | sed -n '/^\\appendix$/,$p')"; defs="$uncommented"; where="the paper"; fi
  for r in $(printf '%s\n' "$src" | grep -o '\\ref{\(tab\|fig\):[A-Za-z0-9_]*}' | sed 's/\\ref{//; s/}//' | sort -u); do
    stem="${r#*:}"
    if ! printf '%s\n' "$defs" | grep -q "\\label{$r}\|orplaceholder\(\[[^]]*\]\)\?{$stem}"; then
      fail "cited float $r is not defined in $where"; unplaced=1
    fi
  done
done
[ "$unplaced" -eq 0 ] && pass "every table and figure cited in the main text is defined there; appendix citations resolve"

# Revision R8 (writing guide items 1 and 2): title at most 12 words, abstract 195 to 200 words.
# Words are whitespace-separated after removing comment lines, LaTeX commands, braces, and $;
# $2\times2$ counts as one word, a numeric range such as 1.5--9.0 as one word.
title_words="$(sed -n 's/^\\title{\(.*\)}$/\1/p' main.tex | wc -w)"
echo "title words: $title_words"
[ "$title_words" -ge 1 ] && [ "$title_words" -le 12 ] && pass "title length ($title_words words <= 12)" \
  || fail "title length ($title_words words, expected 1 to 12)"
abstract_words="$(sed -n '/\\begin{abstract}/,/\\end{abstract}/p' sections/abstract.tex | sed '1d;$d' \
  | grep -v '^[[:space:]]*%' | sed 's/\\times/x/g; s/\\[A-Za-z]*//g; s/[{}$~]//g' | wc -w)"
echo "abstract words: $abstract_words"
[ "$abstract_words" -ge 195 ] && [ "$abstract_words" -le 200 ] && pass "abstract length ($abstract_words words, 195 to 200)" \
  || fail "abstract length ($abstract_words words, expected 195 to 200)"
if printf '%s\n' "$expanded" | grep -q '^\\subsection{Implementation Details}'; then
  pass "Implementation Details is a subsection of Method"
else
  fail "Implementation Details subsection missing"
fi

# 2. Paragraph markers: English and zh-TW must list the same markers in the same order.
# §<n> = section number (§0 = Abstract); §A, §B = Appendix A and B (R9 round 2).
en_markers="$(printf '%s\n' "$expanded" | sed -n 's/^%% \[\(§[0-9AB.]* ¶[0-9]*\)\].*/\1/p')"
zh_markers="$(sed -n 's/^<!-- EN: \(§[0-9AB.]* ¶[0-9]*\) -->.*/\1/p' zh-TW/main.md)"
en_count="$(printf '%s\n' "$en_markers" | grep -c .)"
zh_count="$(printf '%s\n' "$zh_markers" | grep -c .)"
echo "paragraph markers: en=$en_count zh=$zh_count"
if [ "$en_markers" = "$zh_markers" ]; then
  pass "zh-TW markers match English markers one-to-one and in order"
else
  fail "marker mismatch"; diff <(printf '%s\n' "$en_markers") <(printf '%s\n' "$zh_markers")
fi
dups="$(printf '%s\n' "$en_markers" | sort | uniq -d)"
[ -z "$dups" ] && pass "no duplicate markers" || fail "duplicate markers: $dups"
# Every marker is followed by non-empty content in the zh-TW file.
empty_zh="$(awk '/^<!-- EN: /{m=$0; getline nxt; if (nxt ~ /^[[:space:]]*$/) print m}' zh-TW/main.md)"
[ -z "$empty_zh" ] && pass "every zh-TW marker has content" || fail "empty zh-TW paragraphs: $empty_zh"

# 3. Zero placeholders: no \pending{, TODO(14), or 待 14 補上 anywhere in the paper sources.
for f in main.tex sections/*.tex; do
  hits="$(grep -n '\\pending{\|TODO(14)\|待 14 補上' "$f")"
  if [ -n "$hits" ]; then
    fail "placeholder in $f"; printf '%s\n' "$hits"
  fi
done
zh_hits="$(awk '
  /^<!-- EN: / { m=$0; sub(/^<!-- EN: /, "", m); sub(/ -->.*/, "", m); next }
  /待 14 補上/ { print (m == "" ? "header" : m) }' zh-TW/main.md | sort -u)"
if [ -n "$zh_hits" ]; then
  while IFS= read -r m; do fail "zh-TW placeholder in paragraph $m"; done <<< "$zh_hits"
fi
[ -z "$(grep -ln '\\pending{\|TODO(14)\|待 14 補上' main.tex sections/*.tex zh-TW/main.md)" ] \
  && pass "no placeholder in main.tex, sections/*.tex, or zh-TW/main.md"

# 3b. Typography (revision R1): no TeX paired quotes (write \dq{...}), no em dash used as
#     sentence punctuation (table cells "& --- \\" are allowed), no old condition names.
typo=0
for f in main.tex sections/*.tex; do
  hits="$(grep -nE "\`\`|''|—|[A-Za-z}] ?--- ?[A-Za-z\\(]|No aid|No assistance|no-aid|no-assistance" "$f" | grep -v '^[0-9]*:[[:space:]]*%')"
  if [ -n "$hits" ]; then fail "typography in $f"; printf '%s\n' "$hits"; typo=1; fi
done
zh_typo="$(grep -n '—\|``\|No aid\|No assistance\|無輔助' zh-TW/main.md | grep -v '^[0-9]*:|' | grep -v '「—」' | grep -v '^[0-9]*:```')"   # skip table rows, the cell-symbol note, code fences
[ -n "$zh_typo" ] && { fail "typography in zh-TW/main.md"; printf '%s\n' "$zh_typo"; typo=1; }
[ "$typo" -eq 0 ] && pass "typography: no paired TeX quotes, no dash punctuation, no old condition names"

# 4. Every table/figure hook has its generated file (tables/<stem>.tex, figures/<stem>.pdf).
#    CHECK_ALLOW_MISSING=1 downgrades this to a note (self-test compiles without the files).
hooks=0; missing_files=0
for kind in table figure; do
  dir=tables; ext=tex; [ "$kind" = figure ] && { dir=figures; ext=pdf; }
  for s in $(grep -ho "\\\\\(sub\|wide\)\?${kind}orplaceholder\(\[[^]]*\]\)\?{[A-Za-z0-9_]*}" sections/*.tex | sed 's/.*{\(.*\)}/\1/'); do
    hooks=$((hooks + 1))
    if [ ! -f "$dir/$s.$ext" ]; then
      missing_files=$((missing_files + 1))
      if [ "${CHECK_ALLOW_MISSING:-0}" = 1 ]; then echo "NOTE: generated file missing: $dir/$s.$ext"
      else fail "generated file missing: $dir/$s.$ext"; fi
    fi
  done
done
[ "$missing_files" -eq 0 ] && pass "all $hooks table/figure hooks have their generated file"

# 4b. Revision R9: ruled (grid) tables as in the reference paper; no booktabs rules in the generated
#     tables or in the tables written in sections/*.tex. CHECK_GRID=0 skips this check.
if [ "${CHECK_GRID:-1}" = 1 ]; then
  grid_bad=0
  for f in tables/*.tex sections/*.tex; do
    [ -f "$f" ] || continue
    case "$f" in tables/*) grep -q "{$(basename "$f" .tex)}" sections/*.tex || continue;; esac  # used tables only
    hits="$(grep -n '\\toprule\|\\midrule\|\\bottomrule\|\\cmidrule' "$f" | grep -v '^[0-9]*:[[:space:]]*%')"
    [ -n "$hits" ] && { fail "table style in $f (booktabs rule; R9 tables are ruled grids)"; printf '%s\n' "$hits" | head -3; grid_bad=1; }
  done
  [ "$grid_bad" -eq 0 ] && pass "table style: ruled grid tables (no booktabs rules)"
fi

# 5. Bibliography: every entry has doi, eprint, or url.
missing="$(awk '
  /^@/ { if (key != "" && !ok) print key; key=$0; sub(/^@[a-zA-Z]+\{/, "", key); sub(/,.*/, "", key); ok=0; next }
  /^[[:space:]]*(doi|eprint|url)[[:space:]]*=/ { ok=1 }
  /howpublished[[:space:]]*=.*\\url\{/ { ok=1 }
  END { if (key != "" && !ok) print key }' refs.bib)"
entries="$(grep -c '^@' refs.bib)"
echo "bib entries: $entries"
[ -z "$missing" ] && pass "every bib entry has doi/eprint/url" || fail "bib entries without source: $missing"
cited="$(printf '%s\n' "$expanded" | grep -v '^[[:space:]]*%' | grep -o '\\cite{[^}]*}' | sed 's/\\cite{//; s/}//' | tr ',' '\n' | sed 's/^ *//; s/ *$//' | sort -u)"
for k in $cited; do
  grep -q "^@[a-zA-Z]*{$k," refs.bib || fail "cited key not in refs.bib: $k"
done
pass "all cited keys checked ($(printf '%s\n' "$cited" | grep -c .) keys)"

# 7. Optional compile.
if [ "${1:-}" = "--compile" ]; then
  out="${CHECK_OUTDIR:-$(mktemp -d)}"; mkdir -p "$out"
  if tectonic --keep-logs --keep-intermediates --outdir "$out" main.tex > "$out/stdout.txt" 2>&1; then
    pass "tectonic compile ($out/main.pdf)"
  else
    fail "tectonic compile"; tail -30 "$out/stdout.txt"
    echo "failures: $failures"; exit 1
  fi
  if grep -Eiq 'undefined (citation|reference)|Citation .* undefined|Reference .* undefined|There were undefined' "$out/main.log" "$out/stdout.txt" 2>/dev/null; then
    fail "undefined citations/references"; grep -Ei 'undefined' "$out/main.log" "$out/stdout.txt"
  else
    pass "no undefined citation/reference warnings"
  fi
  if grep -q 'Warning--' "$out/main.blg" 2>/dev/null; then
    echo "BibTeX warnings:"; grep 'Warning--' "$out/main.blg"
  fi
  # No TeX paired (curly) quotes in the references; ieeenat_fullname.bst (R9) prints titles
  # unquoted, as in the reference paper. Checked on the generated main.bbl, not on the .bst.
  bbl_paired="$(grep -nF -e '``' -e "''" "$out/main.bbl" 2>/dev/null)"
  if [ ! -f "$out/main.bbl" ]; then fail "main.bbl not produced"
  elif [ -n "$bbl_paired" ]; then fail "paired TeX quotes in main.bbl"; printf '%s\n' "$bbl_paired" | head -5
  else pass "main.bbl: no paired TeX quotes ($(grep -c '^\\bibitem' "$out/main.bbl") entries)"; fi
  # Every \tableorplaceholder / \(sub)figureorplaceholder call must yield a numbered, labelled float.
  for kind in table figure; do
    prefix="${kind:0:3}"; [ "$kind" = figure ] && prefix=fig
    stems="$(grep -ho "\\\\\(sub\|wide\)\?${kind}orplaceholder\(\[[^]]*\]\)\?{[A-Za-z0-9_]*}" sections/*.tex | sed 's/.*{\(.*\)}/\1/')"
    n=0; missing_labels=""
    for s in $stems; do
      n=$((n + 1))
      grep -q "\\\\newlabel{$prefix:$s}" "$out/main.aux" || missing_labels="$missing_labels $s"
    done
    [ -z "$missing_labels" ] && pass "$n ${kind}(s) have caption + label $prefix:<stem> in main.aux" \
      || fail "${kind}s without label in main.aux:$missing_labels"
  done
  pages="$(sed -n 's/.*Output written on main.xdv (\([0-9]*\) pages.*/\1/p' "$out/main.log")"
  echo "pages: $pages (main text + References + Appendix)"
  # R9 round 2: main text + References = exactly PAGE_LIMIT (default 8) pages with the last one filled,
  # then the Appendix from the next page (any length). main.tex writes \label{refs-end} and
  # \zsavepos{refs-end} at the end of the last reference; cvpr.sty on letter paper puts the bottom of
  # the text block 81pt (11in - 1in - 8.875in) above the page bottom, and a reference line is 12pt.
  limit="${PAGE_LIMIT:-8}"
  refs_page="$(sed -n 's/^\\newlabel{refs-end}{{[^}]*}{\([0-9]*\)}.*/\1/p' "$out/main.aux")"
  app_page="$(sed -n 's/^\\newlabel{sec:appendix}{{[^}]*}{\([0-9]*\)}.*/\1/p' "$out/main.aux")"
  read -r posx posy <<< "$(sed -n 's/^\\zref@newlabel{refs-end}{\\posx{\([0-9]*\)}\\posy{\([0-9]*\)}}.*/\1 \2/p' "$out/main.aux")"
  if [ "$missing_files" -gt 0 ]; then
    echo "NOTE: page layout not checked ($missing_files generated files missing; References end on page ${refs_page:-?}, Appendix on page ${app_page:-?})"
  else
    echo "References end on page ${refs_page:-?}; Appendix starts on page ${app_page:-?}"
    if [ "$refs_page" = "$limit" ]; then pass "page budget: References end on page $limit"
    else fail "page budget: References end on page ${refs_page:-?} (must end on page $limit)"; fi
    if [ "$app_page" = "$((limit + 1))" ]; then pass "Appendix starts on page $((limit + 1))"
    else fail "appendix starts on page ${app_page:-?} (must start on page $((limit + 1)))"; fi
    if [ -n "${posy:-}" ]; then
      left_lines="$(awk -v x="$posx" -v y="$posy" 'BEGIN { if (x / 65536 < 306) print 99; else printf "%.1f", (y / 65536 - 81) / 12 }')"
      echo "lines left below the last reference on page ${refs_page:-?}: $left_lines (99 = References end in the left column)"
      if awk -v l="$left_lines" 'BEGIN { exit !(l <= 3) }'; then pass "page $limit filled (last reference line, right column, $left_lines lines left <= 3)"
      else fail "page $limit not filled ($left_lines lines left below the last reference; at most 3)"; fi
    else fail "refs-end position missing in main.aux"; fi
  fi
  overfull="$(grep -c '^Overfull \\hbox' "$out/main.log")"
  if [ "$overfull" -eq 0 ]; then pass "no Overfull \\hbox"
  else fail "Overfull \\hbox: $overfull"; grep '^Overfull \\hbox' "$out/main.log"; fi
fi

echo "failures: $failures"
[ "$failures" -eq 0 ]
