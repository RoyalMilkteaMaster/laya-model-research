#!/usr/bin/env python3
"""把 paper/zh-TW/main.md 轉成繁體中文 PDF（paper/zh-TW/main.pdf）。

流程：Markdown → HTML（含 paper/figures/*.pdf 轉成的 PNG 插圖）→ Windows Edge headless 列印成 PDF。
需求：Python 套件 `markdown`、`pymupdf`（任一 venv 皆可，例如 `uv venv .pdfenv && uv pip install markdown pymupdf`）；
WSL 可呼叫 Windows 端 Edge（預設路徑見 EDGE）。字型使用 Windows 的 Noto Sans TC／微軟正黑體。
用法（於 Code Root）：python paper/zh-TW/build_pdf.py
"""
from __future__ import annotations
import html, pathlib, re, subprocess, sys, time
import markdown
try:
    import pymupdf as fitz
except ImportError:  # 舊版套件名稱
    import fitz

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZH = ROOT / "paper" / "zh-TW"
FIG_DIR = ZH / "figures"
EDGE = pathlib.Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
FIGURES = {  # 中文版「圖 N」→ paper/figures 的檔名
    1: "em_vs_size_hotpotqa", 2: "em_vs_size_2wiki", 3: "em_vs_size_musique",
    4: "latency_vs_size", 5: "hops_distribution",
}

def render_figures() -> dict[int, pathlib.Path]:
    FIG_DIR.mkdir(exist_ok=True)
    out = {}
    for n, stem in FIGURES.items():
        src = ROOT / "paper" / "figures" / f"{stem}.pdf"
        dst = FIG_DIR / f"{stem}.png"
        if src.exists():
            doc = fitz.open(src)
            doc[0].get_pixmap(dpi=170).save(dst)
            out[n] = dst
        else:
            print(f"警告：找不到 {src}，圖 {n} 將只保留說明文字", file=sys.stderr)
    return out

def insert_figures(md: str, figs: dict[int, pathlib.Path]) -> str:
    lines = md.splitlines()
    result = []
    for line in lines:
        m = re.match(r"^\*\*圖 (\d+)：", line)
        if m and int(m.group(1)) in figs:
            png = figs[int(m.group(1))]
            result.append(f'<p class="figure"><img src="figures/{png.name}" alt="圖 {m.group(1)}"></p>')
            result.append("")
        result.append(line)
    return "\n".join(result)

def build_html(md: str) -> str:
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    css = """
    @page { size: A4; margin: 18mm 16mm; }
    body { font-family: "Noto Sans TC", "Microsoft JhengHei", "微軟正黑體", sans-serif; font-size: 10.5pt; line-height: 1.55; color: #111; max-width: 180mm; margin: 0 auto; }
    h1 { font-size: 17pt; line-height: 1.35; margin: 0 0 8pt; }
    h2 { font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; page-break-after: avoid; }
    h3 { font-size: 11.5pt; margin: 12pt 0 4pt; page-break-after: avoid; }
    p { margin: 0 0 6pt; text-align: justify; }
    table { border-collapse: collapse; margin: 6pt auto 8pt; font-size: 8.5pt; page-break-inside: avoid; }
    th, td { border: 1px solid #888; padding: 2pt 5pt; text-align: center; white-space: nowrap; }
    th { background: #eee; }
    code { font-family: Consolas, "Courier New", monospace; font-size: 0.92em; }
    pre { font-size: 8.5pt; background: #f4f4f4; padding: 6pt; white-space: pre-wrap; }
    p.figure { text-align: center; page-break-inside: avoid; margin: 8pt 0 2pt; }
    p.figure img { max-width: 100%; max-height: 95mm; }
    blockquote { margin: 0 0 6pt 12pt; color: #333; }
    """
    return f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>論文繁體中文對照版</title><style>{css}</style></head><body>{body}</body></html>'

def main() -> int:
    md = (ZH / "main.md").read_text(encoding="utf-8")
    figs = render_figures()
    html_path = ZH / "main.html"
    html_path.write_text(build_html(insert_figures(md, figs)), encoding="utf-8")
    pdf_path = ZH / "main.pdf"
    win_html = subprocess.run(["wslpath", "-w", str(html_path)], capture_output=True, text=True, check=True).stdout.strip()
    win_pdf = subprocess.run(["wslpath", "-w", str(pdf_path)], capture_output=True, text=True, check=True).stdout.strip()
    if not EDGE.exists():
        print(f"找不到 Edge：{EDGE}；HTML 已輸出到 {html_path}，請自行列印成 PDF", file=sys.stderr)
        return 1
    if pdf_path.exists():
        pdf_path.unlink()
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={win_pdf}", "file:///" + win_html.replace("\\", "/")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            break
        time.sleep(1)
    if not pdf_path.exists():
        print("Edge 沒有產生 PDF", file=sys.stderr)
        return 1
    pages = len(fitz.open(pdf_path))
    print(f"已輸出 {pdf_path}（{pages} 頁，{pdf_path.stat().st_size // 1024} KB）；HTML：{html_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
