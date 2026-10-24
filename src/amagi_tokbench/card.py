"""Turn a results dict into the shareable card: markdown, plain text and PNG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

REPO_URL = "https://github.com/amagilabs/amagi-tokbench"
_VENDOR = Path(__file__).parent / "vendor"
FONT_FILE = _VENDOR / "fonts" / "Montserrat-VF.ttf"
ICON_FILE = _VENDOR / "amagilabs-icon-128.png"

OFF, INK, CORAL, GOLD, MUTED, WHITE, LINE = (
    "#F6F4F1", "#161616", "#D27A73", "#E6C89C", "#8A8580", "#FFFFFF", "#DDD8D2",
)


# --------------------------------------------------------------------------- formatting


def fmt_num(v, digits=0) -> str:
    if v is None:
        return "-"
    return f"{v:,.{digits}f}"


def fmt_rate(v) -> str:
    """tok/s: one decimal under 1,000, none above."""
    if v is None:
        return "-"
    return f"{v:,.1f}" if v < 1000 else f"{v:,.0f}"


def fmt_ms(v) -> str:
    if v is None:
        return "-"
    return f"{v:,.0f} ms" if v < 10000 else f"{v / 1000:,.1f} s"


def fmt_gib(mib) -> str:
    return "-" if mib is None else f"{mib / 1024:.1f} GiB"


def gpu_text(vram: Optional[dict]) -> Optional[str]:
    if not vram or not vram.get("gpus"):
        return None
    names = [re.sub(r"^NVIDIA\s+", "", g.get("name") or "GPU") for g in vram["gpus"]]
    if len(set(names)) == 1:
        return names[0] if len(names) == 1 else f"{len(names)}x {names[0]}"
    return " + ".join(names)


def model_text(result: dict) -> str:
    b = result.get("backend") or {}
    name = b.get("model_file") or b.get("model") or "unknown model"
    if b.get("model_file"):
        name = re.split(r"[\\/]", name)[-1]
    return re.sub(r"\.gguf$", "", name, flags=re.I)


def backend_text(result: dict) -> str:
    b = result.get("backend") or {}
    name = b.get("name") or b.get("kind") or "server"
    return f"{name} {b['version']}" if b.get("version") else name


def subtitle(result: dict, sep: str = " · ") -> str:
    b = result.get("backend") or {}
    parts = [backend_text(result)]
    model = model_text(result)
    if b.get("quant") and b["quant"].lower() not in model.lower():
        parts.append(b["quant"])
    if b.get("params") and b["params"].lower() not in model.lower():
        parts.append(b["params"])
    gpu = gpu_text(result.get("vram"))
    if gpu:
        parts.append(gpu)
    return sep.join(parts)


def headline_items(result: dict):
    s = result.get("summary") or {}
    vram = result.get("vram") or {}
    return [
        ("Generation", fmt_rate(s.get("gen_tps")), "tok/s", f"{s.get('gen_case') or '-'} case"),
        ("Prompt processing", fmt_rate(s.get("pp_tps")), "tok/s", f"{s.get('pp_case') or '-'} case"),
        ("Time to first token", fmt_num(s.get("ttft_ms")), "ms", f"{s.get('gen_case') or '-'} case"),
        ("Peak VRAM", fmt_gib(vram.get("peak_mib")).replace(" GiB", "") if vram.get("peak_mib") else "n/a",
         "GiB" if vram.get("peak_mib") else "",
         f"of {fmt_gib(vram.get('total_mib'))}" if vram.get("total_mib") else "not measured"),
    ]


def method_line(result: dict) -> str:
    st = result.get("settings") or {}
    return (
        f"Median of {st.get('runs', '?')} runs per case after {st.get('warmup', 0)} warmup. "
        f"Prompt set {st.get('prompt_set', '?')}, temperature 0, timed client-side on the stream."
    )


def stamp_line(result: dict) -> str:
    date = (result.get("created") or "")[:10]
    return f"amagi-tokbench {result.get('tool_version', '?')}, {date}."


def _has_server(cases) -> bool:
    return any(c["median"].get("server_gen_tps") or c["median"].get("server_pp_tps") for c in cases)


def case_rows(result: dict, with_server: Optional[bool] = None) -> List[List[str]]:
    cases = result.get("cases") or []
    if with_server is None:
        with_server = _has_server(cases)
    rows = []
    for c in cases:
        m = c["median"]
        row = [
            c["name"], fmt_num(m["prompt_tokens"]), fmt_num(m["gen_tokens"]), fmt_ms(m["ttft_ms"]),
            fmt_rate(m["pp_tps"]), fmt_rate(m["gen_tps"]),
        ]
        if with_server:
            row += [fmt_rate(m.get("server_pp_tps")), fmt_rate(m.get("server_gen_tps"))]
        rows.append(row)
    return rows


CASE_HEADERS = ["case", "prompt tokens", "generated", "TTFT", "prompt tok/s", "gen tok/s"]
SERVER_HEADERS = ["server prompt tok/s", "server gen tok/s"]


# --------------------------------------------------------------------------- markdown / text


def _md_table(headers, rows, left=1) -> List[str]:
    aligns = ["---" if i < left else "--:" for i in range(len(headers))]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def to_markdown(result: dict) -> str:
    items = headline_items(result)
    with_server = _has_server(result.get("cases") or [])
    headers = CASE_HEADERS + (SERVER_HEADERS if with_server else [])
    lines = [f"### {model_text(result)}", "", subtitle(result)]
    if result.get("label"):
        lines += ["", result["label"]]
    lines += [
        "",
        " · ".join(f"**{v}{(' ' + u) if u else ''}** {k.lower()}" for k, v, u, _ in items),
        "",
    ]
    lines += _md_table(headers, case_rows(result, with_server))
    lines += ["", f"*{method_line(result)} {stamp_line(result)}* [{REPO_URL.replace('https://', '')}]({REPO_URL})"]
    for w in result.get("warnings") or []:
        lines += ["", f"Note: {w}"]
    return "\n".join(lines) + "\n"


def _text_table(headers, rows) -> List[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    def fmt(cells):
        return "  ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))
    return [fmt(headers)] + [fmt(r) for r in rows]


def to_text(result: dict) -> str:
    items = headline_items(result)
    lines = [model_text(result), subtitle(result, " | ")]
    if result.get("label"):
        lines.append(result["label"])
    lines.append("")
    for k, v, u, note in items:
        lines.append(f"  {k:<20} {v:>9} {u:<5}  ({note})")
    lines.append("")
    with_server = _has_server(result.get("cases") or [])
    lines += _text_table(CASE_HEADERS + (SERVER_HEADERS if with_server else []), case_rows(result, with_server))
    for w in result.get("warnings") or []:
        lines.append(f"note: {w}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- compare


def compare_rows(results: List[dict]) -> List[List[str]]:
    rows = []
    for r in results:
        s = r.get("summary") or {}
        b = r.get("backend") or {}
        rows.append([
            r.get("label") or model_text(r),
            backend_text(r),
            b.get("quant") or "-",
            fmt_rate(s.get("gen_tps")),
            fmt_rate(s.get("pp_tps")),
            fmt_ms(s.get("ttft_ms")),
            fmt_gib((r.get("vram") or {}).get("peak_mib")),
        ])
    return rows


COMPARE_HEADERS = ["run", "backend", "quant", "gen tok/s", "prompt tok/s", "TTFT", "peak VRAM"]


def compare_markdown(results: List[dict]) -> str:
    sets = sorted({(r.get("settings") or {}).get("prompt_set", "?") for r in results})
    lines = [f"### amagi-tokbench: {len(results)} runs side by side", ""]
    lines += _md_table(COMPARE_HEADERS, compare_rows(results), left=3)
    gpus = sorted({gpu_text(r.get("vram")) for r in results} - {None})
    lines.append("")
    if gpus:
        lines.append("GPU: " + ", ".join(gpus) + ".")
    lines.append(
        f"*Prompt set {', '.join(sets)}; each number is a median from its own run.* "
        f"[{REPO_URL.replace('https://', '')}]({REPO_URL})"
    )
    if len(sets) > 1:
        lines += ["", "Note: these runs used different prompt sets, so they are not directly comparable."]
    return "\n".join(lines) + "\n"


def compare_text(results: List[dict]) -> str:
    return "\n".join(_text_table(COMPARE_HEADERS, compare_rows(results))) + "\n"


# --------------------------------------------------------------------------- PNG


class _Fonts:
    def __init__(self):
        from PIL import ImageFont  # noqa: F401  (fail early if Pillow is missing)

        self._cache = {}

    def get(self, size: int, weight: int = 400):
        key = (size, weight)
        if key not in self._cache:
            from PIL import ImageFont

            try:
                font = ImageFont.truetype(str(FONT_FILE), size)
                try:
                    font.set_variation_by_axes([weight])
                except Exception:
                    pass
            except Exception:
                font = ImageFont.load_default(size=size)
            self._cache[key] = font
        return self._cache[key]


def _fit(draw, text, font, width) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _spaced(draw, xy, text, font, fill, spacing=2.0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing
    return x


def _footer(img, draw, fonts, text, W, H, M):
    y = H - 54
    draw.line([(M, y - 16), (W - M, y - 16)], fill=LINE, width=1)
    right = REPO_URL.replace("https://", "")
    f = fonts.get(15, 500)
    rw = draw.textlength(right, font=f)
    x_icon = W - M - rw - 34
    try:
        from PIL import Image

        icon = Image.open(ICON_FILE).convert("RGBA").resize((26, 26), Image.LANCZOS)
        img.paste(icon, (int(x_icon), y - 3), icon)
    except Exception:
        x_icon = W - M - rw - 4
    draw.text((W - M - rw, y), right, font=f, fill=INK)
    lines = [text] if isinstance(text, str) else list(text)
    small = fonts.get(13, 400)
    top = y + 9 - 9 * len(lines)
    for i, line in enumerate(lines):
        draw.text((M, top + 18 * i), _fit(draw, line, small, x_icon - M - 24), font=small, fill=MUTED)


def _table(draw, fonts, x0, y, width, headers, rows, left_cols=1, row_h=40, first_bold=True):
    n = len(headers)
    hf, cf, bf = fonts.get(13, 600), fonts.get(19, 400), fonts.get(19, 600)
    # first column takes what's left after fitting the numeric ones
    col_w = [max(draw.textlength(h.upper(), font=hf) + 6 * len(h),
                 *(draw.textlength(r[i], font=cf) for r in rows)) + 28 for i, h in enumerate(headers)]
    fixed = sum(col_w[left_cols:])
    if left_cols:
        rest = max(width - fixed, 160)
        base = sum(col_w[:left_cols])
        col_w[:left_cols] = [w * rest / base for w in col_w[:left_cols]]
    scale = width / sum(col_w)
    col_w = [w * scale for w in col_w]
    xs = [x0]
    for w in col_w[:-1]:
        xs.append(xs[-1] + w)
    for i, h in enumerate(headers):
        label = h.upper()
        tw = draw.textlength(label, font=hf) + 1.5 * len(label)
        x = xs[i] if i < left_cols else xs[i] + col_w[i] - tw - 4
        _spaced(draw, (x, y), label, hf, CORAL, 1.5)
    y += 30
    draw.line([(x0, y - 6), (x0 + width, y - 6)], fill=INK, width=2)
    for r in rows:
        for i, cell in enumerate(r):
            font = bf if (i == 0 and first_bold) else cf
            if i < left_cols:
                cell = _fit(draw, cell, font, col_w[i] - 16)
                draw.text((xs[i], y + 8), cell, font=font, fill=INK)
            else:
                tw = draw.textlength(cell, font=font)
                draw.text((xs[i] + col_w[i] - tw - 4, y + 8), cell, font=font, fill=INK)
        y += row_h
        draw.line([(x0, y), (x0 + width, y)], fill=LINE, width=1)
    return y


def render_png(result: dict, path) -> Path:
    from PIL import Image, ImageDraw

    W, M = 1200, 56
    cases = result.get("cases") or []
    with_server = _has_server(cases)
    H = max(675, 470 + 40 * len(cases) + (28 if result.get("label") else 0) + 90)
    img = Image.new("RGB", (W, H), OFF)
    draw = ImageDraw.Draw(img)
    fonts = _Fonts()

    y = M - 6
    _spaced(draw, (M, y), "AMAGI-TOKBENCH  ·  LOCAL LLM SPEED CARD", fonts.get(15, 600), CORAL, 2.2)
    y += 34
    tf = fonts.get(42, 700)
    draw.text((M, y), _fit(draw, model_text(result), tf, W - 2 * M), font=tf, fill=INK)
    y += 58
    sf = fonts.get(21, 500)
    draw.text((M, y), _fit(draw, subtitle(result), sf, W - 2 * M), font=sf, fill=MUTED)
    y += 32
    if result.get("label"):
        lf = fonts.get(19, 400)
        draw.text((M, y), _fit(draw, result["label"], lf, W - 2 * M), font=lf, fill=INK)
        y += 28
    y += 18

    gap, n = 20, 4
    tw = (W - 2 * M - gap * (n - 1)) / n
    th = 152
    for i, (k, v, u, note) in enumerate(headline_items(result)):
        x = M + i * (tw + gap)
        dark = i == 0
        draw.rounded_rectangle([x, y, x + tw, y + th], radius=16, fill=INK if dark else WHITE,
                               outline=INK, width=2)
        fg = OFF if dark else INK
        _spaced(draw, (x + 20, y + 18), k.upper(), fonts.get(13, 600), CORAL if dark else MUTED, 1.6)
        vf = fonts.get(50, 700)
        draw.text((x + 18, y + 44), v, font=vf, fill=fg)
        vw = draw.textlength(v, font=vf)
        if u:
            draw.text((x + 26 + vw, y + 70), u, font=fonts.get(19, 600), fill=GOLD if dark else CORAL)
        draw.text((x + 20, y + th - 34), note, font=fonts.get(14, 400), fill=GOLD if dark else MUTED)
    y += th + 34

    headers = CASE_HEADERS + (SERVER_HEADERS if with_server else [])
    if with_server:
        headers = [h.replace("server prompt tok/s", "server pp").replace("server gen tok/s", "server gen")
                   for h in headers]
    _table(draw, fonts, M, y, W - 2 * M, headers, case_rows(result, with_server))

    second = stamp_line(result)
    if (result.get("vram") or {}).get("peak_mib"):
        second = "VRAM = peak memory.used on the GPU over the whole run. " + second
    _footer(img, draw, fonts, [method_line(result), second], W, H, M)
    path = Path(path)
    img.save(path, optimize=True)
    return path


def render_compare_png(results: List[dict], path) -> Path:
    from PIL import Image, ImageDraw

    W, M = 1200, 56
    H = max(675, 300 + 44 * len(results) + 90)
    img = Image.new("RGB", (W, H), OFF)
    draw = ImageDraw.Draw(img)
    fonts = _Fonts()
    y = M - 6
    _spaced(draw, (M, y), "AMAGI-TOKBENCH  ·  SIDE BY SIDE", fonts.get(15, 600), CORAL, 2.2)
    y += 34
    draw.text((M, y), f"{len(results)} runs, same prompt set", font=fonts.get(42, 700), fill=INK)
    y += 58
    gpus = sorted({gpu_text(r.get("vram")) for r in results} - {None})
    if gpus:
        draw.text((M, y), _fit(draw, "GPU: " + ", ".join(gpus), fonts.get(21, 500), W - 2 * M),
                  font=fonts.get(21, 500), fill=MUTED)
    y += 56

    rows = compare_rows(results)
    end = _table(draw, fonts, M, y, W - 2 * M, COMPARE_HEADERS, rows, left_cols=3, row_h=44)
    # generation-speed bars under the table
    rates = [(r.get("summary") or {}).get("gen_tps") or 0 for r in results]
    top = max(rates) if rates and max(rates) > 0 else 0
    if top and end + 40 + 26 * len(results) < H - 80:
        y = end + 30
        _spaced(draw, (M, y), "GENERATION TOK/S", fonts.get(13, 600), CORAL, 1.5)
        y += 26
        lab_w = 300
        for r, rate in zip(results, rates):
            name = _fit(draw, r.get("label") or model_text(r), fonts.get(15, 500), lab_w - 10)
            draw.text((M, y), name, font=fonts.get(15, 500), fill=INK)
            bw = (W - 2 * M - lab_w - 90) * rate / top
            draw.rounded_rectangle([M + lab_w, y + 2, M + lab_w + max(bw, 4), y + 18], radius=6, fill=CORAL)
            draw.text((M + lab_w + bw + 10, y), fmt_rate(rate), font=fonts.get(15, 600), fill=INK)
            y += 26
    sets = sorted({(r.get("settings") or {}).get("prompt_set", "?") for r in results})
    _footer(img, draw, fonts, f"Prompt set {', '.join(sets)}. Each number is the median of its own run.", W, H, M)
    path = Path(path)
    img.save(path, optimize=True)
    return path
