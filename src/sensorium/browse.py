"""Build static HTML pages to review generated episodes by eye and ear.

Usage:
    sensorium-browse data/sample        # writes data/sample/index.html

For every episode it writes a page with:
* the video as stored (256 px; training crops and resizes it at load time,
  see ``vision.py``) with its synthesised sound,
* line charts (<= 12 channels) or heatmaps (more channels) of proprio, touch,
  actions and the audio log-mel features, with a cursor that follows the video,
* the step-by-step descriptions (language-probe labels); click one to seek,
* the episode's metadata (seed, policy, source data, quality checks).

Pages are plain HTML + a few lines of JS, no server needed: open index.html.
"""

import html
import json
import logging
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from sensorium.episode import SAMPLE_RATE, Episode

log = logging.getLogger("sensorium.browse")

VIDEO_SIZE = 256  # px; smaller frames (older samples) are upscaled with nearest-neighbour
MAX_LINES = 12  # channels above this are drawn as a heatmap
CHART_W, CHART_H = 760, 120
PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2",
           "#97bbf5", "#9c6b4e", "#9498a0", "#1f77b4", "#d62728"]  # fmt: skip


DYNAMICS = "dynamics/"  # prefix of the touch dynamics channels (see bodies/h1_skin.py)


def write_wav(path: Path, audio: np.ndarray) -> None:
    """Write ``[T, n]`` int16 chunks as one mono 16 kHz WAV file."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.reshape(-1).tobytes())


def encode_video(episode: Episode, path: Path) -> None:
    """Encode frames + audio as H.264/AAC MP4 with the bundled ffmpeg.

    Raises:
        subprocess.CalledProcessError: if ffmpeg fails.
    """
    wav_path = path.with_suffix(".wav")
    write_wav(wav_path, episode.audio)
    fps = round(1.0 / episode.dt, 6)
    size = episode.rgb.shape[1]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
        "-i", str(wav_path),
        "-vf", f"scale={VIDEO_SIZE}:{VIDEO_SIZE}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ]  # fmt: skip
    subprocess.run(command, input=episode.rgb.tobytes(), check=True)
    wav_path.unlink()


def line_chart(values: np.ndarray, names: list[str], title: str) -> str:
    """Inline SVG with all channels overlaid on one shared y range.

    A shared range keeps magnitudes comparable; it is printed on the left.
    """
    steps = values.shape[0]
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo or 1.0
    x = np.linspace(0, CHART_W, steps)
    paths = []
    for i in range(values.shape[1]):
        y = CHART_H - (values[:, i] - lo) / span * CHART_H
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y, strict=True))
        color = PALETTE[i % len(PALETTE)]
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2">'
            f"<title>{html.escape(names[i])}</title></polyline>"
        )
    legend = " ".join(
        f'<span style="color:{PALETTE[i % len(PALETTE)]}">■ {html.escape(n)}</span>'
        for i, n in enumerate(names)
    )
    return (
        f'<div class="chart"><h3>{html.escape(title)}</h3>'
        f'<div class="range">{hi:.3g}<br><br><br>{lo:.3g}</div>'
        f'<svg viewBox="0 0 {CHART_W} {CHART_H}" preserveAspectRatio="none">{"".join(paths)}'
        f'<line class="cursor" x1="0" x2="0" y1="0" y2="{CHART_H}"/></svg>'
        f'<div class="legend">{legend}</div></div>'
    )


def heatmap(values: np.ndarray, names: list[str], title: str, png_path: Path) -> str:
    """Write a channels-x-time heatmap PNG (each row scaled to its own range).

    Returns the HTML block (image + cursor overlay + channel list).
    """
    v = values.astype(np.float32).T  # [channels, T]
    lo, hi = v.min(axis=1, keepdims=True), v.max(axis=1, keepdims=True)
    scaled = (v - lo) / np.where(hi > lo, hi - lo, 1.0)
    rgb = np.stack([scaled**0.8, scaled**1.5 * 0.85, 0.35 + 0.3 * (1 - scaled)], axis=-1) * 255
    image = Image.fromarray(rgb.astype(np.uint8)[::-1]).resize(
        (CHART_W, max(CHART_H, 3 * v.shape[0])), Image.Resampling.NEAREST
    )
    image.save(png_path)
    rows = ", ".join(html.escape(n) for n in names)
    return (
        f'<div class="chart"><h3>{html.escape(title)} '
        f"<small>({len(names)} channels, each scaled to its own range; top = last)</small></h3>"
        f'<div class="heat"><img src="{png_path.name}"><svg viewBox="0 0 {CHART_W} 1" '
        f'preserveAspectRatio="none"><line class="cursor" x1="0" x2="0" y1="0" y2="1"/></svg></div>'
        f"<details><summary>channels</summary>{rows}</details></div>"
    )


def chart(values: np.ndarray, names: list[str], title: str, png_path: Path) -> str:
    """Line chart for few channels, heatmap for many, nothing for none."""
    if values.shape[1] == 0:
        return f'<div class="chart"><h3>{html.escape(title)}</h3><p class="none">none</p></div>'
    if values.shape[1] <= MAX_LINES:
        return line_chart(values.astype(np.float32), names, title)
    return heatmap(values, names, title, png_path)


def touch_charts(episode: Episode, out: Path) -> list[str]:
    """Touch blocks: one chart, or, with dense skin, per-region summaries.

    Dense skin has thousands of channels, too many rows to read. Its force
    channels (``<region>/<k>``) are summed per region, and the ``dynamics/``
    channels get one heatmap per feature (onset, slip, vibration).
    """
    names = episode.touch_names
    if not any(n.startswith(DYNAMICS) for n in names):
        return [chart(episode.touch, names, "touch (contact force)", out / "touch.png")]
    force = [i for i, n in enumerate(names) if not n.startswith(DYNAMICS)]
    regions = list(dict.fromkeys(names[i].rsplit("/", 1)[0] for i in force))
    column = {region: r for r, region in enumerate(regions)}
    summed = np.zeros((episode.length, len(regions)), dtype=np.float32)
    for i in force:
        summed[:, column[names[i].rsplit("/", 1)[0]]] += episode.touch[:, i]
    blocks = [chart(summed, regions, "touch: force per skin region (N, summed)", out / "touch.png")]
    for feature in ("onset", "slip", "vibration"):
        picked = [i for i, n in enumerate(names) if n.startswith(DYNAMICS) and n.endswith(feature)]
        labels = [names[i].removeprefix(DYNAMICS).rsplit("/", 1)[0] for i in picked]
        title = f"touch: {feature} per skin region"
        blocks.append(chart(episode.touch[:, picked], labels, title, out / f"touch_{feature}.png"))
    return blocks


def step_list(episode: Episode) -> str:
    """Descriptions, listed only when they change; each entry seeks the video."""
    items, previous = [], None
    for t, text in enumerate(episode.text):
        if text and text != previous:
            seconds = t * episode.dt
            items.append(
                f'<li data-t="{seconds:.3f}"><code>{seconds:6.2f}s</code> {html.escape(text)}</li>'
            )
        previous = text
    return f'<ol class="steps">{"".join(items)}</ol>'


# Page style and the video-sync script live next to this module.
PAGE_STYLE = (Path(__file__).parent / "review.css").read_text()
PAGE_SCRIPT = (Path(__file__).parent / "review.js").read_text()


def policy_label(entry: dict[str, object]) -> str:
    """``"policy"``, or ``"variant/policy"`` for bodies generated in named variants."""
    variant = entry.get("variant")
    return f"{variant}/{entry['policy']}" if variant else str(entry["policy"])


def episode_page(episode: Episode, entry: dict[str, object], out: Path) -> None:
    """Write ``<out>/index.html`` (+ video, heatmaps) for one episode."""
    out.mkdir(parents=True, exist_ok=True)
    encode_video(episode, out / "video.mp4")
    mel_names = [f"mel {i}" for i in range(episode.audio_logmel.shape[1])]
    blocks = [
        chart(episode.proprio, episode.proprio_names, "proprioception", out / "proprio.png"),
        *touch_charts(episode, out),
        chart(episode.action, episode.action_names, "actions [-1, 1]", out / "action.png"),
        heatmap(
            episode.audio_logmel.astype(np.float32), mel_names, "audio log-mel", out / "mel.png"
        ),
    ]
    title = f"{entry['body']} · {policy_label(entry)} · #{entry['index']}"
    meta = json.dumps({**entry, "meta": episode.meta}, indent=2, default=str)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{PAGE_STYLE}</style></head><body>
<p><a href="../../index.html">← all episodes</a></p>
<h1>{html.escape(title)}</h1>
<p>{episode.length} steps × {episode.dt:g} s = {episode.length * episode.dt:.1f} s ·
end: {html.escape(str(episode.meta.get("end_reason")))} · seed {entry["seed"]}</p>
<div class="top"><div><video controls preload="auto" src="video.mp4"></video>
<div class="now"></div></div>{step_list(episode)}</div>
{"".join(blocks)}
<h2>Metadata</h2><pre>{html.escape(meta)}</pre>
<script>{PAGE_SCRIPT}</script></body></html>"""
    (out / "index.html").write_text(page)


def channels(episodes: list[tuple[dict[str, object], Episode]], field: str) -> str:
    """Channel count of ``field`` over the episodes, ``"248/448"`` if variants differ."""
    counts = sorted({getattr(ep, field).shape[1] for _, ep in episodes})
    return "/".join(str(count) for count in counts)


def body_summary(episodes: list[tuple[dict[str, object], Episode]]) -> str:
    """One table row per body: episodes, total time, policies, steps."""
    rows = []
    for body in sorted({str(entry["body"]) for entry, _ in episodes}):
        mine = [(e, ep) for e, ep in episodes if e["body"] == body]
        seconds = sum(ep.length * ep.dt for _, ep in mine)
        policies = sorted({policy_label(e) for e, _ in mine})
        senses = (
            f"rgb {mine[0][1].rgb.shape[1]}², audio, proprio {channels(mine, 'proprio')}, "
            f"touch {channels(mine, 'touch')}, action {channels(mine, 'action')}"
        )
        rows.append(
            f"<tr><td><b>{body}</b></td><td>{len(mine)}</td><td>{seconds:.0f} s</td>"
            f"<td>{', '.join(policies)}</td><td>{1 / mine[0][1].dt:g} Hz</td><td>{senses}</td></tr>"
        )
    return (
        "<table><tr><th>body</th><th>episodes</th><th>total</th><th>policies</th>"
        f"<th>control</th><th>senses</th></tr>{''.join(rows)}</table>"
    )


def build(data_dir: Path) -> Path:
    """Build pages for every episode in ``data_dir/manifest.json``; return index path.

    Raises:
        FileNotFoundError: if the manifest is missing (run sensorium-generate first).
    """
    manifest = json.loads((data_dir / "manifest.json").read_text())
    pages_dir = data_dir / "pages"
    episodes, rows = [], []
    for entry in manifest["episodes"]:
        episode = Episode.load(data_dir / entry["file"])
        episodes.append((entry, episode))
        slug = Path(entry["file"]).stem
        episode_page(episode, entry, pages_dir / slug)
        Image.fromarray(episode.rgb[episode.length // 2]).save(pages_dir / slug / "thumb.png")
        rows.append(
            f'<tr><td><a href="pages/{slug}/index.html"><img class="thumb" '
            f'src="pages/{slug}/thumb.png"></a></td><td>{entry["body"]}</td>'
            f"<td>{policy_label(entry)}</td>"
            f'<td><a href="pages/{slug}/index.html">#{entry["index"]}</a></td>'
            f"<td>{episode.length}</td><td>{episode.length * episode.dt:.1f} s</td>"
            f"<td>{html.escape(str(entry['end_reason']))}</td><td>{entry['seed']}</td>"
            f"<td><code>{entry['content_hash']}</code></td></tr>"
        )
        log.info("page %d/%d: %s", len(rows), len(manifest["episodes"]), slug)
    prov = manifest["provenance"]
    dirty = " (uncommitted changes)" if prov["git_dirty"] else ""
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sensorium sample</title><style>{PAGE_STYLE}</style></head><body>
<h1>Sensorium · E1 input data sample</h1>
<p>Config <code>{html.escape(manifest["config_file"])}</code> · master seed
{manifest["config"]["master_seed"]} · commit <code>{prov["git_commit"][:10]}</code>{dirty} ·
Python {prov["python"]} · {html.escape(prov["platform"])}</p>
<h2>Bodies</h2>{body_summary(episodes)}
<h2>Episodes</h2><p>Click an episode to watch and listen to what the model receives.</p>
<table><tr><th></th><th>body</th><th>policy</th><th>episode</th><th>steps</th><th>duration</th>
<th>end</th><th>seed</th><th>content hash</th></tr>{"".join(rows)}</table>
<h2>Provenance</h2><pre>{html.escape(json.dumps(prov, indent=2))}</pre></body></html>"""
    index_path = data_dir / "index.html"
    index_path.write_text(index)
    log.info("open %s", index_path)
    return index_path


def main() -> None:
    """CLI entry point: ``sensorium-browse <data_dir>``."""
    if len(sys.argv) != 2:
        sys.exit("usage: sensorium-browse <data_dir>")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
