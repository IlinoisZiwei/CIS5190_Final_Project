"""Local, manifest-based notebook inputs. No raw or sealed data discovery."""
from pathlib import Path
import csv
import json
import os
import re
import sys
from urllib.parse import urlparse, unquote
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "data/manifests/approved_inputs.json").read_text())
READS = set()
LABEL_TO_ID = {"FoxNews": 0, "NBC": 1}

def _registered_path(kind, key):
    record = MANIFEST[kind][key]
    p = (ROOT / record["path"]).resolve()
    if ROOT not in p.parents or "holdout" in p.name.lower() or p.name == "external_metadata.csv":
        raise PermissionError("Unapproved input")
    return p

def load_data(key):
    p = _registered_path("data", key)
    if p.suffix == ".csv":
        frame = pd.read_csv(p, keep_default_na=False)
    else:
        frame = pd.DataFrame([json.loads(s) for s in p.read_text().splitlines() if s.strip()])
    assert len(frame) == MANIFEST["data"][key]["rows"], key
    READS.add(str(p.relative_to(ROOT)))
    return frame

def load_result(key):
    p = _registered_path("results", key)
    READS.add(str(p.relative_to(ROOT)))
    return json.loads(p.read_text())

def source_text(original_relative):
    entry = next(r for r in MANIFEST["source_archive"] if r["original_relative"] == original_relative)
    path = ROOT / entry["path"]
    if "notebook_source" not in entry:
        return path.read_text()
    notebook = json.loads(path.read_text())
    cell = next(
        cell for cell in notebook["cells"]
        if cell.get("metadata", {}).get("dragonthree_source") == entry["notebook_source"]
    )
    source = "".join(cell["source"])
    if cell.get("metadata", {}).get("dragonthree_archived_guard"):
        return "".join(line[4:] for line in source.splitlines(keepends=True)[1:])
    return source.split(chr(96) * 4 + "python\n", 1)[1].rsplit("\n" + chr(96) * 4, 1)[0]

def source_link(original_relative):
    entry = next(r for r in MANIFEST["source_archive"] if r["original_relative"] == original_relative)
    link = entry["path"]
    return link + "#" + entry["anchor"] if "anchor" in entry else link

def install_data_guard():
    allowed = {_registered_path("data", key) for key in MANIFEST["data"]}
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        mode, flags = args[1], args[2]
        reading = not ((mode and any(c in mode for c in "wax")) or (flags and flags & (os.O_WRONLY | os.O_RDWR)))
        if not reading:
            return
        if "holdout" in p.name.lower() or p.name == "external_metadata.csv":
            raise PermissionError("Sealed or mixed data are excluded")
        if p.suffix.lower() in (".csv", ".jsonl") and p.resolve() not in allowed:
            raise PermissionError("Dataset is not on this package's explicit allowlist: " + str(p))
    sys.addaudithook(guard)

def url_to_text(url):
    path = urlparse(str(url).strip()).path
    segments = [part for part in path.split("/") if part]
    tail = unquote(segments[-1]) if segments else ""
    tail = re.sub(r"\.(html?|shtml|php)$", "", tail, flags=re.IGNORECASE)
    return " ".join(w for w in re.split(r"[-_]+", tail) if w and not re.search(r"\d", w))

def label_from_url(url):
    host = (urlparse(str(url).strip()).hostname or "").lower()
    for domain, label in (("foxnews.com", "FoxNews"), ("nbcnews.com", "NBC")):
        if host == domain or host.endswith("." + domain):
            return label
    return ""

def text_group(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

def labeled_urls(key):
    df = load_data(key).copy()
    df["text"] = df["url"].map(url_to_text)
    df["label"] = df["url"].map(label_from_url)
    assert df["label"].isin(LABEL_TO_ID).all()
    df["y"] = df["label"].map(LABEL_TO_ID)
    df["words"] = df["text"].str.split().str.len()
    df["characters"] = df["text"].str.len()
    return df

def show_figure(fig):
    from IPython.display import Image, display
    from io import BytesIO
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    display(Image(data=buf.getvalue()))
    plt.close(fig)
