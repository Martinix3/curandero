# app.py
# Streamlit Community Cloud ready
# Subes imágenes desde el navegador, pedimos captions a Hugging Face (Florence‑2‑base),
# aplicamos tu plantilla, recortamos al ratio, redimensionamos, y te damos un ZIP
# con .jpg + .txt pareados y un CSV de captions.

import io
import csv
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageOps
import streamlit as st

HF_DEFAULT_MODEL = "microsoft/Florence-2-base"
IMG_EXTS = {"jpg","jpeg","png","webp"}

# ---------- Helpers ----------

def exif_fix(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im)


def parse_aspect_ratio(s: str) -> float:
    s = (s or "2:3").strip().replace(" ", "")
    if ":" in s:
        a, b = s.split(":", 1)
        return float(a)/float(b)
    return float(s)


def center_crop_ratio(im: Image.Image, wh_ratio: float) -> Image.Image:
    w, h = im.size
    cur = w/h
    if abs(cur - wh_ratio) < 1e-6:
        return im
    if cur > wh_ratio:  # demasiado ancha
        new_w = int(h * wh_ratio)
        x0 = max(0, (w - new_w)//2)
        return im.crop((x0, 0, x0+new_w, h))
    else:               # demasiado alta
        new_h = int(w/wh_ratio)
        y0 = max(0, (h - new_h)//2)
        return im.crop((0, y0, w, y0+new_h))


def resize_exact(im: Image.Image, width: int, height: int) -> Image.Image:
    return im.resize((width, height), Image.LANCZOS)


def seq_name(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}.jpg"


def apply_template(template: str, raw: str, token: str) -> str:
    out = (template or "{raw} {token}")
    out = out.replace("{raw}", (raw or "").replace("\n"," ").strip())
    out = out.replace("{token}", token)
    return " ".join(out.split())


def _extract_caption_from_json(payload) -> str:
    try:
        if isinstance(payload, list) and payload:
            d0 = payload[0]
            if isinstance(d0, dict):
                for k in ("generated_text","caption","text"):
                    v = d0.get(k)
                    if isinstance(v,str) and v.strip():
                        return v.strip()
        if isinstance(payload, dict):
            for k in ("caption","text","generated_text"):
                v = payload.get(k)
                if isinstance(v,str) and v.strip():
                    return v.strip()
            r = payload.get("result")
            if isinstance(r, dict):
                for k in ("caption","text","generated_text"):
                    v = r.get(k)
                    if isinstance(v,str) and v.strip():
                        return v.strip()
    except Exception:
        pass
    return ""


def caption_hf(image_bytes: bytes, hf_token: str, model: str = HF_DEFAULT_MODEL, timeout=60) -> str:
    if not hf_token:
        return ""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    r = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
    try:
        payload = r.json()
    except Exception:
        return ""
    cap = _extract_caption_from_json(payload)
    # Cuando el modelo está "warming", puede devolver 503 con un JSON; intentamos rascar texto igualmente
    return (cap or "").strip()


# ---------- UI ----------

st.set_page_config(page_title="Caption Builder · Florence‑2 (HF)", layout="wide")
st.title("Caption Builder · Florence‑2 (Hugging Face)")
st.caption("Sube fotos → sugerimos caption → aplicamos tu plantilla → recorte 2:3 → resize → descarga ZIP")

with st.sidebar:
    st.markdown("### Proveedor")
    hf_token = st.text_input("HF Token", value=st.secrets.get("HF_TOKEN",""), type="password")
    hf_model = st.text_input("HF Model", value=HF_DEFAULT_MODEL)

    st.markdown("### Parámetros")
    name_prefix = st.text_input("Prefijo de nombre", value="influencer")
    token = st.text_input("Token", value="<influencer_street>")
    final_w = st.number_input("Ancho final", min_value=128, max_value=4096, value=1024, step=64)
    final_h = st.number_input("Alto final", min_value=128, max_value=4096, value=1536, step=64)
    aspect_str = st.text_input("Aspect ratio (W:H o float)", value="2:3")
    try:
        wh_ratio = parse_aspect_ratio(aspect_str)
    except Exception:
        st.error("Aspect ratio inválido. Usando 2:3.")
        wh_ratio = 2/3

    st.markdown("### Plantilla global")
    template = st.text_area(
        "Usa {raw} para el caption original y {token} para tu token",
        value=(
            "attractive young adult, tasteful streetwear; {raw}; "
            "small-town Spain vibe (whitewashed walls, visible overhead cables, worn formica); "
            "natural skin texture; {token}"
        ), height=140)

st.markdown("### 1) Sube imágenes")
files = st.file_uploader("Arrastra o selecciona varias imágenes", type=list(IMG_EXTS), accept_multiple_files=True)

if not files:
    st.info("Sube algunas imágenes para empezar.")
    st.stop()

# Vista previa pequeña
g = st.columns(min(5, len(files)))
for i, f in enumerate(files[:5]):
    with g[i%len(g)]:
        try:
            im = Image.open(f).convert("RGB")
            st.image(im, caption=f.name, use_column_width=True)
        except Exception:
            st.write(f"(no preview) {f.name}")

st.markdown("### 2) Generar captions (Florence‑2)")
if st.button("📥 Precalcular captions"):
    rows = []
    prog = st.progress(0)
    for i, f in enumerate(files):
        f_bytes = f.getvalue()
        try:
            raw = caption_hf(f_bytes, hf_token, hf_model)
        except Exception as e:
            raw = ""
        templated = apply_template(template, raw, token)
        rows.append({"file": f.name, "raw": raw, "templated": templated})
        prog.progress(int((i+1)*100/len(files)))
    st.success("Captions generados.")
    st.dataframe(rows, use_container_width=True)

st.markdown("### 3) Procesar (crop/resize/rename) y descargar ZIP")
if st.button("✅ Procesar y descargar"):
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED)
    # CSV de captions
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["dst_name","caption"])  # header

    for idx, f in enumerate(files, start=1):
        # abrir imagen
        try:
            im = Image.open(f).convert("RGB")
            im = exif_fix(im)
            im = center_crop_ratio(im, wh_ratio)
            im = resize_exact(im, int(final_w), int(final_h))
        except Exception as e:
            continue
        # nombre destino
        out_name = seq_name(name_prefix, idx)
        # caption
        raw = caption_hf(f.getvalue(), hf_token, hf_model)
        templated = apply_template(template, raw, token)
        # añadir .jpg
        jpg_bytes = io.BytesIO()
        im.save(jpg_bytes, format="JPEG", quality=92, optimize=True)
        z.writestr(out_name, jpg_bytes.getvalue())
        # añadir .txt
        z.writestr(f"{Path(out_name).stem}.txt", (templated.strip()+"\n").encode("utf-8"))
        # CSV
        writer.writerow([out_name, templated])

    # añadir CSV al zip
    z.writestr("captions.csv", csv_buf.getvalue().encode("utf-8"))
    z.close()
    buf.seek(0)
    st.download_button("⬇️ Descargar dataset.zip", data=buf, file_name="dataset.zip", mime="application/zip")
    st.success("Listo. Descarga tu ZIP con imágenes + .txt + CSV.")
