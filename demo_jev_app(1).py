"""
Demo JEV AI, versi visual (Streamlit)
=====================================

Tampilan presentasi untuk demo triase tiket: grafik probabilitas, gauge ambang,
peta keputusan semua tiket, slider kebijakan, dan input pesan live dari audiens.

Jalankan (file demo_jev_triase.py harus ada di folder yang sama):

    pip install streamlit pandas altair          # + typesafe-sdk bila memakai TypeSafe langsung
    streamlit run demo_jev_app.py

Browser terbuka otomatis di http://localhost:8501. Key bisa diisi di sidebar
atau lewat variabel lingkungan SUMOPOD_API_KEY / TYPESAFE_API_KEY.
"""

import html
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import demo_jev_triase as d

st.set_page_config(page_title="Demo Triase Jev", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------------------
# Warna (sama dengan deck) & label keputusan per cabang aturan
# --------------------------------------------------------------------------------------
NAVY, AMBER, TEAL, CORAL, SLATE, SAGE, MIST = "#14213D", "#FCA311", "#2A9D8F", "#E76F51", "#6C7A89", "#8AB17D", "#A8B5C8"
KEPUTUSAN = [  # (label pendek, warna) untuk d.cabang() = 0..5
    ("Ke agen manusia", SLATE),
    ("Eskalasi P1", CORAL),
    ("Prioritas tinggi", AMBER),
    ("Formulir refund", SAGE),
    ("Otomatis", TEAL),
    ("Minta konfirmasi", MIST),
]
SIM, SUMO, TS = "Simulasi (angka ilustratif)", "Jev via SumoPod", "Jev via TypeSafe langsung"
DEFAULT_AMBANG = dict(d.AMBANG)

st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; }
.jev-msg { font-size: 1.15rem; line-height: 1.5; padding: .8rem 1rem; border-radius: 10px;
           background: rgba(127,127,127,.09); margin: .2rem 0 .8rem; }
.jev-code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .85rem; line-height: 1.7;
            background: rgba(127,127,127,.08); border-radius: 10px; padding: .5rem 0; overflow-x: auto; }
.jev-code div { padding: 0 .9rem; white-space: pre; opacity: .65; }
.jev-code div.hit { opacity: 1; font-weight: 700; background: rgba(252,163,17,.22); }
.jev-verdict { border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .7rem; }
.jev-verdict .k { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75; }
.jev-verdict .v { font-size: 1.45rem; font-weight: 800; line-height: 1.25; margin: .15rem 0 .2rem; }
.jev-step { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .7; margin: .3rem 0 .3rem; }
</style>
""",
    unsafe_allow_html=True,
)


def lebar(fn, *args, **kw):
    """Panggil elemen Streamlit selebar kolom, kompatibel versi lama & baru."""
    try:
        return fn(*args, width="stretch", **kw)
    except Exception:
        return fn(*args, use_container_width=True, **kw)


# --------------------------------------------------------------------------------------
# Sidebar: sumber jawaban, key, ambang
# --------------------------------------------------------------------------------------
for k, v in DEFAULT_AMBANG.items():
    st.session_state.setdefault(f"amb_{k}", v)
st.session_state.setdefault("custom", [])


def reset_ambang():
    for k, v in DEFAULT_AMBANG.items():
        st.session_state[f"amb_{k}"] = v


with st.sidebar:
    st.subheader("Sumber jawaban")
    mode = st.radio("Mode", [SIM, SUMO, TS], key="mode", label_visibility="collapsed")

    sumo_key = st.text_input("SUMOPOD_API_KEY", type="password", value=os.environ.get("SUMOPOD_API_KEY", ""),
                             help="Dipakai untuk Jev via SumoPod, draf balasan, dan pembanding LLM.")
    if mode == TS:
        ts_key = st.text_input("TYPESAFE_API_KEY", type="password", value=os.environ.get("TYPESAFE_API_KEY", ""))
        if ts_key:
            os.environ["TYPESAFE_API_KEY"] = ts_key
    with st.expander("Pengaturan SumoPod"):
        base = st.text_input("Base URL", value=os.environ.get("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1")).rstrip("/")
        jev_path = st.text_input("Path endpoint Jev", value=os.environ.get("SUMOPOD_JEV_PATH", "/systemone"))
        jev_model = st.text_input("Model Jev", value=os.environ.get("SUMOPOD_JEV_MODEL", "jev-1.13.0"))
        llm_model = st.text_input("Model LLM (draf & pembanding)", value=os.environ.get("SUMOPOD_MODEL", "gemini/gemini-2.5-flash"))
    if sumo_key:
        os.environ["SUMOPOD_API_KEY"] = sumo_key
    d.SUMOPOD_JEV_URL = base + jev_path
    d.SUMOPOD_JEV_MODEL = jev_model
    d.SUMOPOD_URL = base + "/chat/completions"
    d.SUMOPOD_MODEL = llm_model

    st.subheader("Kebijakan (ambang)")
    st.caption("Geser angka ini; prompt tidak berubah sama sekali.")
    st.slider("Batas ragu: confidence kategori", 0.0, 1.0, step=0.05, key="amb_conf_rendah")
    st.slider("Batas otomatis: confidence kategori", 0.0, 1.0, step=0.05, key="amb_conf_tinggi")
    st.slider("Ambang urgent", 0.0, 1.0, step=0.05, key="amb_urgent")
    st.slider("Ambang refund", 0.0, 1.0, step=0.05, key="amb_refund")
    st.slider("Ambang frustrasi (0–2)", 0.0, 2.0, step=0.1, key="amb_frustrasi_tinggi")
    st.button("Kembalikan ambang awal", on_click=reset_ambang)

AMB = {k: st.session_state[f"amb_{k}"] for k in DEFAULT_AMBANG}


# --------------------------------------------------------------------------------------
# Memanggil Jev (di-cache: menggeser slider tidak memanggil API lagi)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _tanya(mode, endpoint, model, tid, kanal, pesan):
    tiket = {"id": tid, "kanal": kanal, "pesan": pesan}
    t0 = time.perf_counter()
    if mode == SIM:
        j = d.tanya_jev_simulasi(tiket)
    elif mode == SUMO:
        j = d.tanya_jev_sumopod(tiket)
    else:
        j = d.tanya_jev_live(tiket)
    return j, (time.perf_counter() - t0) if mode != SIM else None


def tanya(t):
    endpoint = d.SUMOPOD_JEV_URL if mode == SUMO else ""
    try:
        return _tanya(mode, endpoint, d.SUMOPOD_JEV_MODEL if mode == SUMO else d.MODEL, t["id"], t["kanal"], t["pesan"]), None
    except SystemExit as e:  # skrip CLI memakai sys.exit untuk error yang jelas pesannya
        return (None, None), str(e)
    except Exception as e:
        return (None, None), f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False, ttl=3600)
def _draf(model, pesan, dept, keputusan):
    return d.draf_balasan_sumopod({"pesan": pesan}, d.Jawaban(dept, 0, {}, 0, 0, 0, 0), keputusan)


@st.cache_data(show_spinner=False, ttl=3600)
def _llm(model, tid, kanal, pesan):
    return d.tanya_llm_sumopod({"id": tid, "kanal": kanal, "pesan": pesan})


# --------------------------------------------------------------------------------------
# Data: tiket contoh + pesan dari audiens
# --------------------------------------------------------------------------------------
butuh_key = (mode == SUMO and not os.environ.get("SUMOPOD_API_KEY")) or (mode == TS and not os.environ.get("TYPESAFE_API_KEY"))

st.title("Triase tiket dengan Jev")
if mode == SIM:
    st.info("Mode simulasi: angka disusun manual sebagai ilustrasi, bukan output Jev. Pilih mode live di sidebar untuk Jev asli.")
elif butuh_key:
    st.warning(f"Isi API key di sidebar untuk mode **{mode}**.")
    st.stop()
else:
    st.caption(f"Live · {d.SUMOPOD_JEV_MODEL + ' via ' + base if mode == SUMO else d.MODEL + ' via api.typesafe.ai'}")

tiket_list = d.TIKET + (st.session_state.custom if mode != SIM else [])
hasil, galat = {}, []
with st.spinner(f"Memanggil Jev untuk {len(tiket_list)} tiket…"):
    for t in tiket_list:
        (j, dt), err = tanya(t)
        if err:
            galat.append(f"{t['id']}: {err}")
        else:
            hasil[t["id"]] = (t, j, dt)
if galat:
    st.error("Sebagian panggilan gagal:\n\n" + "\n\n".join(galat[:3]))
if not hasil:
    st.stop()

# Ringkasan angka
tok = sum(j.input_tokens for _, j, _ in hasil.values())
lat = [dt for _, _, dt in hasil.values() if dt is not None]
m = st.columns(5)
m[0].metric("Panggilan API", len(hasil))
m[1].metric("Pertanyaan dijawab", len(hasil) * 4)
m[2].metric("Token input", f"{tok:,}".replace(",", "."))
m[3].metric("Biaya (US$)", f"{tok / 1e6 * d.HARGA_PER_JUTA_TOKEN:.6f}", help="US$0,042 per 1 juta token input; output gratis. Tarif SumoPod bisa berbeda.")
m[4].metric("Latensi rata-rata", f"{sum(lat) / len(lat) * 1000:.0f} ms" if lat else "–")

tab1, tab2, tab3 = st.tabs(["Per tiket", "Peta semua tiket", "Jev vs LLM biasa"])


# --------------------------------------------------------------------------------------
# Grafik
# --------------------------------------------------------------------------------------
def chart_departemen(j):
    probs = j.dept_probs or {j.dept: 1.0}
    df = pd.DataFrame({"departemen": list(probs), "p": list(probs.values())})
    df["pilihan"] = df["departemen"] == j.dept
    base_ = alt.Chart(df).encode(y=alt.Y("departemen:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)))
    bars = base_.mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), title="probabilitas", axis=alt.Axis(format=".0%")),
        color=alt.condition("datum.pilihan", alt.value(AMBER), alt.value(SLATE)),
        tooltip=["departemen", alt.Tooltip("p:Q", format=".2f")],
    )
    text = base_.mark_text(align="left", dx=5, fontSize=13).encode(x="p:Q", text=alt.Text("p:Q", format=".2f"))
    return (bars + text).properties(height=max(150, 34 * len(df)))


def chart_gauge(j):
    rows = [
        ("urgent (Noul)", j.urgent, j.urgent, AMB["urgent"], f"{j.urgent:.2f}", j.urgent >= AMB["urgent"]),
        ("refund (Noul)", j.refund, j.refund, AMB["refund"], f"{j.refund:.2f}", j.refund >= AMB["refund"]),
        ("frustrasi (Score)", j.frustration / 2, j.frustration, AMB["frustrasi_tinggi"] / 2, f"{j.frustration:.2f} / 2",
         j.frustration >= AMB["frustrasi_tinggi"]),
    ]
    df = pd.DataFrame(rows, columns=["pertanyaan", "nilai", "asli", "ambang", "label", "lewat"])
    df["nol"], df["penuh"] = 0.0, 1.0
    y = alt.Y("pertanyaan:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13))
    track = alt.Chart(df).mark_bar(height=22, opacity=0.15, color=SLATE).encode(y=y, x=alt.X("nol:Q", scale=alt.Scale(domain=[0, 1])), x2="penuh:Q")
    bars = alt.Chart(df).mark_bar(height=22, cornerRadiusEnd=4).encode(
        y=y, x=alt.X("nilai:Q", scale=alt.Scale(domain=[0, 1]), title="nilai (frustrasi dinormalkan ke 0–1)", axis=alt.Axis(format=".0%")),
        color=alt.condition("datum.lewat", alt.value(CORAL), alt.value(SLATE)),
        tooltip=["pertanyaan", "label", alt.Tooltip("ambang:Q", format=".2f")],
    )
    thr = alt.Chart(df).mark_tick(thickness=3, size=34, color=AMBER).encode(y=y, x="ambang:Q")
    text = alt.Chart(df).mark_text(align="left", dx=6, fontSize=13).encode(y=y, x="nilai:Q", text="label:N")
    return alt.layer(track, bars, thr, text).properties(height=150)


def chart_peta(rows):
    df = pd.DataFrame(rows)
    domain = [k for k, _ in KEPUTUSAN]
    warna = alt.Scale(domain=domain, range=[c for _, c in KEPUTUSAN])
    shade = alt.Chart(pd.DataFrame({"x": [0], "x2": [AMB["conf_rendah"]]})).mark_rect(opacity=0.10, color=SLATE).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])), x2="x2:Q")
    v1 = alt.Chart(pd.DataFrame({"x": [AMB["conf_rendah"]]})).mark_rule(strokeDash=[5, 4], color=SLATE).encode(x="x:Q")
    v2 = alt.Chart(pd.DataFrame({"x": [AMB["conf_tinggi"]]})).mark_rule(strokeDash=[5, 4], color=TEAL).encode(x="x:Q")
    h = alt.Chart(pd.DataFrame({"y": [AMB["urgent"]]})).mark_rule(strokeDash=[5, 4], color=CORAL).encode(y="y:Q")
    pts = alt.Chart(df).mark_circle(opacity=0.9, stroke="white", strokeWidth=1.5).encode(
        x=alt.X("conf:Q", scale=alt.Scale(domain=[0, 1]), title="confidence kategori (Choice) →", axis=alt.Axis(values=[i / 10 for i in range(11)], format=".1f")),
        y=alt.Y("urgent:Q", scale=alt.Scale(domain=[0, 1.05]), title="urgent (Noul) →", axis=alt.Axis(values=[i / 5 for i in range(6)], format=".1f")),
        size=alt.Size("frustrasi:Q", scale=alt.Scale(domain=[0, 2], range=[150, 1400]), legend=alt.Legend(title="frustrasi (ukuran)", symbolFillColor=SLATE, symbolStrokeColor=SLATE)),
        color=alt.Color("keputusan:N", scale=warna, legend=alt.Legend(title="keputusan", orient="bottom", columns=3)),
        tooltip=["tiket", "departemen", "keputusan", alt.Tooltip("conf:Q", format=".2f"),
                 alt.Tooltip("urgent:Q", format=".2f"), alt.Tooltip("frustrasi:Q", format=".2f")],
    )
    lbl = alt.Chart(df).mark_text(dy=-20, fontSize=13, fontWeight="bold").encode(x="conf:Q", y="urgent:Q", text="tiket:N")
    zona = alt.Chart(pd.DataFrame({"x": [AMB["conf_rendah"] / 2], "y": [1.02], "t": ["zona ragu → manusia"]})).mark_text(
        fontSize=12, color=SLATE).encode(x="x:Q", y="y:Q", text="t:N")
    return alt.layer(shade, v1, v2, h, pts, lbl, zona).properties(height=440)


# --------------------------------------------------------------------------------------
# Tab 1: per tiket
# --------------------------------------------------------------------------------------
with tab1:
    ids = list(hasil)
    if "pilih_next" in st.session_state:
        st.session_state.pilih = st.session_state.pop("pilih_next")
    if st.session_state.get("pilih") not in ids:
        st.session_state.pilih = ids[0]
    pilih = st.radio("Tiket", ids, horizontal=True, key="pilih",
                     format_func=lambda i: f"{i} · {hasil[i][0]['kanal']}", label_visibility="collapsed")
    t, j, dt = hasil[pilih]
    st.markdown(f"<div class='jev-msg'>“{html.escape(t['pesan'])}”</div>", unsafe_allow_html=True)

    kiri, kanan = st.columns([1, 1], gap="large")
    with kiri:
        st.markdown("<div class='jev-step'>1 · Request ke Jev (satu panggilan)</div>", unsafe_allow_html=True)
        model_id = d.SUMOPOD_JEV_MODEL if mode == SUMO else d.MODEL
        req = (
            "{\n"
            f'  "model": "{model_id}",\n'
            f'  "state": {{"channel": "{t["kanal"]}",\n'
            f'            "message": {json.dumps(t["pesan"], ensure_ascii=False)}}},\n'
            '  "questions": {\n'
            f'    "department":  {{"type": "choice", "criteria": {len(d.DEPARTEMEN)} tim}},\n'
            '    "urgent":      {"type": "noul"},\n'
            '    "refund":      {"type": "noul"},\n'
            f'    "frustration": {{"type": "score", "criteria": {len(d.LEVEL_FRUSTRASI)} level}}\n'
            "  }\n}"
        )
        try:
            st.code(req, language="json", wrap_lines=True)
        except TypeError:
            st.code(req, language="json")
        st.markdown("<div class='jev-step'>2a · department (Choice)</div>", unsafe_allow_html=True)
        lebar(st.altair_chart, chart_departemen(j))
        st.caption(f"Pilihan **{j.dept}** · confidence **{j.dept_conf:.2f}**"
                   + (f" · {dt * 1000:.0f} ms" if dt is not None else "") + f" · {j.input_tokens} token input")

    with kanan:
        st.markdown("<div class='jev-step'>2b · Noul & Score (garis kuning = ambang)</div>", unsafe_allow_html=True)
        lebar(st.altair_chart, chart_gauge(j))
        idx = d.cabang(j, AMB)
        label, warna = KEPUTUSAN[idx]
        st.markdown("<div class='jev-step'>3 · Keputusan kode Anda</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='jev-verdict' style='background:{warna}33'><div class='k'>Tindakan</div>"
            f"<div class='v'>{html.escape(d.putuskan(j, AMB))}</div></div>", unsafe_allow_html=True)
        baris = [
            f"if dept.conf < {AMB['conf_rendah']:.2f}:  manusia()",
            f"elif urgent ≥ {AMB['urgent']:.2f} and frust ≥ {AMB['frustrasi_tinggi']:.2f}:  eskalasi_p1()",
            f"elif urgent ≥ {AMB['urgent']:.2f}:  prioritas_tinggi()",
            f"elif refund ≥ {AMB['refund']:.2f}:  form_refund()",
            f"elif dept.conf ≥ {AMB['conf_tinggi']:.2f}:  otomatis()",
            "else:  minta_konfirmasi()",
        ]
        st.markdown("<div class='jev-code'>" + "".join(
            f"<div class='{'hit' if i == idx else ''}'>{html.escape(b)}</div>" for i, b in enumerate(baris)) + "</div>",
            unsafe_allow_html=True)

        st.markdown("<div class='jev-step' style='margin-top:1rem'>4 · Sistem 2: LLM menulis draf (opsional)</div>", unsafe_allow_html=True)
        if idx in (0, 1):
            st.caption("Tiket ini ke manusia, jadi tidak dibuatkan draf otomatis.")
        elif not os.environ.get("SUMOPOD_API_KEY"):
            st.caption("Isi SUMOPOD_API_KEY di sidebar untuk membuat draf balasan dengan LLM via SumoPod.")
        elif st.button("Buat draf balasan (SumoPod)", key=f"draf_{pilih}"):
            try:
                with st.spinner("LLM menulis draf…"):
                    st.success(_draf(d.SUMOPOD_MODEL, t["pesan"], j.dept, d.putuskan(j, AMB)))
            except (SystemExit, Exception) as e:
                st.error(str(e))

    st.divider()
    if mode == SIM:
        st.caption("Coba pesan dari audiens tersedia di mode live (Jev via SumoPod atau TypeSafe).")
    else:
        with st.form("baru", clear_on_submit=True):
            st.markdown("**Coba pesan dari audiens**")
            c1, c2 = st.columns([4, 1])
            pesan = c1.text_area("Pesan pelanggan", placeholder="contoh: kak pesanan sy kok blm dikirim2 udh 5 hari", label_visibility="collapsed")
            kanal = c2.selectbox("Kanal", ["Chat", "Email", "WhatsApp", "Ulasan"])
            if st.form_submit_button("Kirim ke Jev") and pesan.strip():
                nid = f"A-{len(st.session_state.custom) + 1}"
                st.session_state.custom.append({"id": nid, "kanal": kanal, "pesan": pesan.strip()})
                st.session_state.pilih_next = nid
                st.rerun()


# --------------------------------------------------------------------------------------
# Tab 2: peta semua tiket
# --------------------------------------------------------------------------------------
with tab2:
    rows = []
    for tid, (t, j, dt) in hasil.items():
        rows.append({"tiket": tid, "pesan": t["pesan"], "departemen": j.dept, "conf": j.dept_conf, "urgent": j.urgent,
                     "refund": j.refund, "frustrasi": j.frustration, "keputusan": KEPUTUSAN[d.cabang(j, AMB)][0],
                     "latensi_ms": round(dt * 1000) if dt is not None else None})
    st.markdown("Setiap titik satu tiket. Geser ambang di sidebar dan lihat titik berpindah warna; prompt tidak disentuh.")
    lebar(st.altair_chart, chart_peta(rows))
    cc = st.column_config
    lebar(st.dataframe, pd.DataFrame(rows), hide_index=True, column_config={
        "pesan": cc.TextColumn("pesan", width="large"),
        "conf": cc.ProgressColumn("conf", min_value=0, max_value=1, format="%.2f"),
        "urgent": cc.ProgressColumn("urgent", min_value=0, max_value=1, format="%.2f"),
        "refund": cc.ProgressColumn("refund", min_value=0, max_value=1, format="%.2f"),
        "frustrasi": cc.ProgressColumn("frustrasi", min_value=0, max_value=2, format="%.2f"),
        "latensi_ms": cc.NumberColumn("latensi (ms)"),
    })


# --------------------------------------------------------------------------------------
# Tab 3: pembanding LLM biasa via SumoPod
# --------------------------------------------------------------------------------------
with tab3:
    st.markdown("Tugas yang sama dikerjakan **LLM generatif via SumoPod** (prompt minta JSON). "
                "Bandingkan latensi, token output yang dibayar, dan sifat confidence-nya.")
    if not os.environ.get("SUMOPOD_API_KEY"):
        st.info("Isi SUMOPOD_API_KEY di sidebar untuk menjalankan pembanding.")
    elif st.button("Jalankan pembanding", key="cmp") or st.session_state.get("cmp_done"):
        st.session_state.cmp_done = True
        baris_cmp, gagal = [], None
        with st.spinner(f"Memanggil {d.SUMOPOD_MODEL} untuk {len(hasil)} tiket…"):
            for tid, (t, j, dt) in hasil.items():
                try:
                    lj, out_tok, ldt = _llm(d.SUMOPOD_MODEL, tid, t["kanal"], t["pesan"])
                except (SystemExit, Exception) as e:
                    gagal = str(e)
                    break
                baris_cmp.append({"tiket": tid, "Jev: departemen": j.dept, "LLM: departemen": lj.dept,
                                  "Jev: keputusan": KEPUTUSAN[d.cabang(j, AMB)][0], "LLM: keputusan": KEPUTUSAN[d.cabang(lj, AMB)][0],
                                  "Jev ms": round(dt * 1000) if dt is not None else None, "LLM ms": round(ldt * 1000),
                                  "LLM token output": out_tok})
        if gagal:
            st.error(gagal)
        if baris_cmp:
            df = pd.DataFrame(baris_cmp)
            c = st.columns(3)
            c[0].metric("Keputusan sama", f"{(df['Jev: keputusan'] == df['LLM: keputusan']).sum()} / {len(df)}")
            c[1].metric("Token output LLM", f"{int(df['LLM token output'].sum())}", help="Di TypeSafe langsung, output Jev gratis. Lewat SumoPod, cek tarif di dashboard SumoPod.")
            if mode != SIM:
                c[2].metric("Latensi rata-rata", f"Jev {df['Jev ms'].mean():.0f} · LLM {df['LLM ms'].mean():.0f} ms")
                lat_df = df.melt(id_vars="tiket", value_vars=["Jev ms", "LLM ms"], var_name="model", value_name="ms")
                lebar(st.altair_chart, alt.Chart(lat_df).mark_bar(cornerRadiusEnd=3).encode(
                    y=alt.Y("tiket:N", title=None), yOffset="model:N", x=alt.X("ms:Q", title="latensi (ms)"),
                    color=alt.Color("model:N", scale=alt.Scale(domain=["Jev ms", "LLM ms"], range=[AMBER, SLATE]), legend=alt.Legend(orient="bottom")),
                    tooltip=["tiket", "model", "ms"]).properties(height=60 * len(df)))
            else:
                c[2].metric("Latensi", "pakai mode live")
            lebar(st.dataframe, df, hide_index=True)
            st.caption("Confidence dari LLM adalah angka yang ditulis model (verbal), bukan probabilitas terkalibrasi seperti Jev.")
