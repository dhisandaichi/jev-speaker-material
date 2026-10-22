"""
Demo JEV AI: Triase tiket customer service e-commerce (Bahasa Indonesia)
=======================================================================

Satu panggilan Jev per tiket menjawab 4 pertanyaan sekaligus:
  - department  (Choice) : tim mana yang menangani
  - urgent      (Noul)   : apakah mendesak
  - refund      (Noul)   : apakah pelanggan minta uang kembali
  - frustration (Score)  : seberapa frustrasi (0 tenang .. 2 sangat marah)

Lalu KODE (bukan prompt) yang memutuskan tindakan, memakai ambang confidence.

Cara menjalankan
----------------
Mode live (butuh API key TypeSafe):
    pip install typesafe-sdk
    export TYPESAFE_API_KEY=...        # Windows: set TYPESAFE_API_KEY=...
    python demo_jev_triase.py

Mode simulasi (tanpa internet / tanpa key, untuk cadangan saat presentasi):
    python demo_jev_triase.py --simulasi

Catatan: angka di mode simulasi adalah CONTOH ILUSTRATIF yang disusun manual,
bukan output asli Jev. Bentuk datanya sama dengan respons API sungguhan.

SumoPod (gateway AI Indonesia, https://ai.sumopod.com/v1, format OpenAI)
------------------------------------------------------------------------
SumoPod menyediakan Jev (model jev-1.13.0, endpoint POST /v1/systemone, format request
TypeSafe), terlihat di Decision playground SumoPod. Selain memanggil Jev, SumoPod juga bisa
dipakai untuk peran "Sistem 2" (LLM yang menulis), atau sebagai PEMBANDING klasifikasi via LLM biasa:

    export SUMOPOD_API_KEY=...                       # dari ai.sumopod.com
    export SUMOPOD_MODEL=gemini/gemini-2.5-flash     # opsional, model apa pun di akun Anda

    # Jev memutuskan + LLM SumoPod menulis draf balasan untuk tiket yang otomatis:
    python demo_jev_triase.py --draf-sumopod                 # (atau + --simulasi)

    # Tanpa Jev sama sekali: tugas yang sama dikerjakan LLM via SumoPod (pembanding):
    python demo_jev_triase.py --pembanding-sumopod

Jev (jev-1.13.0) LEWAT SUMOPOD, satu key untuk semuanya:
    python demo_jev_triase.py --cek-sumopod          # pastikan jev-1.13.0 ada di akun Anda
    python demo_jev_triase.py --jev-sumopod          # Jev memutuskan via SumoPod
    python demo_jev_triase.py --jev-sumopod --draf-sumopod   # + LLM SumoPod menulis draf

    Default: POST https://ai.sumopod.com/v1/systemone dengan format request TypeSafe
    (pola yang sama dengan Vercel AI Gateway & LiteLLM). Jika SumoPod memakai path lain,
    set SUMOPOD_JEV_PATH; nama model bisa diganti lewat SUMOPOD_JEV_MODEL.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

MODEL = "jev-1.13.0"           # kunci versi untuk produksi; "jev-latest" bisa berpindah versi
HARGA_PER_JUTA_TOKEN = 0.042   # USD per 1 juta token input (Oktober 2026), output gratis

# --------------------------------------------------------------------------------------
# 1) Data contoh: tiket masuk
# --------------------------------------------------------------------------------------
TIKET = [
    {"id": "T-101", "kanal": "Email",
     "pesan": "Paket saya sudah 9 hari belum sampai, resi JNE tidak bergerak sejak Senin. Tolong dicek ya."},
    {"id": "T-102", "kanal": "Chat",
     "pesan": "Saya ditagih DUA KALI untuk pesanan INV-88231!!! Kembalikan uang saya sekarang atau saya laporkan."},
    {"id": "T-103", "kanal": "Chat",
     "pesan": "Halo kak, mau tanya apakah alamat pengiriman masih bisa diganti sebelum paketnya dikirim?"},
    {"id": "T-104", "kanal": "WhatsApp",
     "pesan": "Akun saya tidak bisa login sejak pagi, OTP tidak masuk. Saya mau checkout promo yang berakhir jam 12 siang ini."},
    {"id": "T-105", "kanal": "Ulasan",
     "pesan": "barangnya udah sampe tp warnanya beda sama yg di foto, gmn ya"},
]

# --------------------------------------------------------------------------------------
# 2) Pertanyaan untuk Jev (instruksi dalam bahasa Inggris = bahasa terkuat model;
#    state tetap Bahasa Indonesia. Uji juga versi instruksi Bahasa Indonesia.)
# --------------------------------------------------------------------------------------
DEPARTEMEN = {
    "pengiriman": "Shipping, delivery delays, tracking, or changing the delivery address",
    "pembayaran": "Payments, double charges, invoices, or payment failures",
    "retur_refund": "Returns, wrong or damaged items, exchanges, or getting money back for an item",
    "akun": "Login, OTP, password, or account access problems",
    "lainnya": "Anything that does not fit the other teams",
}
LEVEL_FRUSTRASI = [
    "Calm or neutral, just asking or stating facts",
    "Annoyed or concerned but still polite",
    "Very angry, uses capitals, threats, or strong language",
]


def bangun_pertanyaan():
    from typesafe_sdk import Choice, Noul, Score
    return {
        "department": Choice(
            instructions="Which support team should handle `message`?",
            criteria=DEPARTEMEN,
        ),
        "urgent": Noul(
            instructions="Does `message` describe a problem that must be handled within the next few hours?",
        ),
        "refund": Noul(
            instructions="Does the customer in `message` explicitly ask for their money back?",
        ),
        "frustration": Score(
            instructions="How frustrated does the customer in `message` appear?",
            criteria=LEVEL_FRUSTRASI,
        ),
    }


# --------------------------------------------------------------------------------------
# 3) Bentuk jawaban yang dipakai kode (sama untuk live maupun simulasi)
# --------------------------------------------------------------------------------------
@dataclass
class Jawaban:
    dept: str
    dept_conf: float
    dept_probs: dict
    urgent: float
    refund: float
    frustration: float
    input_tokens: int


def conf_choice(probs: dict) -> float:
    """Rumus confidence Choice dari dokumentasi TypeSafe: (pmax - 1/n) / (1 - 1/n)."""
    n = len(probs)
    return (max(probs.values()) - 1 / n) / (1 - 1 / n)


def tanya_jev_live(tiket) -> Jawaban:
    from typesafe_sdk import TypeSafeClient
    with TypeSafeClient(model=MODEL) as client:
        r = client.system_one(
            state={"channel": tiket["kanal"], "message": tiket["pesan"]},
            questions=bangun_pertanyaan(),
        )
    a = r.answers
    usage = getattr(r, "usage", None)
    return Jawaban(
        dept=a["department"].choice,
        dept_conf=a["department"].confidence,
        dept_probs=dict(a["department"].probabilities),
        urgent=a["urgent"].noul,
        refund=a["refund"].noul,
        frustration=a["frustration"].score,
        input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
    )


# Nilai ILUSTRATIF untuk mode simulasi (disusun manual agar realistis).
SIMULASI = {
    "T-101": dict(probs={"pengiriman": .90, "pembayaran": .02, "retur_refund": .03, "akun": .01, "lainnya": .04},
                  urgent=.62, refund=.08, frustration=0.92, tokens=301),
    "T-102": dict(probs={"pengiriman": .01, "pembayaran": .80, "retur_refund": .17, "akun": .01, "lainnya": .01},
                  urgent=.97, refund=.96, frustration=1.85, tokens=309),
    "T-103": dict(probs={"pengiriman": .66, "pembayaran": .03, "retur_refund": .03, "akun": .14, "lainnya": .14},
                  urgent=.21, refund=.03, frustration=0.10, tokens=303),
    "T-104": dict(probs={"pengiriman": .01, "pembayaran": .03, "retur_refund": .01, "akun": .93, "lainnya": .02},
                  urgent=.93, refund=.05, frustration=1.10, tokens=312),
    "T-105": dict(probs={"pengiriman": .20, "pembayaran": .06, "retur_refund": .55, "akun": .04, "lainnya": .15},
                  urgent=.18, refund=.52, frustration=0.71, tokens=296),
}


def tanya_jev_simulasi(tiket) -> Jawaban:
    s = SIMULASI[tiket["id"]]
    probs = s["probs"]
    return Jawaban(
        dept=max(probs, key=probs.get), dept_conf=conf_choice(probs), dept_probs=probs,
        urgent=s["urgent"], refund=s["refund"], frustration=s["frustration"], input_tokens=s["tokens"],
    )


# --------------------------------------------------------------------------------------
# 4) LOGIKA BISNIS: di sinilah keputusan dibuat. Ubah angka, bukan prompt.
# --------------------------------------------------------------------------------------
AMBANG = {
    "conf_rendah": 0.50,    # di bawah ini: Jev ragu soal kategori -> manusia
    "conf_tinggi": 0.80,    # di atas ini: boleh otomatis
    "urgent": 0.80,
    "refund": 0.80,
    "frustrasi_tinggi": 1.50,
}


def cabang(j: Jawaban, amb=AMBANG) -> int:
    """Nomor aturan yang terpenuhi (0-5). Dipakai juga oleh aplikasi visual."""
    if j.dept_conf < amb["conf_rendah"]:
        return 0
    if j.urgent >= amb["urgent"] and j.frustration >= amb["frustrasi_tinggi"]:
        return 1
    if j.urgent >= amb["urgent"]:
        return 2
    if j.refund >= amb["refund"]:
        return 3
    if j.dept_conf >= amb["conf_tinggi"]:
        return 4
    return 5


def putuskan(j: Jawaban, amb=AMBANG) -> str:
    return [
        "Ke agen manusia: kategori tidak jelas",
        f"Eskalasi P1 ke senior agent ({j.dept})",
        f"Prioritas tinggi: antrean {j.dept}",
        "Buka formulir refund otomatis",
        f"Otomatis: antrean {j.dept}",
        f"Usulan {j.dept}, minta konfirmasi agen",
    ][cabang(j, amb)]


# --------------------------------------------------------------------------------------
# 5) SumoPod: LLM generatif lewat endpoint OpenAI-compatible
# --------------------------------------------------------------------------------------
SUMOPOD_URL = os.environ.get("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1") + "/chat/completions"
SUMOPOD_MODEL = os.environ.get("SUMOPOD_MODEL", "gemini/gemini-2.5-flash")


def sumopod_chat(messages, max_tokens=400, json_mode=False):
    """Panggil SumoPod (format OpenAI). Mengembalikan (teks, usage, detik)."""
    key = os.environ.get("SUMOPOD_API_KEY")
    if not key:
        sys.exit("SUMOPOD_API_KEY belum di-set (buat di ai.sumopod.com).")
    body = {"model": SUMOPOD_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(SUMOPOD_URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        if json_mode and e.code in (400, 422):   # model tidak mendukung response_format: coba tanpa
            return sumopod_chat(messages, max_tokens, json_mode=False)
        sys.exit(f"SumoPod error {e.code}: {detail}")
    dt = time.perf_counter() - t0
    return data["choices"][0]["message"]["content"], data.get("usage", {}), dt


SUMOPOD_JEV_MODEL = os.environ.get("SUMOPOD_JEV_MODEL", "jev-1.13.0")
# Gateway lain (Vercel, LiteLLM) meneruskan Jev di <base>/v1/systemone dengan format TypeSafe.
# Jika SumoPod memakai path lain, set SUMOPOD_JEV_PATH sesuai dokumentasi/dashboard mereka.
SUMOPOD_JEV_URL = os.environ.get("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1") + os.environ.get("SUMOPOD_JEV_PATH", "/systemone")


def pertanyaan_json():
    """Pertanyaan yang sama dengan bangun_pertanyaan(), tapi JSON biasa (tanpa SDK)."""
    return {
        "department": {"type": "choice", "instructions": "Which support team should handle `message`?", "criteria": DEPARTEMEN},
        "urgent": {"type": "noul", "instructions": "Does `message` describe a problem that must be handled within the next few hours?"},
        "refund": {"type": "noul", "instructions": "Does the customer in `message` explicitly ask for their money back?"},
        "frustration": {"type": "score", "instructions": "How frustrated does the customer in `message` appear?", "criteria": LEVEL_FRUSTRASI},
    }


def _post_sumopod(url, body):
    key = os.environ.get("SUMOPOD_API_KEY")
    if not key:
        sys.exit("SUMOPOD_API_KEY belum di-set (buat di ai.sumopod.com).")
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def tanya_jev_sumopod(tiket) -> Jawaban:
    """Jev asli, dipanggil lewat SumoPod dengan format request TypeSafe."""
    body = {"model": SUMOPOD_JEV_MODEL, "state": {"channel": tiket["kanal"], "message": tiket["pesan"]},
            "questions": pertanyaan_json()}
    try:
        data = _post_sumopod(SUMOPOD_JEV_URL, body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        sys.exit(f"SumoPod menolak request Jev ({e.code}) di {SUMOPOD_JEV_URL}: {detail}\n"
                 "→ Jalankan --cek-sumopod, lalu cek path endpoint Jev di dashboard/dokumentasi SumoPod "
                 "dan set SUMOPOD_JEV_PATH. Bandingkan dengan tab cURL di Decision playground SumoPod.")
    try:
        a = data["answers"]
        usage = data.get("usage", {})
        probs = a["department"].get("probabilities") or {}
        conf = a["department"].get("confidence")
        return Jawaban(
            dept=a["department"]["choice"],
            dept_conf=float(conf) if conf is not None else conf_choice(probs),
            dept_probs=probs,
            urgent=_nilai_noul(a["urgent"]), refund=_nilai_noul(a["refund"]),
            frustration=_nilai_score(a["frustration"]),
            input_tokens=usage.get("input_tokens", 0),
        )
    except (KeyError, TypeError, ValueError) as e:
        sys.exit(f"Bentuk respons SumoPod tidak dikenali ({e!r}). Respons mentah:\n"
                 f"{json.dumps(data, ensure_ascii=False)[:800]}")


def _nilai_noul(a: dict) -> float:
    """Probabilitas 'ya'. TypeSafe memakai field 'noul'; beberapa gateway memakai 'probability'."""
    for k in ("noul", "probability", "p_true"):
        if a.get(k) is not None:
            return float(a[k])
    pr = a.get("probabilities") or {}
    for k in ("true", "yes", "Yes", "True"):
        if k in pr:
            return float(pr[k])
    raise KeyError("noul")


def _nilai_score(a: dict) -> float:
    """Posisi pada skala level (0..n-1). Pakai 'score'; jika tidak ada, hitung rata-rata berbobot."""
    if a.get("score") is not None:
        return float(a["score"])
    pr = a.get("probabilities") or {}
    if pr:
        return sum(float(p) * i for i, (_, p) in enumerate(sorted(pr.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])))
    raise KeyError("score")


def cek_sumopod():
    """Tampilkan model Jev yang tersedia di akun SumoPod Anda."""
    url = os.environ.get("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1") + "/models"
    try:
        data = _post_sumopod(url, None)
    except urllib.error.HTTPError as e:
        sys.exit(f"Gagal membaca daftar model ({e.code}): {e.read().decode(errors='replace')[:200]}")
    ids = [m.get("id", "") for m in data.get("data", [])]
    jev = [i for i in ids if "jev" in i.lower()]
    print(f"{len(ids)} model di akun SumoPod Anda. Model Jev: {', '.join(jev) if jev else 'tidak terlihat di /models (cek Decision playground SumoPod)'}")
    print(f"Endpoint Jev yang akan dipakai skrip: POST {SUMOPOD_JEV_URL}  (model {SUMOPOD_JEV_MODEL})")


PROMPT_KLASIFIKASI = """You triage customer-service messages for an Indonesian e-commerce shop.
Return ONLY a JSON object with these keys:
  "department": one of {depts}
  "department_confidence": number 0-1
  "urgent": number 0-1, probability it must be handled within hours
  "refund": number 0-1, probability the customer explicitly asks for money back
  "frustration": number 0-2 (0 calm, 1 annoyed but polite, 2 very angry)"""


def ambil_json(teks: str) -> dict:
    """LLM kadang membungkus JSON dengan teks/```; ambil objek pertamanya."""
    m = re.search(r"\{.*\}", teks, re.S)
    if not m:
        raise ValueError(f"Tidak ada JSON di jawaban LLM: {teks[:120]!r}")
    return json.loads(m.group(0))


def tanya_llm_sumopod(tiket) -> tuple:
    """PEMBANDING: tugas yang sama dikerjakan LLM generatif. Confidence di sini hanya
    angka yang DITULIS model (verbal), bukan probabilitas terkalibrasi seperti Jev."""
    teks, usage, dt = sumopod_chat([
        {"role": "system", "content": PROMPT_KLASIFIKASI.format(depts=list(DEPARTEMEN))},
        {"role": "user", "content": json.dumps({"channel": tiket["kanal"], "message": tiket["pesan"]}, ensure_ascii=False)},
    ], max_tokens=200, json_mode=True)
    d = ambil_json(teks)
    dept = d.get("department") if d.get("department") in DEPARTEMEN else "lainnya"
    j = Jawaban(dept=dept, dept_conf=float(d.get("department_confidence", 0)), dept_probs={},
                urgent=float(d.get("urgent", 0)), refund=float(d.get("refund", 0)),
                frustration=float(d.get("frustration", 0)), input_tokens=usage.get("prompt_tokens", 0))
    return j, usage.get("completion_tokens", 0), dt


def draf_balasan_sumopod(tiket, j: Jawaban, keputusan: str) -> str:
    """Sistem 2: LLM menulis draf balasan, setelah Jev (Sistem 1) memutuskan jalurnya."""
    teks, _, _ = sumopod_chat([
        {"role": "system", "content": "Kamu agen CS e-commerce Indonesia. Tulis draf balasan singkat (maks 3 kalimat), "
                                      "sopan, Bahasa Indonesia, tanpa menjanjikan hal di luar keputusan internal."},
        {"role": "user", "content": f"Pesan pelanggan: {tiket['pesan']}\nTim: {j.dept}\nKeputusan internal: {keputusan}"},
    ], max_tokens=200)
    return teks.strip()


# --------------------------------------------------------------------------------------
# 6) Jalankan
# --------------------------------------------------------------------------------------
def jalankan_pembanding():
    print(f"\nPEMBANDING · tugas yang sama dengan LLM generatif via SumoPod · model: {SUMOPOD_MODEL}\n")
    print(f"{'Tiket':<7}{'Departemen':<15}{'conf*':>6}{'urgent':>8}{'refund':>8}{'frust':>7}{'detik':>7}  Keputusan kode")
    print("-" * 110)
    tin = tout = 0
    for t in TIKET:
        j, out_tok, dt = tanya_llm_sumopod(t)
        tin += j.input_tokens; tout += out_tok
        print(f"{t['id']:<7}{j.dept:<15}{j.dept_conf:>6.2f}{j.urgent:>8.2f}{j.refund:>8.2f}{j.frustration:>7.2f}{dt:>7.2f}  {putuskan(j)}")
    print(f"\nToken input {tin:,} · token OUTPUT {tout:,} (dibayar, tarif cek di dashboard SumoPod)")
    print("* conf di sini angka yang ditulis LLM, bukan probabilitas terkalibrasi. Bandingkan dengan mode Jev.\n")


def main():
    ap = argparse.ArgumentParser(description="Demo triase tiket dengan JEV AI (+ SumoPod)")
    ap.add_argument("--simulasi", action="store_true", help="pakai jawaban Jev contoh, tanpa memanggil TypeSafe")
    ap.add_argument("--draf-sumopod", action="store_true", help="LLM via SumoPod menulis draf balasan setelah Jev memutuskan")
    ap.add_argument("--pembanding-sumopod", action="store_true", help="kerjakan klasifikasi dengan LLM via SumoPod (tanpa Jev)")
    ap.add_argument("--jev-sumopod", action="store_true", help="panggil Jev (jev-1.13.0) lewat SumoPod, bukan TypeSafe langsung")
    ap.add_argument("--cek-sumopod", action="store_true", help="cek model Jev yang tersedia di akun SumoPod")
    args = ap.parse_args()

    if args.cek_sumopod:
        return cek_sumopod()
    if args.pembanding_sumopod:
        return jalankan_pembanding()

    if args.jev_sumopod:
        tanya, label = tanya_jev_sumopod, f"LIVE {SUMOPOD_JEV_MODEL} via SumoPod"
    elif args.simulasi:
        tanya, label = tanya_jev_simulasi, "SIMULASI (angka ilustratif)"
    else:
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("TYPESAFE_API_KEY belum di-set. Pakai --jev-sumopod (key SumoPod) atau --simulasi.")
        tanya, label = tanya_jev_live, f"LIVE {MODEL} via TypeSafe"

    print(f"\nDEMO JEV AI · Triase tiket · mode: {label}"
          f"{' · draf balasan: SumoPod ' + SUMOPOD_MODEL if args.draf_sumopod else ''}\n")
    kepala = f"{'Tiket':<7}{'Departemen':<15}{'conf':>6}{'urgent':>8}{'refund':>8}{'frust':>7}  Keputusan kode"
    print(kepala)
    print("-" * (len(kepala) + 30))

    total_token = 0
    for t in TIKET:
        j = tanya(t)
        total_token += j.input_tokens
        keputusan = putuskan(j)
        print(f"{t['id']:<7}{j.dept:<15}{j.dept_conf:>6.2f}{j.urgent:>8.2f}{j.refund:>8.2f}{j.frustration:>7.2f}  {keputusan}")
        print(f"{'':<7}“{t['pesan'][:78]}{'…' if len(t['pesan']) > 78 else ''}”")
        # Hanya tiket yang boleh jalan tanpa manusia yang dibuatkan draf oleh LLM
        if args.draf_sumopod and not keputusan.startswith(("Ke agen manusia", "Eskalasi")):
            print(f"{'':<7}✉ Draf (SumoPod): {draf_balasan_sumopod(t, j, keputusan)}")
        print()

    biaya = total_token / 1_000_000 * HARGA_PER_JUTA_TOKEN
    print(f"Total token input: {total_token:,}  →  biaya ≈ US${biaya:.6f} untuk {len(TIKET)} tiket "
          f"(≈ US${biaya / len(TIKET) * 1_000_000:,.0f} per 1 juta tiket)")
    print("Catatan: 1 panggilan per tiket, 4 pertanyaan dijawab paralel di dalamnya.\n")


if __name__ == "__main__":
    main()
