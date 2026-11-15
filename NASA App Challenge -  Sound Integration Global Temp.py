import os
import time
import wave
import shutil
import tempfile
import threading
import subprocess
import cv2
import numpy as np
import sounddevice as sd

# ==================== FILE PATHS ====================
VIDEO_PATH   = r"C:\Users\Ash\Downloads\2023GISTEMP_Map_HD.mp4"
SPECTRUM_TXT = r"C:\Users\Ash\Downloads\spectrum_hsv.txt"
OUT_MP4      = r"C:\Users\Ash\Downloads\global_heat_sonification.mp4"
# ====================================================

# ==================== SETTINGS ======================
START_YEAR    = 1880
END_YEAR      = 2023
BEAT_END_YEAR = 2020   # Beat reaches full added BPM at 2020

NUM_BINS      = 18
PIXEL_STRIDE  = 4
MAX_HSV_DIST  = 75.0
SAMPLE_RATE   = 44100
BLOCK_SIZE    = 1024

EXP_W, EXP_H  = 1280, 720
MAIN_WIN      = "Global Heat Exhibition"
CTRL_WIN      = "Exhibition Controls"
# ====================================================

# 18 frequencies capped at 300 Hz max (Cold 49Hz -> Hot 300Hz)
HEAT_FREQS = np.array([
    49.00, 55.00, 65.41, 73.42, 82.41, 87.31,
    98.00, 110.00, 123.47, 130.81, 146.83, 164.81,
    174.61, 196.00, 220.00, 246.94, 261.63, 300.00
], dtype=np.float64)

_pos         = np.linspace(0.0, 1.0, NUM_BINS)
BIN_PAN      = np.linspace(-0.35, 0.35, NUM_BINS)
BASE_WEIGHTS = (0.12 + 0.88 * (np.abs(_pos - 0.5) * 2.0) ** 1.35) * np.clip((140.0 / HEAT_FREQS) ** 0.55, 0.28, 1.35)
HOT_MASK     = np.maximum(0.0, _pos - 0.55) / 0.45

PAD_FREQS = np.array([43.65, 49.00, 55.00, 65.41, 73.42, 82.41, 98.00, 110.00, 123.47, 130.81, 146.83, 164.81, 196.00, 246.94])
COOL_CHORDS = np.array([
    [0, 0, 0, 1.0, 0, 0, 0.8, 0, 0, 0, 0.5, 0.5, 0, 0.4],
    [0, 0, 0.9, 0, 0, 0.8, 0, 0, 0.4, 0.6, 0, 0, 0.5, 0],
    [0.9, 0, 0, 0.8, 0, 0, 0.6, 0.6, 0, 0, 0, 0.5, 0, 0],
    [0, 0.9, 0, 0, 0.8, 0, 0, 0.6, 0, 0.6, 0, 0.4, 0, 0],
], dtype=np.float64)
HOT_CHORDS = np.array([
    [0, 0, 0.9, 0, 0, 0.8, 0, 0.5, 0.7, 0.6, 0, 0.4, 0, 0.5],
    [0.9, 0, 0, 0.7, 0, 0, 0, 0.4, 0.7, 0, 0, 0.6, 0.5, 0],
    [0, 0.9, 0, 0, 0.8, 0, 0, 0.7, 0, 0.6, 0.6, 0, 0, 0.6],
    [0, 0, 0, 0, 0.4, 0.9, 0.6, 0, 0.7, 0, 0.6, 0, 0.6, 0.7],
], dtype=np.float64)

OST_RATIOS  = np.array([1.0, 1.5, 1.25, 1.5, 1.0, 1.5, 1.3333, 1.5])
OST_ACCENTS = np.array([1.0, 0.72, 0.85, 0.70, 0.92, 0.72, 0.82, 0.75])
BEAT_MODES  = ["VIOLIN", "PULSE", "OFF"]


def soft_limit(x, thresh=0.84):
    ax = np.abs(x)
    over = ax > thresh
    if not np.any(over):
        return x
    out, head = x.copy(), 1.0 - thresh
    out[over] = np.sign(x[over]) * (thresh + head * np.tanh((ax[over] - thresh) / head))
    return out


def frame_to_year(frame_idx, total_frames):
    frac = np.clip(frame_idx / max(1.0, total_frames - 1.0), 0.0, 1.0)
    return float(START_YEAR + (END_YEAR - START_YEAR) * frac)


def compute_year_bpm(year, b_bpm, a_bpm):
    prog = np.clip((year - START_YEAR) / max(1e-6, BEAT_END_YEAR - START_YEAR), 0.0, 1.0)
    return float(np.clip(b_bpm + a_bpm * prog, 30.0, 650.0))


def compute_heat_score(mean_h, hist):
    warm_f = float(np.sum(hist[12:]))
    cold_f = float(np.sum(hist[:6]))
    bal = np.clip(0.5 + 0.9 * (warm_f - cold_f), 0.0, 1.0)
    return float(np.clip(0.55 * mean_h + 0.45 * bal, 0.0, 1.0))


def stripe_bgr(val):
    """Hawkins-style warming stripe palette (BGR): Deep Blue -> Soft Slate -> Amber -> Deep Crimson."""
    t = float(np.clip(val, 0.0, 1.0))
    stops = [
        (0.00, (165,  60,  15)),
        (0.28, (235, 145,  55)),
        (0.48, (215, 210, 200)),
        (0.66, ( 55, 145, 245)),
        (0.84, ( 35,  65, 225)),
        (1.00, ( 25,  25, 165)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            return tuple(int(c0[k] + u * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


def era_caption(yr):
    if yr < 1920:   return "PRE-INDUSTRIAL BASELINE"
    if yr < 1950:   return "EARLY INDUSTRIAL SHIFT"
    if yr < 1980:   return "POST-WAR ACCELERATION"
    if yr < 2000:   return "RAPID WARMING ERA"
    if yr < 2015:   return "DECADE OF EXTREMES"
    return "UNCHARTED GLOBAL HEAT"


class SynthEngine:
    def __init__(self):
        self.d_cap = int(0.6 * SAMPLE_RATE)
        self.d_L, self.d_R = int(0.31 * SAMPLE_RATE), int(0.43 * SAMPLE_RATE)
        self.d_buf = np.zeros((self.d_cap, 2), dtype=np.float64)
        self.d_idx = 0

        self.amps, self.pans = np.zeros(NUM_BINS), BIN_PAN.copy()
        self.ph_L, self.ph_R, self.sub_ph = np.zeros(NUM_BINS), np.zeros(NUM_BINS), np.zeros(NUM_BINS)
        self.pad_L, self.pad_R, self.pad_amps = np.zeros(len(PAD_FREQS)), np.zeros(len(PAD_FREQS)), np.zeros(len(PAD_FREQS))
        self.clock = 0.0
        self.lp = np.zeros(2)

        self.max_v = 10
        self.v_on, self.v_age, self.v_step, self.v_bpm = [False]*10, [0]*10, [0]*10, [60.0]*10
        self.b_phase, self.b_count = 0.999, 0

    def reset_beats(self):
        self.v_on = [False] * self.max_v
        self.b_phase, self.b_count = 0.999, 0

    def _beats(self, n, bpm, b_vol, b_mode, heat, running):
        bL, bR = np.zeros(n), np.zeros(n)
        if running and b_vol > 0.001 and b_mode != "OFF":
            dp = bpm / (60.0 * SAMPLE_RATE)
            end_p = self.b_phase + n * dp
            while end_p >= 1.0:
                hit = int(np.clip((1.0 - self.b_phase) / dp, 0, n - 1))
                v = self.b_count % self.max_v
                self.v_on[v], self.v_age[v], self.v_step[v], self.v_bpm[v] = True, -hit, self.b_count % 8, bpm
                self.b_count += 1
                self.b_phase -= 1.0
                end_p -= 1.0
            self.b_phase = end_p

        idx = np.arange(n, dtype=np.float64)
        drive = 0.80 + 0.45 * np.clip(heat, 0.0, 1.0)

        for v in range(self.max_v):
            if not self.v_on[v]:
                continue
            age = idx + self.v_age[v]
            mask = age >= 0.0
            if np.any(mask):
                t = np.where(mask, age / SAMPLE_RATE, 0.0)
                st = self.v_step[v]
                decay = np.clip(11.0 / max(30.0, self.v_bpm[v]), 0.024, 0.14)
                if b_mode == "VIOLIN":
                    nf = 196.0 * OST_RATIOS[st]  # <= 294 Hz
                    env = np.where(mask, np.minimum(1.0, t / 0.008) * np.exp(-t / decay), 0.0) * OST_ACCENTS[st]
                    ks = np.arange(1, 6)[:, None]
                    hg = (1.0 / (ks ** 1.35)) * np.exp(-(ks * nf / (1400.0 + 400.0 * heat)) ** 2)
                    sL = np.sum(hg * np.sin(2 * np.pi * (ks * nf + 0.7) * t[None, :]), axis=0)
                    sR = np.sum(hg * np.sin(2 * np.pi * (ks * nf - 0.7) * t[None, :]), axis=0)
                    body = 0.22 * np.exp(-t / 0.022) * np.sin(np.pi * nf * t)
                    g = b_vol * 0.24 * np.clip((200.0 / nf) ** 0.45, 0.4, 1.25) * drive
                    bL += env * (sL + body) * g
                    bR += env * (sR + body) * g
                else:
                    env = np.where(mask, np.minimum(1.0, t / 0.007) * np.exp(-t / (decay * 0.85)), 0.0) * OST_ACCENTS[st]
                    pw = 0.75 * np.sin(2 * np.pi * 98.0 * t) + (0.35 + 0.2 * heat) * np.sin(2 * np.pi * 196.0 * t) * np.exp(-t / 0.028)
                    g = b_vol * 0.34 * drive
                    bL += env * pw * g
                    bR += env * pw * g
            self.v_age[v] += n
            if self.v_age[v] > int(0.45 * SAMPLE_RATE):
                self.v_on[v] = False
        return bL, bR

    def render(self, n, tgt_a, tgt_p, heat, m_vol, bpm, b_mode, running=True, muted=False):
        tgt = np.zeros_like(tgt_a) if muted else tgt_a
        a_vol = 0.0 if muted else 0.48
        b_vol = 0.0 if muted else 0.65

        s_a, e_a = self.amps.copy(), self.amps + 0.25 * (tgt - self.amps)
        s_p, e_p = self.pans.copy(), self.pans + 0.20 * (tgt_p - self.pans)
        self.amps, self.pans = e_a, e_p

        t_i = np.arange(n, dtype=np.float64)[None, :]
        ramp = np.linspace(0.0, 1.0, n)[None, :]
        env = s_a[:, None] + (e_a - s_a)[:, None] * ramp
        pan = s_p[:, None] + (e_p - s_p)[:, None] * ramp
        gL, gR = np.cos((pan + 1.0) * np.pi / 4.0), np.sin((pan + 1.0) * np.pi / 4.0)

        phL = self.ph_L[:, None] + 2 * np.pi * (HEAT_FREQS[:, None] + 0.14) / SAMPLE_RATE * t_i
        phR = self.ph_R[:, None] + 2 * np.pi * (HEAT_FREQS[:, None] - 0.14) / SAMPLE_RATE * t_i
        lfo_t = self.clock + np.arange(n) / SAMPLE_RATE
        shim = 1.0 + (0.14 * heat) * HOT_MASK[:, None] * np.sin(2 * np.pi * 4.2 * lfo_t[None, :])
        h2 = 0.28 * (0.35 + 0.65 * heat) * HOT_MASK[:, None]
        h3 = 0.12 * heat * (HOT_MASK[:, None] ** 1.4)

        sigL = np.sum(env * shim * gL * (np.sin(phL) + h2 * np.sin(2 * phL) + h3 * np.sin(3 * phL)), axis=0)
        sigR = np.sum(env * shim * gR * (np.sin(phR) + h2 * np.sin(2 * phR) + h3 * np.sin(3 * phR)), axis=0)

        sub = np.sum(env * HOT_MASK[:, None] * (0.38 + 0.22 * heat) *
                     np.sin(self.sub_ph[:, None] + np.pi * HEAT_FREQS[:, None] / SAMPLE_RATE * t_i), axis=0)
        sigL += sub
        sigR += sub

        self.ph_L = (self.ph_L + 2 * np.pi * (HEAT_FREQS + 0.14) * n / SAMPLE_RATE) % (2 * np.pi)
        self.ph_R = (self.ph_R + 2 * np.pi * (HEAT_FREQS - 0.14) * n / SAMPLE_RATE) % (2 * np.pi)
        self.sub_ph = (self.sub_ph + np.pi * HEAT_FREQS * n / SAMPLE_RATE) % (2 * np.pi)

        self.clock += n / SAMPLE_RATE
        cp = (self.clock / 9.0) % len(COOL_CHORDS)
        c0, c1 = int(cp), (int(cp) + 1) % len(COOL_CHORDS)
        bl = 0.5 - 0.5 * np.cos(np.pi * (cp - c0))
        tens = float(np.clip((heat - 0.35) / 0.45, 0.0, 1.0))
        ptgt = ((1 - tens) * ((1 - bl) * COOL_CHORDS[c0] + bl * COOL_CHORDS[c1]) +
                tens * ((1 - bl) * HOT_CHORDS[c0] + bl * HOT_CHORDS[c1])) * a_vol * 0.22
        ptgt *= 0.74 + 0.26 * np.sin(2 * np.pi * 0.10 * self.clock)
        ptgt[7:] *= 0.65 + 0.60 * heat

        ps, pe = self.pad_amps.copy(), self.pad_amps + 0.08 * (ptgt - self.pad_amps)
        self.pad_amps = pe
        penv = ps[:, None] + (pe - ps)[:, None] * ramp
        padL = np.sum(penv * np.sin(self.pad_L[:, None] + 2 * np.pi * (PAD_FREQS[:, None] - 0.3) / SAMPLE_RATE * t_i), axis=0)
        padR = np.sum(penv * np.sin(self.pad_R[:, None] + 2 * np.pi * (PAD_FREQS[:, None] + 0.3) / SAMPLE_RATE * t_i), axis=0)
        self.pad_L = (self.pad_L + 2 * np.pi * (PAD_FREQS - 0.3) * n / SAMPLE_RATE) % (2 * np.pi)
        self.pad_R = (self.pad_R + 2 * np.pi * (PAD_FREQS + 0.3) * n / SAMPLE_RATE) % (2 * np.pi)

        bL, bR = self._beats(n, bpm, b_vol, b_mode, heat, running and not muted)
        dry = np.stack([sigL * 0.45 + padL + bL, sigR * 0.45 + padR + bR], axis=1)

        ia = (self.d_idx + np.arange(n)) % self.d_cap
        wL, wR = self.d_buf[(ia - self.d_L) % self.d_cap, 0], self.d_buf[(ia - self.d_R) % self.d_cap, 1]
        self.d_buf[ia, 0] = soft_limit(sigL * 0.45 + padL + bL * 0.55 + 0.36 * (0.78 * wL + 0.22 * wR), 0.90)
        self.d_buf[ia, 1] = soft_limit(sigR * 0.45 + padR + bR * 0.55 + 0.36 * (0.78 * wR + 0.22 * wL), 0.90)
        self.d_idx = (self.d_idx + n) % self.d_cap

        mix = (dry + 0.40 * np.stack([wL, wR], axis=1)) * m_vol * 0.85
        alpha = 0.16 + 0.38 * float(np.clip(heat, 0.0, 1.0))
        lp = self.lp
        for i in range(n):
            lp = lp + alpha * (mix[i] - lp)
            mix[i] = lp
        self.lp = lp
        return soft_limit(mix, 0.84).astype(np.float32)


# Shared audio state
live_engine = SynthEngine()
lock = threading.Lock()
tgt_amps, tgt_pans = np.zeros(NUM_BINS), BIN_PAN.copy()
is_playing, is_muted, reset_beats_req = True, False, False
master_vol, hot_boost, play_speed = 0.75, 1.25, 1.0
base_bpm, add_bpm = 60.0, 180.0
mean_heat_rt, live_bpm_rt, beat_pulse_rt, beat_mode_i = 0.5, 60.0, 0.0, 0


def build_hsv_lut(spec_hsv):
    def cyl(h, s, v):
        th = h * (2.0 * np.pi / 180.0)
        return np.stack([s * np.cos(th), s * np.sin(th), v], axis=-1)

    spec_xyz = cyl(spec_hsv[:, 0], spec_hsv[:, 1], spec_hsv[:, 2])
    s_v, v_v = np.meshgrid(np.arange(64) * 4.0 + 1.5, np.arange(64) * 4.0 + 1.5, indexing="ij")
    sf, vf = s_v.ravel(), v_v.ravel()
    lut = np.full((180, 64, 64), -1, dtype=np.int16)
    for h in range(180):
        d2 = np.sum((cyl(np.full_like(sf, float(h)), sf, vf)[:, None, :] - spec_xyz[None, :, :]) ** 2, axis=2)
        ni = np.argmin(d2, axis=1)
        bins = (ni * NUM_BINS) // len(spec_hsv)
        bins[np.sqrt(d2[np.arange(len(ni)), ni]) > MAX_HSV_DIST] = -1
        lut[h] = bins.reshape(64, 64)
    return lut


def detect_earth_mask(cap):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_fr = None
    for fid in np.linspace(0, max(0, n - 1), 6, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, fr = cap.read()
        if ok:
            max_fr = fr if max_fr is None else np.maximum(max_fr, fr)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if max_fr is None:
        return np.ones((int(cap.get(4) or 720), int(cap.get(3) or 1280)), dtype=np.uint8), []

    closed = cv2.morphologyEx((np.max(max_fr, axis=2) > 18).astype(np.uint8),
                              cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, lbl, stats, _ = cv2.connectedComponentsWithStats(closed)
    if len(stats) <= 1:
        return np.ones_like(closed), []
    mask = (lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled, contours


def audio_cb(outdata, frames, time_info, status):
    global reset_beats_req, beat_pulse_rt
    with lock:
        if reset_beats_req:
            live_engine.reset_beats()
            reset_beats_req = False
        ta, tp, hv, mv, bpm = tgt_amps.copy(), tgt_pans.copy(), mean_heat_rt, master_vol, live_bpm_rt
        bm, run, mut = BEAT_MODES[beat_mode_i], is_playing, is_muted

    outdata[:] = live_engine.render(frames, ta, tp, hv, mv, bpm, bm, running=run, muted=mut)
    with lock:
        beat_pulse_rt = float(max(0.0, 1.0 - live_engine.b_phase)) if (run and bm != "OFF") else 0.0


def render_exhibition(disp_vid, hist, heat_score, stripe_scores, curr_idx, n_frames,
                      curr_yr, bpm, b_pulse, contours):
    """Pure 1280x720 museum exhibition canvas — zero knobs, zero debug charts."""
    canvas = np.full((EXP_H, EXP_W, 3), (14, 12, 10), dtype=np.uint8)
    hcol = stripe_bgr(heat_score)
    yr_int = int(round(curr_yr))

    # --- Header Placard ---
    cv2.putText(canvas, "THE SOUND OF A WARMING PLANET", (36, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 238, 244), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GLOBAL SURFACE TEMPERATURE ANOMALIES  |  1880 - 2023", (38, 63),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (135, 142, 160), 1, cv2.LINE_AA)

    # Top-right heartbeat pulse pill
    bm = BEAT_MODES[beat_mode_i]
    dot_r = 5 + int(round(b_pulse * 4))
    dot_col = hcol if bm != "OFF" else (90, 95, 110)
    cv2.circle(canvas, (1015, 42), dot_r, dot_col, -1, cv2.LINE_AA)
    hb_txt = f"EARTH PULSE: {int(round(bpm))} BPM" if bm != "OFF" else "AMBIENT MODE"
    cv2.putText(canvas, hb_txt, (1032, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 212, 225), 1, cv2.LINE_AA)

    # --- Left: Hero Earth Map with subtle breathing thermal aura ---
    vp = disp_vid.copy()
    if contours:
        glow = vp.copy()
        cv2.drawContours(glow, contours, -1, hcol, 6 + int(round(b_pulse * 4)), cv2.LINE_AA)
        cv2.addWeighted(glow, 0.28 + 0.15 * b_pulse, vp, 0.72 - 0.15 * b_pulse, 0.0, dst=vp)
        cv2.drawContours(vp, contours, -1, hcol, 2, cv2.LINE_AA)
    canvas[78:494, 36:776] = vp
    cv2.rectangle(canvas, (35, 77), (777, 495), (42, 46, 56), 1)

    # --- Right Column: Giant Year, Era Narrative, Fever Gauge, Planet Share ---
    rx = 808
    cv2.rectangle(canvas, (rx, 77), (1244, 495), (20, 22, 28), -1)
    cv2.rectangle(canvas, (rx, 77), (1244, 495), (42, 46, 56), 1)

    cv2.putText(canvas, "YEAR", (rx + 26, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 148, 165), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{yr_int}", (rx + 22, 184), cv2.FONT_HERSHEY_DUPLEX, 2.75, hcol, 5, cv2.LINE_AA)
    cv2.putText(canvas, era_caption(yr_int), (rx + 26, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (215, 220, 232), 1, cv2.LINE_AA)
    cv2.line(canvas, (rx + 24, 236), (1220, 236), (40, 44, 54), 1)

    # Circular Global Fever Gauge
    gx, gy, gr = rx + 96, 330, 62
    cv2.ellipse(canvas, (gx, gy), (gr, gr), 0, 135, 405, (42, 46, 58), 10, cv2.LINE_AA)
    end_ang = 135 + int(round(270.0 * np.clip(heat_score, 0.0, 1.0)))
    if end_ang > 135:
        cv2.ellipse(canvas, (gx, gy), (gr, gr), 0, 135, end_ang, hcol, 10, cv2.LINE_AA)
    pct_s = f"{int(round(heat_score * 100))}%"
    (tw, _), _ = cv2.getTextSize(pct_s, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    cv2.putText(canvas, pct_s, (gx - tw // 2, gy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 242, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, "HEAT INDEX", (gx - 42, gy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (145, 152, 168), 1, cv2.LINE_AA)

    # Cool vs Warm Zones readout next to gauge
    cold_pct = float(np.sum(hist[:6]) * 100.0)
    warm_pct = float(np.sum(hist[12:]) * 100.0)
    bx = rx + 195
    cv2.putText(canvas, f"COOL ZONES: {cold_pct:4.1f}%", (bx, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 165, 80), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bx, 296), (1218, 308), (38, 42, 52), -1)
    cv2.rectangle(canvas, (bx, 296), (int(bx + np.clip(cold_pct / 75.0, 0, 1) * (1218 - bx)), 308), (235, 145, 55), -1)

    cv2.putText(canvas, f"WARM ZONES: {warm_pct:4.1f}%", (bx, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 140, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bx, 354), (1218, 366), (38, 42, 52), -1)
    cv2.rectangle(canvas, (bx, 354), (int(bx + np.clip(warm_pct / 75.0, 0, 1) * (1218 - bx)), 366), (45, 85, 240), -1)

    pk_txt = "TEMPO PEAK REACHED (2020+)" if yr_int >= BEAT_END_YEAR else f"ACCELERATING TO {BEAT_END_YEAR}"
    cv2.putText(canvas, pk_txt, (bx, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 158, 175), 1, cv2.LINE_AA)

    # Planetary Balance Bar
    cv2.putText(canvas, "PLANETARY THERMAL BALANCE", (rx + 26, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 148, 165), 1, cv2.LINE_AA)
    bar_l, bar_r = rx + 26, 1218
    bw = bar_r - bar_l
    cw_w = int(round(bw * np.clip(cold_pct / 100.0, 0, 1)))
    wm_w = int(round(bw * np.clip(warm_pct / 100.0, 0, 1)))
    cv2.rectangle(canvas, (bar_l, 452), (bar_r, 472), (170, 168, 162), -1)
    if cw_w > 0:
        cv2.rectangle(canvas, (bar_l, 452), (bar_l + cw_w, 472), (230, 140, 50), -1)
    if wm_w > 0:
        cv2.rectangle(canvas, (bar_r - wm_w, 452), (bar_r, 472), (40, 65, 230), -1)

    # --- Bottom: Live Ed Hawkins-style "Warming Stripes" Timeline ---
    sx1, sy1, sx2, sy2 = 36, 536, 1244, 668
    sw = sx2 - sx1
    cv2.putText(canvas, "CLIMATE WARMING STRIPES  (1880 - 2023)", (sx1, 524),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (195, 202, 215), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (24, 26, 33), -1)

    # Render stripes cleanly column-by-column across the timeline width
    cols_drawn = int(np.clip((curr_idx + 1) / max(1, n_frames), 0.0, 1.0) * sw)
    if cols_drawn > 0:
        f_map = np.clip((np.arange(cols_drawn) * (n_frames - 1)) // max(1, sw - 1), 0, n_frames - 1)
        vals = stripe_scores[f_map]
        # Fill any unscanned gap with nearest valid prior value
        last_v = heat_score
        for c in range(cols_drawn):
            v = vals[c]
            if np.isnan(v):
                v = last_v
            else:
                last_v = float(v)
            col_bgr = stripe_bgr(v)
            cv2.line(canvas, (sx1 + c, sy1 + 1), (sx1 + c, sy2 - 1), col_bgr, 1)

    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (55, 60, 72), 1)

    # Year ticks & 2020 beat milestone marker
    for yr_tick in (1880, 1920, 1950, 1980, 2000, 2020, 2023):
        tx = int(sx1 + np.clip((yr_tick - START_YEAR) / max(1, END_YEAR - START_YEAR), 0, 1) * sw)
        tcol = (120, 210, 255) if yr_tick == BEAT_END_YEAR else (155, 162, 175)
        cv2.line(canvas, (tx, sy2 + 1), (tx, sy2 + 7), tcol, 1)
        cv2.putText(canvas, str(yr_tick), (tx - 16, sy2 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, tcol, 1, cv2.LINE_AA)

    play_x = int(sx1 + np.clip(curr_idx / max(1, n_frames - 1), 0.0, 1.0) * sw)
    cv2.line(canvas, (play_x, sy1 - 4), (play_x, sy2 + 4), (245, 248, 255), 2)

    return canvas


# ==================== SEPARATE CONTROL PANEL WINDOW ====================
BTN_BOXES = {
    "play":    (20,  20, 135, 72),
    "mute":    (148, 20, 255, 72),
    "beat":    (268, 20, 412, 72),
    "restart": (425, 20, 520, 72),
    "save":    (532, 20, 660, 72),
}


def draw_ctrl_panel(curr_yr, bpm, heat_score, status_txt="", export_pct=None):
    panel = np.full((190, 680, 3), (24, 27, 34), dtype=np.uint8)
    bm = BEAT_MODES[beat_mode_i]

    def btn(key, label, active, col_on):
        x1, y1, x2, y2 = BTN_BOXES[key]
        cv2.rectangle(panel, (x1, y1), (x2, y2), col_on if active else (48, 53, 66), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (120, 130, 150), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(panel, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 248, 252), 1, cv2.LINE_AA)

    btn("play",    "PAUSE" if is_playing else "PLAY", is_playing, (45, 135, 85))
    btn("mute",    "MUTED" if is_muted else "MUTE",   is_muted,   (165, 55, 55))
    btn("beat",    f"BEAT: {bm}",                     bm != "OFF", (145, 95, 40) if bm == "VIOLIN" else (45, 125, 145))
    btn("restart", "RESTART",                         False,       (60, 65, 80))
    btn("save",    "SAVE MP4",                        True,        (35, 120, 195))

    cv2.putText(panel, f"Year: {int(round(curr_yr))}   |   Live BPM: {int(round(bpm))}   |   Heat Index: {int(round(heat_score*100))}%",
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 228, 240), 1, cv2.LINE_AA)

    msg = status_txt or "Sliders above stay off the exhibition screen. Click SAVE MP4 when ready."
    cv2.putText(panel, msg, (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (165, 210, 255), 1, cv2.LINE_AA)

    if export_pct is not None:
        cv2.rectangle(panel, (20, 150), (660, 162), (45, 50, 62), -1)
        cv2.rectangle(panel, (20, 150), (int(20 + np.clip(export_pct / 100.0, 0, 1) * 640), 162), (55, 175, 255), -1)
    else:
        cv2.putText(panel, "Keys: [Space]=Play/Pause  [M]=Mute  [B]=Beat  [R]=Restart  [S]=Save MP4  [Q]=Quit",
                    (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (140, 148, 165), 1, cv2.LINE_AA)

    cv2.imshow(CTRL_WIN, panel)


def export_mp4(proc_fn, contours, src_fps, n_frames, curr_yr, bpm, heat_score):
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("ERROR: ffmpeg not found! Run: pip install imageio-ffmpeg")
        return "Export failed: install imageio-ffmpeg"

    out_fps = 30.0
    src_dur = n_frames / src_fps
    out_dur = src_dur / play_speed
    n_out = max(1, int(round(out_dur * out_fps)))
    total_samp = int(round((n_out / out_fps) * SAMPLE_RATE))

    cap = cv2.VideoCapture(VIDEO_PATH)
    hists, heats, pans, scores = [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, m, p, sc, _ = proc_fn(fr, False)
        hists.append(h); heats.append(m); pans.append(p); scores.append(sc)
        if len(hists) % 50 == 0:
            pct = int(100 * len(hists) / max(1, n_frames))
            draw_ctrl_panel(curr_yr, bpm, heat_score, f"Exporting (1/2 Scanning Video): {pct}%", pct)
            cv2.waitKey(1)

    hists, heats, pans, scores = np.array(hists), np.array(heats), np.array(pans), np.array(scores)
    n_src = len(hists)
    eng = SynthEngine()
    weights = BASE_WEIGHTS * (1.0 + hot_boost * HOT_MASK)
    chunks, done = [], 0
    bm = BEAT_MODES[beat_mode_i]

    while done < total_samp:
        blk = min(BLOCK_SIZE, total_samp - done)
        t_mid = (done + blk * 0.5) / SAMPLE_RATE
        si_f = min(n_src - 1.0, t_mid * play_speed * src_fps)
        si = int(si_f)
        yr = frame_to_year(si_f, n_src)
        bpm_now = compute_year_bpm(yr, base_bpm, add_bpm)
        chunks.append(eng.render(blk, hists[si] * weights, pans[si], float(heats[si]), master_vol, bpm_now, bm, True, False))
        done += blk

    audio = np.vstack(chunks)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.int16(np.clip(audio * 32767.0, -32768, 32767)).tobytes())

        proc = subprocess.Popen([
            ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{EXP_W}x{EXP_H}",
            "-pix_fmt", "bgr24", "-r", f"{out_fps:.4f}", "-i", "-", "-i", wav_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_MP4
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cached_si, cached_disp, bp = -1, None, 0.0
        for k in range(n_out):
            t_out = k / out_fps
            si_f = min(n_src - 1.0, t_out * play_speed * src_fps)
            si = int(si_f)
            if si != cached_si:
                if si != cached_si + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, si)
                ok, fr = cap.read()
                if ok:
                    _, _, _, _, cached_disp = proc_fn(fr, True)
                    cached_si = si

            yr = frame_to_year(si_f, n_src)
            bpm_now = compute_year_bpm(yr, base_bpm, add_bpm)
            bp = (bp + (bpm_now / 60.0) / out_fps) % 1.0

            frame = render_exhibition(
                cached_disp, hists[si], float(scores[si]), scores, si, n_src,
                yr, bpm_now, max(0.0, 1.0 - bp) if bm != "OFF" else 0.0, contours
            )
            proc.stdin.write(frame.tobytes())
            if k % 25 == 0 or k == n_out - 1:
                pct = int(100 * (k + 1) / n_out)
                draw_ctrl_panel(yr, bpm_now, float(scores[si]), f"Exporting (2/2 Rendering MP4): {pct}%", pct)
                cv2.waitKey(1)

        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="ignore")
        if proc.wait() == 0:
            print(f"\nSaved MP4 -> {OUT_MP4}")
            return f"SAVED! -> {OUT_MP4}"
        print(err)
        return "Export failed (see terminal)"
    finally:
        cap.release()
        if os.path.exists(wav_path):
            os.remove(wav_path)


def main():
    global is_playing, is_muted, beat_mode_i, master_vol, hot_boost, play_speed
    global base_bpm, add_bpm, mean_heat_rt, live_bpm_rt, reset_beats_req

    if not os.path.exists(VIDEO_PATH) or not os.path.exists(SPECTRUM_TXT):
        print("ERROR: Video or spectrum_hsv.txt not found.")
        return

    spec = np.loadtxt(SPECTRUM_TXT, skiprows=1)
    if len(spec) > 10:
        spec = spec[1:-1]
    lut = build_hsv_lut(spec[::-1, 3:6])

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    earth_mask, contours = detect_earth_mask(cap)
    hv, wv = earth_mask.shape
    s_contours = [np.round(c * np.array([740.0 / wv, 416.0 / hv])).astype(np.int32) for c in contours]
    ys, xs = np.where(earth_mask[::PIXEL_STRIDE, ::PIXEL_STRIDE] > 0)
    ys, xs = ys * PIXEL_STRIDE, xs * PIXEL_STRIDE
    xs_norm = (xs.astype(np.float64) / max(1, wv - 1)) * 2.0 - 1.0

    def proc_frame(bgr, need_disp=True):
        hsv = cv2.cvtColor(bgr[ys, xs].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        bins = lut[hsv[:, 0], hsv[:, 1] >> 2, hsv[:, 2] >> 2]
        vm = bins >= 0
        vb = bins[vm]
        if len(vb) > 0:
            cnt = np.bincount(vb, minlength=NUM_BINS).astype(np.float64)
            hist = cnt / cnt.sum()
            mh = float(np.mean(vb)) / (NUM_BINS - 1)
            gp = np.where(cnt > 0, np.bincount(vb, weights=xs_norm[vm], minlength=NUM_BINS) / np.maximum(cnt, 1.0), 0.0)
            pans = np.clip(0.45 * BIN_PAN + 0.55 * gp, -0.75, 0.75)
            hscore = compute_heat_score(mh, hist)
        else:
            hist, mh, pans, hscore = np.zeros(NUM_BINS), 0.5, BIN_PAN.copy(), 0.5
        disp = cv2.resize(bgr, (740, 416)) if need_disp else None
        return hist, mh, pans, hscore, disp

    ok, fr0 = cap.read()
    if not ok:
        return
    hist, mh, pans, hscore, disp_vid = proc_frame(fr0)
    stripe_scores = np.full(n_frames, np.nan, dtype=np.float64)
    stripe_scores[0] = hscore

    # Create two separate windows: Clean Exhibition + Separate Controls
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WIN, EXP_W, EXP_H)

    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)

    seek_req, prog_tb, save_req = None, 0, False
    def on_seek(v):
        nonlocal seek_req
        if v != prog_tb:
            seek_req = v
    def on_vol(v):
        global master_vol
        with lock: master_vol = v / 100.0
    def on_base_bpm(v):
        global base_bpm
        with lock: base_bpm = float(max(30, v))
    def on_add_bpm(v):
        global add_bpm
        with lock: add_bpm = float(v)
    def on_hot(v):
        global hot_boost
        hot_boost = max(0.2, v / 100.0)
    def on_spd(v):
        global play_speed
        play_speed = max(0.25, v / 100.0)

    def on_ctrl_mouse(event, x, y, flags, param):
        nonlocal seek_req, save_req
        global is_playing, is_muted, beat_mode_i, reset_beats_req
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for key, (x1, y1, x2, y2) in BTN_BOXES.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key == "play":
                    with lock: is_playing = not is_playing
                elif key == "mute":
                    with lock: is_muted = not is_muted
                elif key == "beat":
                    with lock: beat_mode_i = (beat_mode_i + 1) % len(BEAT_MODES)
                elif key == "restart":
                    seek_req = 0
                    with lock: reset_beats_req = True
                elif key == "save":
                    save_req = True
                return

    cv2.setMouseCallback(CTRL_WIN, on_ctrl_mouse)
    cv2.createTrackbar("Timeline",  CTRL_WIN, 0,   max(1, n_frames - 1), on_seek)
    cv2.createTrackbar("Volume %",  CTRL_WIN, 75,  100, on_vol)
    cv2.createTrackbar("Base BPM",  CTRL_WIN, 60,  240, on_base_bpm)
    cv2.createTrackbar("Add BPM",   CTRL_WIN, 180, 400, on_add_bpm)
    cv2.createTrackbar("RedBoost%", CTRL_WIN, 125, 250, on_hot)
    cv2.createTrackbar("Speed %",   CTRL_WIN, 100, 300, on_spd)

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK_SIZE, dtype="float32", callback=audio_cb)
    stream.start()

    curr_f, f_float, last_t, status_msg = 0, 0.0, time.perf_counter(), ""

    try:
        while True:
            now = time.perf_counter()
            dt, last_t = min(0.1, now - last_t), now

            if seek_req is not None:
                prev_f = curr_f
                curr_f = int(np.clip(seek_req, 0, n_frames - 1))
                seek_req, f_float = None, float(curr_f)
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
                ok, fr = cap.read()
                if ok:
                    hist, mh, pans, hscore, disp_vid = proc_frame(fr)
                    if curr_f < prev_f:
                        stripe_scores[curr_f + 1:] = np.nan
                    stripe_scores[curr_f] = hscore
                if curr_f == 0:
                    with lock: reset_beats_req = True
            elif is_playing:
                f_float += dt * fps * play_speed
                nxt = int(f_float)
                if nxt >= n_frames:
                    f_float, nxt, curr_f = 0.0, 0, -1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    stripe_scores[:] = np.nan
                    with lock: reset_beats_req = True
                if nxt != curr_f:
                    if nxt - curr_f > 8 or nxt < curr_f:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, nxt)
                    else:
                        for _ in range(nxt - curr_f - 1):
                            cap.grab()
                    ok, fr = cap.read()
                    if ok:
                        prev_score = hscore
                        prev_idx = max(0, curr_f)
                        curr_f = nxt
                        hist, mh, pans, hscore, disp_vid = proc_frame(fr)
                        if curr_f > prev_idx:
                            stripe_scores[prev_idx:curr_f + 1] = np.linspace(prev_score, hscore, curr_f - prev_idx + 1)
                        else:
                            stripe_scores[curr_f] = hscore

            prog_tb = curr_f
            cv2.setTrackbarPos("Timeline", CTRL_WIN, curr_f)

            curr_yr = frame_to_year(curr_f, n_frames)
            weights = BASE_WEIGHTS * (1.0 + hot_boost * HOT_MASK)
            with lock:
                tgt_amps[:] = hist * weights
                tgt_pans[:] = pans
                mean_heat_rt = mh
                live_bpm_rt = compute_year_bpm(curr_yr, base_bpm, add_bpm)
                bpm_d, bp_d = live_bpm_rt, beat_pulse_rt

            cv2.imshow(MAIN_WIN, render_exhibition(
                disp_vid, hist, hscore, stripe_scores, curr_f, n_frames,
                curr_yr, bpm_d, bp_d, s_contours
            ))
            draw_ctrl_panel(curr_yr, bpm_d, hscore, status_msg)

            if save_req:
                save_req = False
                with lock: was_m, is_muted = is_muted, True
                status_msg = export_mp4(proc_frame, s_contours, fps, n_frames, curr_yr, bpm_d, hscore)
                with lock: is_muted = was_m
                last_t = time.perf_counter()

            k = cv2.waitKey(15) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break
            elif k == 32:
                with lock: is_playing = not is_playing
            elif k in (ord('m'), ord('M')):
                with lock: is_muted = not is_muted
            elif k in (ord('b'), ord('B')):
                with lock: beat_mode_i = (beat_mode_i + 1) % len(BEAT_MODES)
            elif k in (ord('r'), ord('R')):
                seek_req = 0
                with lock: reset_beats_req = True
            elif k in (ord('s'), ord('S')):
                save_req = True

            if cv2.getWindowProperty(MAIN_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stream.stop(); stream.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()