"""Faster-Whisper Tabanlı Çoklu Dublaj Adayı Hakemi (Kelime Doğruluğu %75 + Doğal Süre %25).

Özellikler:
- Üretilen N adet Türkçe dublaj adayını (Aday 1..N) faster-whisper deşifresi ve
  süre ölçümüyle otomatik puanlar:
  1. Metin Doğruluğu (%75 ağırlık):
     Hedef Türkçe dublaj metni ile adayın Whisper Türkçe deşifresi arasındaki
     kelime ve karakter düzeyinde normalize edilmiş benzerlik. Kelime yutma,
     harf bozulması, kekeleme/tekrar veya halüsinasyon yapan adayların puanı düşürülür.
  2. Doğal Süre Uyumu (%25 ağırlık):
     Orijinal konuşma süresine hızlandırma gerektirmeden en yakın ve doğal
     tempodaki aday öne geçer.
- En yüksek puanlı adayı otomatik olarak `is_best=True` işaretler ve döndürür.
"""

from __future__ import annotations

import difflib
import re
from typing import Sequence

from core.models import DubCandidate

TEXT_ACCURACY_WEIGHT: float = 0.75
DURATION_FIT_WEIGHT: float = 0.25

_TR_LOWER_MAP = str.maketrans(
    {
        "İ": "i",
        "I": "ı",
        "Ç": "ç",
        "Ğ": "ğ",
        "Ö": "ö",
        "Ş": "ş",
        "Ü": "ü",
    }
)

_TR_ASCII_FOLD_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)


def normalize_turkish_text(text: str) -> str:
    """Türkçe büyük/küçük harf kurallarına uygun olarak noktalama işaretlerini temizler."""
    raw = (text or "").translate(_TR_LOWER_MAP).lower()
    # Parantez içi olası oyunculuk notu kalıntısı varsa temizle
    if raw.startswith("(") and ")" in raw:
        raw = raw.split(")", 1)[1]
    cleaned = re.sub(r"[^a-z0-9çğıöşüâîû\s]", " ", raw)
    return " ".join(cleaned.split())


def fold_turkish_diacritics(norm_text: str) -> str:
    """Türkçe özel harfleri ASCII eşdeğerlerine dönüştürür (küçük Whisper şapka/nokta farkları için)."""
    return norm_text.translate(_TR_ASCII_FOLD_MAP)


def compute_text_accuracy_score(target_tr_text: str, whisper_tr_text: str) -> float:
    """Hedef Türkçe metin ile Whisper deşifresi arasındaki doğruluğu 0..100 arasında puanlar."""
    norm_target = normalize_turkish_text(target_tr_text)
    norm_whisper = normalize_turkish_text(whisper_tr_text)

    if not norm_target:
        return 100.0
    if not norm_whisper:
        return 0.0

    fold_target = fold_turkish_diacritics(norm_target)
    fold_whisper = fold_turkish_diacritics(norm_whisper)

    target_words = fold_target.split()
    whisper_words = fold_whisper.split()
    if not whisper_words:
        return 0.0

    # 1. Karakter düzeyinde benzerlik (tam Türkçe + fonetik katlanmış)
    char_exact_ratio = difflib.SequenceMatcher(None, norm_target, norm_whisper).ratio()
    char_folded_ratio = difflib.SequenceMatcher(None, fold_target, fold_whisper).ratio()
    char_score = 0.45 * char_exact_ratio + 0.55 * char_folded_ratio

    # 2. Kelime dizisi benzerliği ve kelime kapsama (kelime yutma / eksik okuma tespiti)
    word_seq_ratio = difflib.SequenceMatcher(None, target_words, whisper_words).ratio()

    matched_words = 0
    used_indices: set[int] = set()
    for tw in target_words:
        best_sim = 0.0
        best_idx = -1
        for idx, ww in enumerate(whisper_words):
            if idx in used_indices:
                continue
            sim = 1.0 if tw == ww else difflib.SequenceMatcher(None, tw, ww).ratio()
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_sim >= 0.78 and best_idx >= 0:
            used_indices.add(best_idx)
            matched_words += best_sim

    word_recall = matched_words / max(1, len(target_words))

    # 3. Uzunluk / Halüsinasyon / Kekeleme (ardışık kelime tekrarı) cezası
    len_ratio = min(len(target_words), len(whisper_words)) / max(
        1, max(len(target_words), len(whisper_words))
    )
    # Tek kelimelik veya kısa cümlelerde küçük ek/çekim farklarını aşırı cezalandırma,
    # ancak ciddi kelime yutma veya halüsinasyonda puanı düşür
    length_factor = min(1.0, 0.35 + 0.65 * len_ratio)

    stutter_duplicates = 0
    for i in range(1, len(whisper_words)):
        if whisper_words[i] == whisper_words[i - 1]:
            # Hedef metinde aynı ardışık tekrar yoksa kekeleme/tekrar say
            if i >= len(target_words) or target_words[i] != target_words[i - 1]:
                stutter_duplicates += 1
    stutter_penalty = stutter_duplicates * 12.0

    raw_score = (0.45 * char_score + 0.30 * word_seq_ratio + 0.25 * word_recall) * 100.0
    penalized = raw_score * length_factor - stutter_penalty
    return round(max(0.0, min(100.0, penalized)), 2)


def compute_duration_fit_score(
    candidate_duration_ms: int,
    original_duration_ms: int,
    max_tempo_speedup: float = 1.05,
) -> float:
    """Adayın konuşma süresinin orijinal süreye doğal uyumunu 0..100 arasında puanlar."""
    orig_ms = max(150.0, float(original_duration_ms))
    cand_ms = max(50.0, float(candidate_duration_ms))
    ratio = cand_ms / orig_ms

    # İdeal doğal aralık: 0.90x .. 1.02x (hızlandırma gerektirmeden tam oturan süre)
    if 0.90 <= ratio <= 1.02:
        score = 100.0 - abs(1.00 - ratio) * 45.0
    elif 1.02 < ratio <= max(1.03, float(max_tempo_speedup)):
        # Doğal tempo sınırları içinde hafif uzun (1.02..1.05)
        score = 95.0 - (ratio - 1.02) * 280.0
    elif ratio > max(1.03, float(max_tempo_speedup)):
        # Hızlandırma sınırını aşan aşırı uzun adayları belirgin şekilde düşür
        over = ratio - max(1.03, float(max_tempo_speedup))
        score = 84.0 - over * 145.0
    else:
        # Orijinalden kısa adaylar (<0.90x): çok kısa ("kelime yutmuş") adayları düşür
        under = 0.90 - ratio
        score = 95.0 - under * 115.0

    return round(max(0.0, min(100.0, score)), 2)


def score_candidate(
    candidate: DubCandidate,
    target_tr_text: str,
    original_duration_ms: int,
    max_tempo_speedup: float = 1.05,
) -> DubCandidate:
    """Tek bir adayın metin doğruluğu (%75), doğal süre uyumu (%25) ve toplam puanını hesaplar."""
    dur_ms = candidate.raw_duration_ms or candidate.synced_duration_ms or original_duration_ms
    txt_score = compute_text_accuracy_score(target_tr_text, candidate.whisper_transcript)
    dur_score = compute_duration_fit_score(dur_ms, original_duration_ms, max_tempo_speedup)
    total = round(TEXT_ACCURACY_WEIGHT * txt_score + DURATION_FIT_WEIGHT * dur_score, 2)

    candidate.text_accuracy_score = txt_score
    candidate.duration_fit_score = dur_score
    candidate.total_score = total
    return candidate


def select_best_candidate(candidates: Sequence[DubCandidate]) -> DubCandidate:
    """Aday listesi içinden en yüksek puanlı adayı seçer ve `is_best` bayrağını günceller."""
    if not candidates:
        raise ValueError("Seçilecek aday listesi boş olamaz.")

    for c in candidates:
        c.is_best = False

    best = max(
        candidates,
        key=lambda c: (round(c.total_score, 2), round(c.text_accuracy_score, 2), -c.candidate_index),
    )
    best.is_best = True
    return best
