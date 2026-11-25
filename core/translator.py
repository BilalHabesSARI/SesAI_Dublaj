"""Sinematik Oyun Dublajı İçin Çoklu Model, Otomatik Geçiş (Fallback) ve Toplu (Batch) Gemini Çevirmeni.

Özellikler:
- 8 Ücretsiz Gemini Modeli Desteği & Gerçek API Kimliği Doğrulaması:
  ListModels sorgusuyla ve otomatik ad düzeltmeyle (`normalize_gemini_model_id`)
  404 NOT_FOUND hatalarını tamamen engeller.
- Akıllı Otomatik Model Geçişi (Multi-Model Fallback):
  Seçili modelde 429 (kota/hız limiti), 503 (yoğunluk), 404 veya günlük kota
  dolumu yaşandığında işlemi durdurmaz; anında kotası müsait sıradaki modele
  (öncelikle 500/gün kotalı 3.5 Flash Lite ve 3.1 Flash Lite modellerine)
  otomatik geçerek dublajı kesintisiz sürdürür.
- Canlı Günlük Kota Takibi (`./_veri/gemini_kota.json`):
  Her başarılı isteği ilgili modelin günlük sayacına işler; her gün 11:00 TRT
  saatinde tüm sayaçları otomatik sıfırlar.
- Toplu (Batch JSON) Çeviri (20–25 satır/istek) + yerel SQLite önbelleği +
  büyük/küçük harf duyarlı sözlük + 4 RPM güvenli hız limiti.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from core.gemini_models import (
    GEMINI_FREE_MODELS,
    GeminiQuotaManager,
    normalize_gemini_model_id,
    resolve_model_spec,
)
from core.glossary import GlossaryEntry, GlossaryManager
from core.translation_cache import TranslationCache

DEFAULT_GEMINI_RPM: int = 4
DEFAULT_BATCH_SIZE: int = 22


DUBBING_SCREENWRITER_SYSTEM_INSTRUCTION = """Sen AAA PC oyunları için çalışan baş dublaj senaristi, otomatik dublaj yönetmeni ve seslendirme uyarlayıcısısın (örnek kalite standardı: The Last of Us, God of War, Hitman, Cyberpunk 2077 Türkçe dublajları).

TEMEL İLKELERİN:
1. SÜRE VE HECE UYUMU (ALTIN KURAL):
   - Dublajda ses hızlandırılmadan (1.00x–1.05x doğal tempo) orijinal repliğin süresine rahatça sığan Türkçe replikler yazarsın.
   - Verilen hedef hece sayısına ve saniye bütçesine sadık kalırsın; gereksiz dolgu kelimeleri, uzun yan cümleleri ve kitabi yapıları elersin.
2. SANSÜRSÜZ VE DOĞAL OYUN DUBLAJI TÜRKÇESİ:
   - Kelimesi kelimesine ("çeviri kokan") mekanik ifadeler ASLA kullanmazsın.
   - Orijinal repliğin sertliğini, tehdidini, kara mizahını, alaycılığını veya öfkesini yumuşatmadan, sansürlemeden, karakterin doğasına uygun akıcı konuşma Türkçesiyle uyarlarsın.
3. OTOMATİK DUBLAJ YÖNETMENİ — DÜZ SPİKER OKUMASINI BİTİR (ACTING PROMPT — VOXCPM2 48 kHz):
   - Toplu (JSON) çeviride her satır için orijinal İngilizce cümlenin anlamından, noktalama işaretlerinden ve sesin LUFS/süre bilgisinden 4–8 kelimelik keskin, sinematik ve dramatik İngilizce oyunculuk/duygu/tempo talimatı (`acting_prompt`) üretirsin.
   - YASAKLI KELİMELER: `acting_prompt` içinde sesi düzleştiren ve haber spikeri gibi ruhsuz okutan `calm`, `neutral`, `professional`, `professional tone`, `normal`, `standard`, `polite` kelimelerini KESİNLİKLE KULLANMA!
   - Örnek sinematik `acting_prompt` değerleri:
     * "threatening, sinister dark tone, dramatic pause, intense"
     * "sarcastic, mocking dark humor, expressive"
     * "angry, urgent, strained voice, fast intense delivery"
     * "hushed, ominous whisper, tense dramatic pause"
4. DRAMATİK DURAKSAMA VE VURGU NOKTALAMASI:
   - Türkçe dublaj metninde (`tr`) dramatik duraksama, gerilim ve vurgu gereken yerlerde doğal noktalama işaretleri (`...`, `!`, `?`) kullan ki ses motoru doğru yerde nefes, duraksama ve sert vurgu yapsın.
5. SÖZLÜK VE FONETİK ÖZEL İSİM KURALLARI:
   - Zorunlu sözlük terimleri verilmişse büyük/küçük harf kuralına uyarak birebir belirtilen karşılıkları kullanırsın.
   - Fonetik özel isim kuralı açık olduğunda, sözlük dışında kalan yabancı kişi/yer/şirket isimlerini Türkçe ses motorunun pürüzsüz okuyacağı şekilde Türkçe okunuşuyla yazarsın (örn. "Hamilton Lowe" -> "Hamilton Lov").
6. SAF ÇIKTI FORMATI:
   - Tekli çeviride yalnızca saf Türkçe repliği; toplu (JSON) çeviride yalnızca istenen JSON yapısını döndürürsün.
   - Türkçe dublaj metninin (`tr_text` / `tr`) içine asla parantez içi sahne notu eklemezsin; oyunculuk yönlendirmesini yalnızca `acting_prompt` alanına yazarsın."""


BANNED_FLAT_ACTING_WORDS: tuple[str, ...] = (
    "professional tone",
    "calm",
    "neutral",
    "professional",
    "normal",
    "standard",
    "polite",
)

_BANNED_FLAT_REGEX = re.compile(
    r"\b(?:professional\s+tone|calm|neutral|professional|normal|standard|polite)\b",
    flags=re.IGNORECASE,
)


def infer_fallback_acting_prompt(
    source_text: str,
    duration_sec: float = 3.0,
    lufs: float = -23.0,
) -> str:
    """Önbellekte veya tekli yanıtta acting_prompt boş kaldığında metin, süre ve LUFS'tan dramatik İngilizce oyunculuk talimatı çıkarır."""
    txt = (source_text or "").strip()
    low = txt.lower()
    words = re.findall(r"[A-Za-z']+", txt)
    wps = len(words) / max(0.5, float(duration_sec))

    if wps < 2.1:
        pace = "dramatic pause, intense"
    elif wps > 3.4:
        pace = "fast intense delivery"
    else:
        pace = "sharp dramatic emphasis"

    if any(k in low for k in ("die", "kill", "terminated", "contract", "eliminat", "dead", "target", "blood")):
        return f"threatening, sinister dark tone, {pace}"
    if any(k in low for k in ("sushi", "enjoy", "pleasure", "welcome", "sir", "good evening", "taste")):
        return "sarcastic, mocking dark humor, expressive"
    if "!" in txt or float(lufs) > -19.0:
        return "angry, urgent, strained voice, fast intense delivery"
    if "?" in txt:
        return "suspicious, sharp interrogative tone, tense delivery"
    if "..." in txt or float(lufs) < -26.0:
        return "hushed, ominous whisper, tense dramatic pause"
    return f"gritty, intense dark tone, {pace}"


def clean_acting_prompt(
    raw_prompt: str,
    source_text: str = "",
    duration_sec: float = 3.0,
    lufs: float = -23.0,
) -> str:
    """Oyunculuk/duygu talimatındaki parantezleri ve yasaklı düz spiker kelimelerini (calm, neutral, professional, polite vb.) temizler."""
    val = " ".join((raw_prompt or "").strip().splitlines()).strip()
    while len(val) >= 2 and val.startswith("(") and val.endswith(")"):
        val = val[1:-1].strip()

    if not val:
        return ""

    # Yasaklı düz spiker kelimelerini çıkar
    cleaned = _BANNED_FLAT_REGEX.sub("", val)
    # Virgülle ayrılmış parçaları düzenle ve boş parçaları at
    parts: list[str] = []
    for chunk in cleaned.split(","):
        c = " ".join(chunk.split()).strip(" -_;:")
        if c:
            parts.append(c)
    result = ", ".join(parts).strip()

    # Eğer tüm kelimeler yasaklı kelimelerden oluşuyorsa veya çok kısaldıysa dramatik yedek üret
    alpha_words = re.findall(r"[A-Za-z]+", result)
    if len(alpha_words) < 2:
        return infer_fallback_acting_prompt(source_text, duration_sec=duration_sec, lufs=lufs)
    return result


def parse_retry_delay_seconds(error_text: str) -> Optional[float]:
    """Google Gemini 429 RESOURCE_EXHAUSTED hatasındaki `retryDelay` süresini saniye olarak okur."""
    if not error_text:
        return None

    patterns = (
        r"['\"]?retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s\b",
        r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s\b",
        r"retry_delay\s*\{[^}]*seconds\s*:\s*(\d+(?:\.\d+)?)",
    )
    for pat in patterns:
        m = re.search(pat, error_text, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


def translate_text_free_fallback(
    text: str,
    source_lang: str = "en",
    target_lang: str = "tr",
    glossary_matches: Optional[Sequence[GlossaryEntry]] = None,
) -> str:
    """Tüm Gemini modelleri 503/kota hatası verdiğinde devreye giren ücretsiz, API anahtarsız yedek çevirmen.

    Öncelikle doğrudan Google Translate HTTPS JSON uç noktasını (`translate.googleapis.com/translate_a/single`),
    yedek olarak `deep-translator` paketini kullanır ve büyük/küçük harf duyarlı sözlük terimlerini korur.
    """
    src = (text or "").strip()
    if not src:
        return ""

    # Büyük/küçük harf duyarlı sözlük terimlerini yer tutucu veya doğrudan hedef karşılıkla koru
    working_text = src
    placeholder_map: dict[str, str] = {}
    if glossary_matches:
        sorted_matches = sorted(glossary_matches, key=lambda e: len(e.source), reverse=True)
        for idx, entry in enumerate(sorted_matches):
            if entry.source and entry.target and entry.source in working_text:
                token = f"__GLOSS{idx}__"
                placeholder_map[token] = entry.target
                working_text = working_text.replace(entry.source, token)

    sl = (source_lang or "en").strip().lower() or "auto"
    tl = (target_lang or "tr").strip().lower() or "tr"
    translated = ""

    # 1. Doğrudan Google Translate HTTPS JSON uç noktası (API anahtarı gerektirmez)
    try:
        params = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": sl,
                "tl": tl,
                "dt": "t",
                "q": working_text,
            }
        )
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            segments = [
                str(seg[0])
                for seg in payload[0]
                if isinstance(seg, list) and seg and seg[0] is not None
            ]
            translated = "".join(segments).strip()
    except Exception:
        translated = ""

    # 2. Yedek: deep-translator (GoogleTranslator / MyMemoryTranslator)
    if not translated:
        try:
            from deep_translator import GoogleTranslator  # type: ignore

            translated = (
                GoogleTranslator(source=sl, target=tl).translate(working_text) or ""
            ).strip()
        except Exception:
            translated = ""

    if not translated:
        try:
            from deep_translator import MyMemoryTranslator  # type: ignore

            translated = (
                MyMemoryTranslator(source=sl if sl != "auto" else "en-US", target=tl).translate(
                    working_text
                )
                or ""
            ).strip()
        except Exception:
            translated = ""

    if not translated:
        translated = working_text

    # Sözlük yer tutucularını gerçek Türkçe karşılıklarıyla geri koy
    for token, target_val in placeholder_map.items():
        translated = re.sub(re.escape(token), target_val, translated, flags=re.IGNORECASE)
        bare_num = token.strip("_")
        translated = re.sub(rf"_*{re.escape(bare_num)}_*", target_val, translated, flags=re.IGNORECASE)

    return " ".join(translated.splitlines()).strip()


class RateLimiter:
    """Dakika başına istek (RPM) hız limiti koruyucusu (Varsayılan: 4 RPM, model bazlı kovalarla)."""

    def __init__(self, max_requests_per_minute: int = DEFAULT_GEMINI_RPM) -> None:
        self.rpm = max(1, int(max_requests_per_minute))
        self._timestamps: deque[float] = deque()
        self._by_model: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _bucket_for(self, model_id: str) -> deque[float]:
        key = (model_id or "").strip().lower()
        if not key:
            return self._timestamps
        if key not in self._by_model:
            self._by_model[key] = deque()
        return self._by_model[key]

    def release_last(self, model_id: str = "") -> None:
        """503/404 gibi işlenmeyen sunucu hatalarında son zaman damgasını serbest bırakır."""
        with self._lock:
            bucket = self._bucket_for(model_id)
            if bucket:
                bucket.pop()
            if model_id and self._timestamps:
                self._timestamps.pop()

    def acquire(
        self,
        cancel_event: Optional[threading.Event] = None,
        model_id: str = "",
    ) -> None:
        while True:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Çeviri hız limiti beklemesi kullanıcı tarafından iptal edildi.")
            wait_sec = 0.0
            with self._lock:
                now = time.monotonic()
                bucket = self._bucket_for(model_id)
                while bucket and (now - bucket[0]) >= 60.0:
                    bucket.popleft()
                while self._timestamps and (now - self._timestamps[0]) >= 60.0:
                    self._timestamps.popleft()

                if len(bucket) < self.rpm:
                    bucket.append(now)
                    if model_id:
                        self._timestamps.append(now)
                    return
                oldest = bucket[0]
                wait_sec = max(0.05, 60.0 - (now - oldest) + 0.05)

            slept = 0.0
            while slept < wait_sec:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Çeviri işlemi iptal edildi.")
                step = min(0.2, wait_sec - slept)
                time.sleep(step)
                slept += step


@dataclass
class TranslationResult:
    """Tek satır çeviri ve otomatik oyunculuk/duygu yönlendirmesi sonucu."""

    translated_text: str
    from_cache: bool
    matched_glossary_count: int
    acting_prompt: str = ""


@dataclass
class BatchLineInput:
    """Toplu (batch) çeviri için tek bir satır girdisi."""

    item_id: str
    source_text: str
    duration_sec: float
    source_lang: str = "en"
    target_lang: str = "tr"
    lufs: float = -23.0


class GeminiTranslator:
    """Oyun dublajı için çoklu model yedeklemeli (fallback) ve toplu (batch JSON) Gemini çevirmeni."""

    SYSTEM_INSTRUCTION = DUBBING_SCREENWRITER_SYSTEM_INSTRUCTION

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        rpm: int = DEFAULT_GEMINI_RPM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache: Optional[TranslationCache] = None,
        glossary: Optional[GlossaryManager] = None,
        quota_manager: Optional[GeminiQuotaManager] = None,
        max_retries: int = 10,
        enable_model_fallback: bool = True,
        phonetic_foreign_names: bool = True,
        enable_free_fallback: bool = True,
    ) -> None:
        self.api_key = api_key.strip()
        self.model_name = normalize_gemini_model_id(model_name)
        self.batch_size = max(1, min(30, int(batch_size)))
        self.cache = cache or TranslationCache()
        self.glossary = glossary or GlossaryManager()
        self.quota_manager = quota_manager or GeminiQuotaManager()
        self.quota_manager.set_active_model(self.model_name)
        self.rate_limiter = RateLimiter(rpm)
        self.max_retries = max(1, int(max_retries))
        self.enable_model_fallback = bool(enable_model_fallback)
        self.phonetic_foreign_names = bool(phonetic_foreign_names)
        self.enable_free_fallback = bool(enable_free_fallback)
        self._free_translate_fn = translate_text_free_fallback
        self._client = None
        self._models_verified_once = False

    def _cache_glossary_key(self, glossary_hash: str) -> str:
        """Önbellek anahtarına sözlük özeti ve fonetik özel isim ayarını kodlar."""
        if self.phonetic_foreign_names:
            return f"v2:{glossary_hash}"
        return f"v2:pf=0:{glossary_hash}"

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError(
                "Google Gemini API anahtarı boş! Lütfen arayüzden geçerli bir Gemini API anahtarı girin."
            )
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=25000, retry_options=None),
        )
        return self._client

    def verify_models_online(
        self,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> dict[str, bool]:
        """Gemini API `models.list()` sorgusuyla 8 modelin gerçek API kimliklerini doğrular."""
        try:
            client = self._get_client()
            if not hasattr(client.models, "list"):
                return {}
            live_names = [
                str(getattr(m, "name", ""))
                for m in client.models.list()
                if getattr(m, "name", None)
            ]
            if not live_names:
                return {}
            verified = self.quota_manager.verify_models_with_api(live_names)
            self._models_verified_once = True
            if log_cb:
                ok_cnt = sum(1 for v in verified.values() if v)
                log_cb(
                    f"Gemini ListModels doğrulaması tamamlandı: {ok_cnt}/{len(verified)} ücretsiz model aktif."
                )
            return verified
        except Exception:
            return {}

    @staticmethod
    def estimate_source_syllables(text: str) -> int:
        """Orijinal metnin yaklaşık hece sayısını kestirir."""
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿçğıöşüÇĞİÖŞÜ']+", text)
        total = 0
        for w in words:
            groups = re.findall(r"[aeiouyAEIOUYaeıioöuüAEIİOÖUÜ]+", w)
            total += max(1, len(groups))
        return max(1, total)

    @classmethod
    def compute_syllable_budget(cls, source_text: str, duration_sec: float) -> tuple[float, int, int, int]:
        """Doğal (1.00x–1.05x) konuşma temposu için hedef ve maksimum hece bütçesini hesaplar."""
        safe_dur = max(0.35, float(duration_sec))
        src_syllables = cls.estimate_source_syllables(source_text)
        dur_target_syl = max(2, int(round(safe_dur * 4.4)))
        target_syllables = max(2, min(dur_target_syl, int(round(src_syllables * 1.15)) + 2))
        max_syllables = max(target_syllables + 2, int(round(safe_dur * 5.0)))
        return safe_dur, src_syllables, target_syllables, max_syllables

    def build_prompt(
        self,
        source_text: str,
        duration_sec: float,
        source_lang: str = "en",
        target_lang: str = "tr",
    ) -> tuple[str, int, str]:
        """Tek satır için süre, hece bütçesi, fonetik özel isim ve sözlük kısıtlarını içeren dublaj istemini oluşturur."""
        matched_terms = self.glossary.find_matches(source_text)
        glossary_hash = self.glossary.signature_hash(matched_terms)
        glossary_block = self.glossary.format_prompt_block(matched_terms)

        safe_dur, src_syllables, target_syllables, max_syllables = self.compute_syllable_budget(
            source_text, duration_sec
        )

        prompt_parts = [
            f"KAYNAK DİL: {source_lang} -> HEDEF DİL: {target_lang} (Oyun Karakteri Dublajı)",
            "",
            "ZAMANLAMA VE HECE BÜTÇESİ:",
            f"- Orijinal ses süresi: {safe_dur:.2f} saniye",
            f"- Orijinal metin yaklaşık uzunluğu: ~{src_syllables} hece",
            f"- İdeal Türkçe dublaj uzunluğu: yaklaşık {target_syllables} hece (üst sınır: {max_syllables} hece)",
            "- Ses hızlandırmaya gerek kalmadan doğal konuşma temposunda bu süreye tam oturmalıdır.",
            "- Orijinal repliğin sertliğini, tehdidini veya kara mizahını yumuşatmadan/sansürlemeden doğal konuşma Türkçesine uyarla.",
        ]

        if self.phonetic_foreign_names:
            prompt_parts.append(
                "- YABANCI İSİM OKUNUŞU (TTS): Sözlükte özel karşılığı verilmemiş yabancı özel isimleri "
                "(kişi, soyadı, yer, şirket adları) Türkçe ses motorunun pürüzsüz okuması için Türkçe okunuşuyla "
                "(fonetik) yaz (Örn: 'Hamilton Lowe' -> 'Hamilton Lov', 'Diana Burnwood' -> 'Dayana Börnvud')."
            )

        if glossary_block:
            prompt_parts.append("")
            prompt_parts.append(glossary_block)

        prompt_parts.extend(
            [
                "",
                f"ORİJİNAL REPLİK ({safe_dur:.2f} sn):",
                source_text,
                "",
                "SADECE TÜRKÇE DUBLAJ REPLİĞİ:",
            ]
        )

        return "\n".join(prompt_parts), len(matched_terms), glossary_hash

    def build_batch_prompt(
        self,
        batch_items: Sequence[BatchLineInput],
    ) -> tuple[str, dict[str, list[GlossaryEntry]]]:
        """20-25 satırlık bir grubu tek seferde çevirmek ve oyunculuk talimatı (acting_prompt) üretmek için JSON formatlı istem üretir."""
        all_matched: dict[str, GlossaryEntry] = {}
        per_item_matches: dict[str, list[GlossaryEntry]] = {}
        payload_rows: list[dict] = []

        for item in batch_items:
            matches = self.glossary.find_matches(item.source_text)
            per_item_matches[item.item_id] = matches
            for m in matches:
                all_matched[m.source] = m

            safe_dur, src_syl, target_syl, max_syl = self.compute_syllable_budget(
                item.source_text, item.duration_sec
            )
            payload_rows.append(
                {
                    "id": item.item_id,
                    "duration_sec": round(safe_dur, 2),
                    "lufs": round(float(item.lufs), 1),
                    "source_syllables": src_syl,
                    "target_syllables": target_syl,
                    "max_syllables": max_syl,
                    "source": item.source_text,
                }
            )

        src_lang = batch_items[0].source_lang if batch_items else "en"
        tgt_lang = batch_items[0].target_lang if batch_items else "tr"
        glossary_block = self.glossary.format_prompt_block(list(all_matched.values()))

        prompt_parts = [
            f"GÖREV: Aşağıdaki {len(batch_items)} adet ardışık oyun repliğini ({src_lang}) sahne bağlamını, "
            f"karakter tonunu ve her satırın kendi süre/hece bütçesini koruyarak sansürsüz, doğal konuşma Türkçesine ({tgt_lang}) çevir "
            "ve her satır için VoxCPM2 48 kHz ses klonlama motoruna verilecek keskin, sinematik İngilizce oyunculuk/duygu/tempo talimatı (acting_prompt) üret.",
            "",
            "TOPLU ÇEVİRİ VE OTOMATİK DUBLAJ YÖNETMENİ KURALLARI (DÜZ SPİKER OKUMASINI BİTİR):",
            "- Her satırın 'duration_sec', 'target_syllables' ve 'max_syllables' değerlerine titizlikle uy.",
            "- Replikler hızlandırma yapılmadan (1.00x–1.05x doğal konuşma hızında) belirtilen saniyeye sığmalıdır.",
            "- Türkçe dublaj metninde ('tr') dramatik duraksama ve vurgu gereken yerlerde doğal noktalama ('...', '!', '?') kullan.",
            "- Orijinal metnin anlamından, noktalama işaretlerinden, 'duration_sec' ve 'lufs' ses yüksekliği bilgisinden "
            "her satır için 4–8 kelimelik keskin, sinematik ve dramatik İngilizce bir oyunculuk talimatı ('acting_prompt') çıkar.",
            "- YASAKLI KELİMELER: 'acting_prompt' içinde sesi düzleştiren ve haber spikeri gibi okutan "
            "'calm', 'neutral', 'professional', 'professional tone', 'normal', 'standard', 'polite' kelimelerini KESİNLİKLE KULLANMA!",
            "  * Örnek 1: \"threatening, sinister dark tone, dramatic pause, intense\"",
            "  * Örnek 2: \"sarcastic, mocking dark humor, expressive\"",
            "  * Örnek 3: \"angry, urgent, strained voice, fast intense delivery\"",
            "  * Örnek 4: \"hushed, ominous whisper, tense dramatic pause\"",
            "- Çıktıyı SADECE geçerli bir JSON dizisi (array) olarak döndür:",
            '  [{"id": "<satir_id>", "tr": "<Turkce dublaj metni>", "acting_prompt": "<4-8 word dramatic English acting prompt>"}, ...]',
            "- Hiçbir satırı atlama; girdi listesindeki tüm 'id' değerlerini birebir aynı sırada döndür.",
        ]

        if self.phonetic_foreign_names:
            prompt_parts.append(
                "- YABANCI İSİM OKUNUŞU (TTS): Sözlükte özel karşılığı verilmemiş yabancı özel isimleri "
                "(kişi, soyadı, yer, şirket adları) Türkçe ses motorlarının pürüzsüz okuyacağı şekilde Türkçe okunuşuyla "
                "(fonetik) yaz (Örn: 'Hamilton Lowe' -> 'Hamilton Lov', 'Diana Burnwood' -> 'Dayana Börnvud')."
            )

        if glossary_block:
            prompt_parts.append("")
            prompt_parts.append(glossary_block)

        prompt_parts.extend(
            [
                "",
                "GİRDİ REPLİKLERİ (JSON):",
                json.dumps(payload_rows, ensure_ascii=False, indent=2),
                "",
                "ÇIKTI JSON DİZİSİ:",
            ]
        )
        return "\n".join(prompt_parts), per_item_matches

    @staticmethod
    def _clean_response_text(raw_text: str) -> str:
        text = (raw_text or "").strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = " ".join(ln.strip() for ln in lines if ln.strip())
        text = " ".join(text.splitlines()).strip()
        if len(text) >= 2 and (
            (text[0] == '"' and text[-1] == '"')
            or (text[0] == "'" and text[-1] == "'")
            or (text[0] == "«" and text[-1] == "»")
        ):
            text = text[1:-1].strip()
        return text

    @classmethod
    def parse_batch_json_entries(
        cls,
        raw_text: str,
        expected_ids: Sequence[str],
    ) -> dict[str, tuple[str, str]]:
        """Gemini'nin döndürdüğü JSON yanıtını ayrıştırıp `{item_id: (turkce_metin, acting_prompt)}` sözlüğüne dönüştürür."""
        cleaned = (raw_text or "").strip()
        if cleaned.startswith("```"):
            lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            m_arr = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
            if m_arr:
                try:
                    parsed = json.loads(m_arr.group(0))
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                m_obj = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if m_obj:
                    try:
                        parsed = json.loads(m_obj.group(0))
                    except json.JSONDecodeError:
                        parsed = None

        if parsed is None:
            raise ValueError(f"Gemini toplu çeviri yanıtı geçerli JSON değil: {cleaned[:200]}")

        out: dict[str, tuple[str, str]] = {}

        if isinstance(parsed, dict):
            list_candidate = (
                parsed.get("translations")
                or parsed.get("items")
                or parsed.get("lines")
                or parsed.get("results")
                or parsed.get("data")
            )
            if isinstance(list_candidate, list):
                parsed = list_candidate
            else:
                for k, v in parsed.items():
                    if isinstance(v, str) and v.strip():
                        out[str(k)] = (cls._clean_response_text(v), "")
                    elif isinstance(v, dict):
                        txt = (
                            v.get("tr")
                            or v.get("tr_text")
                            or v.get("text")
                            or v.get("translation")
                            or ""
                        )
                        act = (
                            v.get("acting_prompt")
                            or v.get("acting")
                            or v.get("emotion")
                            or v.get("style")
                            or ""
                        )
                        if isinstance(txt, str) and txt.strip():
                            out[str(k)] = (
                                cls._clean_response_text(txt),
                                clean_acting_prompt(str(act)) if act else "",
                            )
                return out

        if isinstance(parsed, list):
            for idx, entry in enumerate(parsed):
                if isinstance(entry, dict):
                    raw_id = entry.get("id")
                    item_id = (
                        str(raw_id)
                        if raw_id is not None
                        else (expected_ids[idx] if idx < len(expected_ids) else str(idx))
                    )
                    txt = (
                        entry.get("tr")
                        or entry.get("tr_text")
                        or entry.get("translated_text")
                        or entry.get("translation")
                        or entry.get("text")
                        or entry.get("turkish")
                        or ""
                    )
                    act = (
                        entry.get("acting_prompt")
                        or entry.get("acting")
                        or entry.get("emotion")
                        or entry.get("style")
                        or entry.get("voice_prompt")
                        or ""
                    )
                    if isinstance(txt, str) and txt.strip():
                        out[item_id] = (
                            cls._clean_response_text(txt),
                            clean_acting_prompt(str(act)) if act else "",
                        )
                elif isinstance(entry, str) and idx < len(expected_ids) and entry.strip():
                    out[expected_ids[idx]] = (cls._clean_response_text(entry), "")

        return out

    @classmethod
    def parse_batch_json_response(
        cls,
        raw_text: str,
        expected_ids: Sequence[str],
    ) -> dict[str, str]:
        """Gemini'nin döndürdüğü JSON yanıtını ayrıştırıp `{item_id: turkce_metin}` sözlüğüne dönüştürür."""
        entries = cls.parse_batch_json_entries(raw_text, expected_ids)
        return {k: v[0] for k, v in entries.items()}

    @staticmethod
    def _sleep_cancellable(
        wait_sec: float,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        waited = 0.0
        while waited < wait_sec:
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Çeviri iptal edildi.")
            step = min(0.25, wait_sec - waited)
            time.sleep(step)
            waited += step

    def _switch_to_fallback_model(
        self,
        failed_model_id: str,
        tried_in_round: set[str],
        reason_label: str,
        mark_exhausted: bool = False,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Hata veya kota dolumu yaşayan modelden sıradaki müsait modele otomatik geçiş yapar.

        Returns:
            Yeni bir müsait modele geçildiyse True, tüm modeller denendiyse False.
        """
        if mark_exhausted:
            self.quota_manager.mark_model_exhausted(failed_model_id, reason=reason_label)

        if not self.enable_model_fallback:
            return False

        tried_in_round.add(failed_model_id.lower())
        next_spec = self.quota_manager.select_available_model(
            preferred_model=self.model_name,
            exclude_api_ids=tried_in_round,
        )
        if next_spec is None or next_spec.api_id.lower() == failed_model_id.lower():
            return False

        old_spec = resolve_model_spec(failed_model_id)
        self.model_name = next_spec.api_id
        self.quota_manager.set_active_model(next_spec.api_id)
        if log_cb:
            log_cb(
                f"🔄 Otomatik Model Geçişi ({reason_label}): '{old_spec.short_name}' -> "
                f"'{next_spec.short_name}' ({next_spec.api_id}, Limit: {next_spec.daily_limit}/gün)"
            )
        return True

    def _generate_with_retry(
        self,
        prompt: str,
        max_output_tokens: int = 256,
        response_mime_type: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Gemini API çağrısını çoklu model geçişi (fallback), 429 retryDelay ve 503 korumasıyla yürütür."""
        client = self._get_client()
        if not self._models_verified_once:
            self.verify_models_online(log_cb=log_cb)

        from google.genai import types  # type: ignore

        tried_in_round: set[str] = set()
        retries_503_per_model: dict[str, int] = {}
        total_attempts = max(self.max_retries, len(GEMINI_FREE_MODELS) * 2)
        last_err: Optional[Exception] = None

        for attempt in range(total_attempts):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Çeviri iptal edildi.")

            # Seçili modelin günlük kotası dolmuşsa istek göndermeden önce müsait modele geç
            if self.enable_model_fallback:
                avail_spec = self.quota_manager.select_available_model(
                    preferred_model=self.model_name,
                    exclude_api_ids=tried_in_round,
                )
                if avail_spec is not None and avail_spec.api_id != self.model_name:
                    old_spec = resolve_model_spec(self.model_name)
                    self.model_name = avail_spec.api_id
                    self.quota_manager.set_active_model(avail_spec.api_id)
                    if log_cb:
                        log_cb(
                            f"🔄 '{old_spec.short_name}' günlük limiti dolu -> "
                            f"'{avail_spec.short_name}' ({avail_spec.daily_limit}/gün) modeline geçildi."
                        )

            current_model = self.model_name
            self.rate_limiter.acquire(cancel_event=cancel_event, model_id=current_model)

            try:
                cfg_kwargs: dict = {
                    "system_instruction": self.SYSTEM_INSTRUCTION,
                    "temperature": 0.35,
                    "max_output_tokens": max_output_tokens,
                }
                if response_mime_type:
                    cfg_kwargs["response_mime_type"] = response_mime_type

                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                raw_text = (getattr(response, "text", "") or "").strip()
                if not raw_text:
                    raise RuntimeError("Gemini API boş yanıt döndürdü.")

                # Başarılı kullanımı canlı kota sayacına işle
                self.quota_manager.record_usage(current_model, count=1)
                return raw_text

            except InterruptedError:
                raise
            except Exception as exc:
                last_err = exc
                err_raw = str(exc)
                err_str = err_raw.lower()
                self.rate_limiter.release_last(current_model)

                # 1) 404 NOT_FOUND -> ListModels ile doğrula ve anında sıradaki çalışan modele geç
                if ("404" in err_str or "not_found" in err_str) and attempt < total_attempts - 1:
                    self.verify_models_online(log_cb=log_cb)
                    if self._switch_to_fallback_model(
                        failed_model_id=current_model,
                        tried_in_round=tried_in_round,
                        reason_label="404 Model Bulunamadı",
                        mark_exhausted=True,
                        log_cb=log_cb,
                    ):
                        continue

                # 2) 429 RESOURCE_EXHAUSTED -> Günlük kotası dolan modeli işaretle;
                #    başka müsait model varsa beklemeden ona geç, yoksa retryDelay kadar bekle!
                is_429 = any(
                    tok in err_str for tok in ("429", "resource_exhausted", "quota", "rate limit")
                )
                if is_429 and attempt < total_attempts - 1:
                    retry_delay = parse_retry_delay_seconds(err_raw)
                    is_daily_exhausted = (
                        any(k in err_str for k in ("perday", "per_day", "rpd", "daily", "day"))
                        or (retry_delay is None)
                    )
                    if self._switch_to_fallback_model(
                        failed_model_id=current_model,
                        tried_in_round=tried_in_round,
                        reason_label="429 Kota/Hız Sınırı",
                        mark_exhausted=is_daily_exhausted,
                        log_cb=log_cb,
                    ):
                        continue

                    # Tüm modellerin günlük kotası dolduysa ve retryDelay yoksa boşuna bekleme yapma
                    if is_daily_exhausted and self.enable_model_fallback:
                        break

                    # Fallback kapalıysa veya dakikalık (RPM) retryDelay varsa Google'ın retryDelay süresini bekle
                    tried_in_round.clear()
                    if retry_delay is not None:
                        wait_sec = retry_delay + 1.5
                        if log_cb:
                            log_cb(
                                f"Gemini API kota sınırı (429 RESOURCE_EXHAUSTED): Google tarafından bildirilen "
                                f"retryDelay={retry_delay:.1f} sn (+1.5 sn güvenlik payı = {wait_sec:.1f} sn) "
                                f"bekleniyor ({attempt + 1}/{total_attempts})..."
                            )
                    else:
                        wait_sec = min(65.0, 25.0 * (attempt + 1))
                        if log_cb:
                            log_cb(
                                f"Gemini API kota sınırı (429): {wait_sec:.1f} sn beklenip "
                                f"yeniden denenecek ({attempt + 1}/{total_attempts})..."
                            )
                    self._sleep_cancellable(wait_sec, cancel_event)
                    continue

                # 3) 503 UNAVAILABLE / 500 geçici yoğunluk ->
                #    Aynı modelde 10 kez 15'er saniye BEKLEME!
                #    503 hatasını günlük kota tüketimi olarak SAYMA (mark_exhausted=False)!
                #    Model başına en fazla 1 kısa yeniden deneme (3 saniye) veya anında sıradaki müsait modele geç!
                is_503 = any(
                    tok in err_str
                    for tok in (
                        "503",
                        "unavailable",
                        "overloaded",
                        "high demand",
                        "500",
                        "502",
                        "504",
                        "timeout",
                        "temporarily",
                    )
                )
                if is_503 and attempt < total_attempts - 1:
                    # Önce kotası kalan sıradaki müsait modele (3.8 Flash dahil tüm 8 model) anında geç
                    if self._switch_to_fallback_model(
                        failed_model_id=current_model,
                        tried_in_round=tried_in_round,
                        reason_label="503 Sunucu Yoğunluğu",
                        mark_exhausted=False,
                        log_cb=log_cb,
                    ):
                        continue

                    # Sıradaki başka model kalmadıysa (veya model geçişi kapalıysa)
                    # bu modelde en fazla 1 kısa yeniden deneme (3.0 saniye) yap;
                    # 2. kez 503 gelirse dakikalarca beklemeden döngüyü sonlandır!
                    cur_503_cnt = retries_503_per_model.get(current_model, 0)
                    if cur_503_cnt < 1:
                        retries_503_per_model[current_model] = cur_503_cnt + 1
                        wait_sec = 3.0
                        if log_cb:
                            log_cb(
                                f"Gemini API geçici sunucu yoğunluğu (503 UNAVAILABLE - {current_model}): "
                                f"{wait_sec:.1f} sn beklenip 1 kez yeniden deneniyor..."
                            )
                        self._sleep_cancellable(wait_sec, cancel_event)
                        continue

                    # Tüm müsait modeller denendi ve 503 döndü -> dakikalarca beklemeden hemen çık
                    break

                raise

        raise RuntimeError(f"Gemini API çeviri başarısız (503/Kota): {last_err}")

    def _translate_single_via_free_fallback(
        self,
        source_text: str,
        duration_sec: float,
        source_lang: str,
        target_lang: str,
        lufs: float = -23.0,
    ) -> tuple[str, str, int]:
        """Ücretsiz yedek çevirmenle tek bir satırı çevirir ve yerel dramatik acting_prompt üretir."""
        matched_terms = self.glossary.find_matches(source_text)
        tr_text = self._free_translate_fn(
            source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_matches=matched_terms,
        ).strip()
        if not tr_text:
            tr_text = source_text.strip()
        act_prompt = clean_acting_prompt(
            infer_fallback_acting_prompt(source_text, duration_sec=duration_sec, lufs=lufs),
            source_text=source_text,
            duration_sec=duration_sec,
            lufs=lufs,
        )
        return tr_text, act_prompt, len(matched_terms)

    def translate_line(
        self,
        source_text: str,
        duration_sec: float,
        source_lang: str = "en",
        target_lang: str = "tr",
        use_cache: bool = True,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        lufs: float = -23.0,
    ) -> TranslationResult:
        """Tek bir oyun repliğini sinematik süre ve hece kısıtıyla Türkçeye çevirir."""
        cleaned_src = source_text.strip()
        if not cleaned_src:
            return TranslationResult(
                translated_text="",
                from_cache=True,
                matched_glossary_count=0,
                acting_prompt="",
            )

        prompt, matched_count, glossary_hash = self.build_prompt(
            cleaned_src,
            duration_sec=duration_sec,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        cache_glossary_key = self._cache_glossary_key(glossary_hash)

        if use_cache:
            cached_entry = self.cache.get_entry(
                source_text=cleaned_src,
                source_lang=source_lang,
                target_lang=target_lang,
                duration_sec=duration_sec,
                glossary_hash=cache_glossary_key,
            )
            if cached_entry and cached_entry[0]:
                c_tr, c_act = cached_entry
                c_act = clean_acting_prompt(c_act, cleaned_src, duration_sec, lufs)
                if not c_act:
                    c_act = infer_fallback_acting_prompt(cleaned_src, duration_sec, lufs)
                return TranslationResult(
                    translated_text=c_tr,
                    from_cache=True,
                    matched_glossary_count=matched_count,
                    acting_prompt=c_act,
                )

        used_model_label = self.model_name
        try:
            raw_out = self._generate_with_retry(
                prompt=prompt,
                max_output_tokens=256,
                response_mime_type=None,
                cancel_event=cancel_event,
                log_cb=log_cb,
            )
            translated = self._clean_response_text(raw_out)
            if not translated:
                raise RuntimeError("Gemini API boş çeviri döndürdü.")
            acting_prompt = infer_fallback_acting_prompt(cleaned_src, duration_sec, lufs)
        except InterruptedError:
            raise
        except Exception as exc:
            if not self.enable_free_fallback:
                raise
            if log_cb:
                log_cb(
                    f"⚠️ Gemini sunucuları yoğun (503) — ücretsiz yedek çevirmenle devam edildi ({exc})."
                )
            translated, acting_prompt, matched_count = self._translate_single_via_free_fallback(
                cleaned_src,
                duration_sec=duration_sec,
                source_lang=source_lang,
                target_lang=target_lang,
                lufs=lufs,
            )
            used_model_label = "free-fallback-translator"

        if use_cache:
            self.cache.put(
                source_text=cleaned_src,
                source_lang=source_lang,
                target_lang=target_lang,
                duration_sec=duration_sec,
                glossary_hash=cache_glossary_key,
                model_name=used_model_label,
                translated_text=translated,
                acting_prompt=acting_prompt,
            )
        return TranslationResult(
            translated_text=translated,
            from_cache=False,
            matched_glossary_count=matched_count,
            acting_prompt=acting_prompt,
        )

    def translate_batch(
        self,
        items: Sequence[BatchLineInput],
        use_cache: bool = True,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, TranslationResult]:
        """Replikleri 20-25 satırlık JSON paketleri halinde (Türkçe dublaj + acting_prompt) toplu çevirir."""
        results: dict[str, TranslationResult] = {}
        if not items:
            return results

        if type(self).translate_line is not GeminiTranslator.translate_line:
            for idx, it in enumerate(items, start=1):
                res = self.translate_line(
                    source_text=it.source_text,
                    duration_sec=it.duration_sec,
                    source_lang=it.source_lang,
                    target_lang=it.target_lang,
                    use_cache=use_cache,
                    cancel_event=cancel_event,
                    log_cb=log_cb,
                )
                if not res.acting_prompt and res.translated_text:
                    res.acting_prompt = infer_fallback_acting_prompt(
                        it.source_text, it.duration_sec, it.lufs
                    )
                results[it.item_id] = res
                if progress_cb:
                    progress_cb(idx, len(items))
            return results

        pending: list[tuple[BatchLineInput, int, str]] = []
        total_items = len(items)
        done_count = 0

        for it in items:
            cleaned_src = it.source_text.strip()
            if not cleaned_src:
                results[it.item_id] = TranslationResult("", True, 0, "")
                done_count += 1
                if progress_cb:
                    progress_cb(done_count, total_items)
                continue

            matched = self.glossary.find_matches(cleaned_src)
            g_hash = self._cache_glossary_key(self.glossary.signature_hash(matched))
            if use_cache:
                cached_entry = self.cache.get_entry(
                    source_text=cleaned_src,
                    source_lang=it.source_lang,
                    target_lang=it.target_lang,
                    duration_sec=it.duration_sec,
                    glossary_hash=g_hash,
                )
                if cached_entry and cached_entry[0]:
                    c_tr, c_act = cached_entry
                    c_act = clean_acting_prompt(c_act, cleaned_src, it.duration_sec, it.lufs)
                    if not c_act:
                        c_act = infer_fallback_acting_prompt(cleaned_src, it.duration_sec, it.lufs)
                    results[it.item_id] = TranslationResult(
                        translated_text=c_tr,
                        from_cache=True,
                        matched_glossary_count=len(matched),
                        acting_prompt=c_act,
                    )
                    done_count += 1
                    if progress_cb:
                        progress_cb(done_count, total_items)
                    continue

            pending.append(
                (
                    BatchLineInput(
                        item_id=it.item_id,
                        source_text=cleaned_src,
                        duration_sec=it.duration_sec,
                        source_lang=it.source_lang,
                        target_lang=it.target_lang,
                        lufs=it.lufs,
                    ),
                    len(matched),
                    g_hash,
                )
            )

        if not pending:
            return results

        num_batches = (len(pending) + self.batch_size - 1) // self.batch_size
        for b_idx in range(num_batches):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Toplu çeviri iptal edildi.")

            chunk_tuples = pending[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
            chunk_inputs = [t[0] for t in chunk_tuples]
            expected_ids = [it.item_id for it in chunk_inputs]

            cur_spec = resolve_model_spec(self.model_name)
            if log_cb:
                log_cb(
                    f"Gemini toplu çeviri & oyunculuk yönetmeni paketi {b_idx + 1}/{num_batches} gönderiliyor "
                    f"({len(chunk_inputs)} replik tek istekte | Model: {cur_spec.short_name})..."
                )

            batch_prompt, _ = self.build_batch_prompt(chunk_inputs)
            parsed_entries: dict[str, tuple[str, str]] = {}
            used_free_fallback_for_batch = False

            try:
                raw_json = self._generate_with_retry(
                    prompt=batch_prompt,
                    max_output_tokens=min(8192, max(512, len(chunk_inputs) * 240)),
                    response_mime_type="application/json",
                    cancel_event=cancel_event,
                    log_cb=log_cb,
                )
                try:
                    parsed_entries = self.parse_batch_json_entries(raw_json, expected_ids)
                except Exception as parse_exc:
                    if log_cb:
                        log_cb(
                            f"Toplu JSON ayrıştırma uyarısı ({parse_exc}); eksik satırlar tekli tamamlanacak."
                        )
            except InterruptedError:
                raise
            except Exception as batch_exc:
                if not self.enable_free_fallback:
                    raise
                used_free_fallback_for_batch = True
                if log_cb:
                    log_cb(
                        f"⚠️ Gemini sunucuları yoğun (503) — ücretsiz yedek çevirmenle devam edildi "
                        f"({len(chunk_inputs)} replik | Neden: {batch_exc})."
                    )

            for it, matched_cnt, g_hash in chunk_tuples:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Toplu çeviri iptal edildi.")

                if used_free_fallback_for_batch:
                    tr_text, act_text, fb_matched = self._translate_single_via_free_fallback(
                        it.source_text,
                        duration_sec=it.duration_sec,
                        source_lang=it.source_lang,
                        target_lang=it.target_lang,
                        lufs=it.lufs,
                    )
                    if use_cache:
                        self.cache.put(
                            source_text=it.source_text,
                            source_lang=it.source_lang,
                            target_lang=it.target_lang,
                            duration_sec=it.duration_sec,
                            glossary_hash=g_hash,
                            model_name="free-fallback-translator",
                            translated_text=tr_text,
                            acting_prompt=act_text,
                        )
                    results[it.item_id] = TranslationResult(
                        translated_text=tr_text,
                        from_cache=False,
                        matched_glossary_count=fb_matched,
                        acting_prompt=act_text,
                    )
                    done_count += 1
                    if progress_cb:
                        progress_cb(done_count, total_items)
                    continue

                entry_pair = parsed_entries.get(it.item_id)
                tr_text = (entry_pair[0] if entry_pair else "").strip()
                act_text = (entry_pair[1] if entry_pair else "").strip()
                if not tr_text:
                    single_res = self.translate_line(
                        source_text=it.source_text,
                        duration_sec=it.duration_sec,
                        source_lang=it.source_lang,
                        target_lang=it.target_lang,
                        use_cache=use_cache,
                        cancel_event=cancel_event,
                        log_cb=log_cb,
                        lufs=it.lufs,
                    )
                    results[it.item_id] = single_res
                else:
                    act_text = clean_acting_prompt(
                        act_text, it.source_text, it.duration_sec, it.lufs
                    )
                    if not act_text:
                        act_text = infer_fallback_acting_prompt(
                            it.source_text, it.duration_sec, it.lufs
                        )
                    if use_cache:
                        self.cache.put(
                            source_text=it.source_text,
                            source_lang=it.source_lang,
                            target_lang=it.target_lang,
                            duration_sec=it.duration_sec,
                            glossary_hash=g_hash,
                            model_name=self.model_name,
                            translated_text=tr_text,
                            acting_prompt=act_text,
                        )
                    results[it.item_id] = TranslationResult(
                        translated_text=tr_text,
                        from_cache=False,
                        matched_glossary_count=matched_cnt,
                        acting_prompt=act_text,
                    )

                done_count += 1
                if progress_cb:
                    progress_cb(done_count, total_items)

        return results

    def transcribe_audio_fallback(
        self,
        audio_wav_path: Path,
        source_lang: str = "en",
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Colab/Yerel Whisper bağlı değilken (saf ElevenLabs API modunda) Gemini ile ses deşifresi yapar."""
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Deşifre iptal edildi.")
        self.rate_limiter.acquire(cancel_event=cancel_event)
        client = self._get_client()
        from google.genai import types  # type: ignore

        wav_bytes = audio_wav_path.read_bytes()
        prompt = (
            f"Transcribe the exact spoken dialogue in this audio ({source_lang}). "
            "Output ONLY the verbatim spoken text with proper punctuation. "
            "If there is no human speech, return an empty string."
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
        )
        self.quota_manager.record_usage(self.model_name, count=1)
        return self._clean_response_text(getattr(response, "text", "") or "")
