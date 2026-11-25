"""Sözlük, SQLite Önbellek, 8 Modelli Günlük Kota (11:00 TRT), Otomatik Model Geçişi (Fallback) ve Toplu Çeviri Testleri."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from core.config import AppConfig
from core.gemini_models import (
    GEMINI_FREE_MODELS,
    TOTAL_DAILY_FREE_QUOTA,
    GeminiQuotaManager,
    normalize_gemini_model_id,
)
from core.glossary import GlossaryManager
from core.translation_cache import TranslationCache
from core.translator import (
    BANNED_FLAT_ACTING_WORDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMINI_RPM,
    BatchLineInput,
    GeminiTranslator,
    clean_acting_prompt,
    infer_fallback_acting_prompt,
    parse_retry_delay_seconds,
)


def test_eight_free_models_and_11_00_trt_daily_reset(tmp_path: Path) -> None:
    assert len(GEMINI_FREE_MODELS) == 8
    assert TOTAL_DAILY_FREE_QUOTA == 1120  # 6*20 + 2*500 = 1120

    # Eski 404 veren gemini-3.0-flash / gemini-3-flash otomatik gemini-3.5-flash'a düzeltilmeli
    assert normalize_gemini_model_id("gemini-3.0-flash") == "gemini-3.5-flash"
    assert normalize_gemini_model_id("3 Flash") == "gemini-3.5-flash"
    assert normalize_gemini_model_id("4 Hydrogen") == "gemini-4-hydrogen"
    assert normalize_gemini_model_id("3.5 Flash Lite") == "gemini-3.5-flash-lite"
    assert normalize_gemini_model_id("3.1 Flash Lite") == "gemini-3.1-flash-lite"

    # Saat 10:59 TRT (07:59 UTC) -> Kullanım kaydet
    clock = {"now": datetime(2026, 11, 23, 7, 59, 0, tzinfo=timezone.utc)}
    qpath = tmp_path / "gemini_kota.json"
    qm = GeminiQuotaManager(path=qpath, now_fn=lambda: clock["now"])

    qm.record_usage("gemini-4-hydrogen", count=20)
    qm.record_usage("gemini-3.5-flash-lite", count=10)
    snap_before = qm.get_snapshot()
    assert snap_before.total_used == 30
    assert snap_before.total_remaining == 1090
    h_state = next(m for m in snap_before.models if m.api_id == "gemini-4-hydrogen")
    assert h_state.is_available is False
    assert h_state.exhausted is True

    # Saat 11:00 TRT (08:00 UTC) -> Otomatik olarak tüm sayaçlar 0'a sıfırlanmalı!
    clock["now"] = datetime(2026, 11, 23, 8, 0, 1, tzinfo=timezone.utc)
    snap_after = qm.get_snapshot()
    assert snap_after.total_used == 0
    assert snap_after.total_remaining == 1120
    h_state_after = next(m for m in snap_after.models if m.api_id == "gemini-4-hydrogen")
    assert h_state_after.is_available is True
    assert h_state_after.used == 0


def test_glossary_strict_case_sensitivity(tmp_path: Path) -> None:
    gpath = tmp_path / "sozluk.json"
    gm = GlossaryManager(path=gpath)
    gm.upsert("Shade", "Gölge Varlık", "Özel boss adı")
    gm.upsert("shade", "gölge", "Normal isim")
    gm.upsert("Iron Citadel", "Demir Kale")

    gm2 = GlossaryManager(path=gpath)
    matches_upper = gm2.find_matches("The Shade awaits inside Iron Citadel.")
    sources_upper = {m.source for m in matches_upper}
    assert sources_upper == {"Shade", "Iron Citadel"}
    assert "shade" not in sources_upper

    matches_lower = gm2.find_matches("Stay in the shade, soldier.")
    sources_lower = {m.source for m in matches_lower}
    assert sources_lower == {"shade"}
    assert "Shade" not in sources_lower


def test_sqlite_translation_cache_and_screenwriter_prompt(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    cache = TranslationCache(db_path=db_path)
    gm = GlossaryManager(path=tmp_path / "glossary.json")
    gm.upsert("Kratos", "Kratos")

    translator = GeminiTranslator(
        api_key="DUMMY_KEY",
        model_name="gemini-3.5-flash-lite",
        cache=cache,
        glossary=gm,
        quota_manager=GeminiQuotaManager(path=tmp_path / "kota.json"),
    )

    assert DEFAULT_GEMINI_RPM == 4
    assert translator.rate_limiter.rpm == 4
    assert AppConfig().gemini_rpm == 4
    assert 20 <= DEFAULT_BATCH_SIZE <= 25
    assert "dublaj senaristi" in translator.SYSTEM_INSTRUCTION.lower()

    prompt, matched_cnt, g_hash = translator.build_prompt(
        source_text="Kratos, watch out behind you!",
        duration_sec=2.35,
        source_lang="en",
        target_lang="tr",
    )
    assert "2.35 saniye" in prompt
    assert "hece" in prompt
    assert matched_cnt == 1
    assert '"Kratos" -> "Kratos"' in prompt

    cache.put(
        source_text="Kratos, watch out behind you!",
        source_lang="en",
        target_lang="tr",
        duration_sec=2.35,
        glossary_hash=f"v2:{g_hash}",
        model_name="gemini-3.5-flash-lite",
        translated_text="Kratos, arkana dikkat et!",
    )

    res = translator.translate_line(
        source_text="Kratos, watch out behind you!",
        duration_sec=2.40,
        source_lang="en",
        target_lang="tr",
        use_cache=True,
    )
    assert res.from_cache is True
    assert res.translated_text == "Kratos, arkana dikkat et!"
    assert cache.count() == 1


def test_batch_json_translation_22_lines_single_request_and_cache(tmp_path: Path) -> None:
    cache = TranslationCache(db_path=tmp_path / "batch_cache.db")
    gm = GlossaryManager(path=tmp_path / "batch_glossary.json")
    gm.upsert("Agent 47", "Ajan 47")

    translator = GeminiTranslator(
        api_key="DUMMY_KEY",
        cache=cache,
        glossary=gm,
        quota_manager=GeminiQuotaManager(path=tmp_path / "kota.json"),
        batch_size=22,
    )

    api_calls: list[str] = []

    def fake_generate(
        prompt: str,
        max_output_tokens: int = 256,
        response_mime_type=None,
        cancel_event=None,
        log_cb=None,
    ) -> str:
        api_calls.append(prompt)
        assert response_mime_type == "application/json"
        assert "acting_prompt" in prompt
        assert "Hamilton Lov" in prompt
        out_arr = [
            {
                "id": f"L{i:05d}",
                "tr": f"Ajan 47... Türkçe replik {i}!",
                "acting_prompt": f"threatening, sinister dark tone, dramatic pause, line {i}",
            }
            for i in range(1, 23)
        ]
        return json.dumps(out_arr, ensure_ascii=False)

    translator._generate_with_retry = fake_generate  # type: ignore[method-assign]

    batch_inputs = [
        BatchLineInput(
            item_id=f"L{i:05d}",
            source_text=f"Agent 47 line number {i}.",
            duration_sec=3.0,
            lufs=-22.5,
        )
        for i in range(1, 23)
    ]

    results = translator.translate_batch(batch_inputs, use_cache=True)
    assert len(api_calls) == 1
    assert len(results) == 22
    assert results["L00001"].translated_text == "Ajan 47... Türkçe replik 1!"
    assert results["L00001"].acting_prompt == "threatening, sinister dark tone, dramatic pause, line 1"
    assert results["L00022"].translated_text == "Ajan 47... Türkçe replik 22!"
    assert results["L00022"].acting_prompt == "threatening, sinister dark tone, dramatic pause, line 22"
    assert cache.count() == 22

    results_cached = translator.translate_batch(batch_inputs, use_cache=True)
    assert len(api_calls) == 1
    assert all(r.from_cache is True for r in results_cached.values())
    assert results_cached["L00001"].acting_prompt == "threatening, sinister dark tone, dramatic pause, line 1"


def test_dramatic_dubbing_director_bans_flat_newsreader_words() -> None:
    # Spiker gibi düz/robotik okuma yapan kelimelerin tamamı yasaklı listede olmalı
    for w in ("calm", "neutral", "professional", "professional tone", "normal", "standard", "polite"):
        assert w in BANNED_FLAT_ACTING_WORDS

    # Karışık talimattan yasaklı düz kelimeler tamamen temizlenmeli, dramatik kısım korunmalı
    cleaned = clean_acting_prompt(
        "calm, neutral, professional tone, polite, threatening, sinister dark tone, dramatic pause"
    )
    assert cleaned == "threatening, sinister dark tone, dramatic pause"
    for banned in BANNED_FLAT_ACTING_WORDS:
        assert banned not in cleaned.lower()

    # Sadece yasaklı kelimelerden oluşan bir talimat gelirse otomatik keskin dramatik talimatla değiştirilmeli
    replaced = clean_acting_prompt(
        "calm, neutral, professional tone, normal, standard",
        source_text="Your contract is terminated. Good luck.",
        lufs=-22.5,
    )
    assert replaced
    for banned in BANNED_FLAT_ACTING_WORDS:
        assert banned not in replaced.lower()
    assert any(k in replaced.lower() for k in ("threatening", "sinister", "intense", "dramatic"))

    # Farklı sahne türleri için infer_fallback_acting_prompt hiçbir zaman yasaklı düz kelime üretmemeli
    samples = [
        ("Watch out behind you!", -14.0),
        ("Please... don't shoot, hush.", -30.0),
        ("Try the fugu fish. It's to die for.", -22.0),
        ("Your contract has just been terminated.", -21.0),
        ("What are you doing here?", -20.0),
    ]
    for txt, lufs in samples:
        fb = infer_fallback_acting_prompt(txt, lufs)
        assert fb
        for banned in BANNED_FLAT_ACTING_WORDS:
            assert banned not in fb.lower()


def test_multi_model_automatic_fallback_on_429_and_503(tmp_path: Path) -> None:
    qm = GeminiQuotaManager(path=tmp_path / "fallback_kota.json")
    translator = GeminiTranslator(
        api_key="DUMMY_KEY",
        model_name="gemini-4-hydrogen",  # 20/gün modelle başla
        cache=TranslationCache(tmp_path / "fb_cache.db"),
        glossary=GlossaryManager(tmp_path / "fb_gloss.json"),
        quota_manager=qm,
        rpm=100,
    )
    translator._models_verified_once = True

    slept_durations: list[float] = []
    translator._sleep_cancellable = lambda sec, cancel_event=None: slept_durations.append(sec)  # type: ignore[method-assign]

    models_tried: list[str] = []

    class _FakeModels:
        def generate_content(self, model, contents, config):
            models_tried.append(model)
            if model == "gemini-4-hydrogen":
                # 1. modelde 429 günlük kota dolumu -> anında 500 kotalı 3.5 Flash Lite'a geçmeli!
                raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded per day.")
            if model == "gemini-3.5-flash-lite":
                # 2. modelde 503 geçici yoğunluk -> anında 500 kotalı 3.1 Flash Lite'a geçmeli!
                raise RuntimeError("503 UNAVAILABLE: Model overloaded.")
            return SimpleNamespace(text="Kesintisiz yedek model çevirisi.")

    translator._client = SimpleNamespace(models=_FakeModels())

    out = translator._generate_with_retry("Test prompt")
    assert out == "Kesintisiz yedek model çevirisi."
    assert models_tried == [
        "gemini-4-hydrogen",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    snap = qm.get_snapshot()
    assert snap.active_model_id == "gemini-3.1-flash-lite"
    # 4 Hydrogen dolu (20/20), 3.1 Flash Lite 1/500 kullanılmış olmalı
    st_h = next(m for m in snap.models if m.api_id == "gemini-4-hydrogen")
    st_31 = next(m for m in snap.models if m.api_id == "gemini-3.1-flash-lite")
    assert st_h.is_available is False
    assert st_31.used == 1


def test_429_retry_delay_parsing_and_503_when_fallback_disabled(tmp_path: Path) -> None:
    err1 = "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '55s'}]}}"
    err2 = 'ClientError: 429 RESOURCE_EXHAUSTED. Please retry in 42.5s. "retryDelay": "42.5s"'
    err3 = "quota exceeded; retry_delay { seconds: 58 }"
    assert parse_retry_delay_seconds(err1) == 55.0
    assert parse_retry_delay_seconds(err2) == 42.5
    assert parse_retry_delay_seconds(err3) == 58.0
    assert parse_retry_delay_seconds("500 internal error") is None

    translator = GeminiTranslator(
        api_key="DUMMY_KEY",
        cache=TranslationCache(tmp_path / "retry_cache.db"),
        glossary=GlossaryManager(tmp_path / "retry_gloss.json"),
        quota_manager=GeminiQuotaManager(path=tmp_path / "retry_kota.json"),
        rpm=100,
        enable_model_fallback=False,
    )
    translator._models_verified_once = True

    slept_durations: list[float] = []
    translator._sleep_cancellable = lambda sec, cancel_event=None: slept_durations.append(sec)  # type: ignore[method-assign]

    call_idx = {"n": 0}

    class _FakeModels:
        def generate_content(self, model, contents, config):
            call_idx["n"] += 1
            if call_idx["n"] == 1:
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED: Quota exceeded. {'retryDelay': '55s'}"
                )
            if call_idx["n"] == 2:
                raise RuntimeError("503 UNAVAILABLE: The model is overloaded.")
            return SimpleNamespace(text="Sözleşmeniz sona erdi.")

    translator._client = SimpleNamespace(models=_FakeModels())

    out = translator._generate_with_retry("Test prompt")
    assert out == "Sözleşmeniz sona erdi."
    assert call_idx["n"] == 3
    assert len(slept_durations) == 2
    assert abs(slept_durations[0] - 56.5) < 0.05
    assert 3.0 <= slept_durations[1] <= 8.5


def test_all_models_503_fast_switch_and_free_backup_translator(tmp_path: Path) -> None:
    qm = GeminiQuotaManager(path=tmp_path / "all503_kota.json")
    gm = GlossaryManager(path=tmp_path / "all503_gloss.json")
    gm.upsert("Agent 47", "Ajan 47")
    cache = TranslationCache(db_path=tmp_path / "all503_cache.db")

    translator = GeminiTranslator(
        api_key="DUMMY_KEY",
        model_name="gemini-3.5-flash-lite",
        cache=cache,
        glossary=gm,
        quota_manager=qm,
        rpm=4,
    )
    translator._models_verified_once = True

    slept_durations: list[float] = []
    translator._sleep_cancellable = lambda sec, cancel_event=None: slept_durations.append(sec)  # type: ignore[method-assign]

    models_tried: list[str] = []

    class _All503Models:
        def generate_content(self, model, contents, config):
            models_tried.append(model)
            raise RuntimeError(
                "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"
            )

    translator._client = SimpleNamespace(models=_All503Models())

    logs: list[str] = []
    batch_items = [
        BatchLineInput(
            item_id="L00001",
            source_text="Your contract is terminated, Agent 47!",
            duration_sec=2.8,
            lufs=-18.5,
        ),
        BatchLineInput(
            item_id="L00002",
            source_text="Welcome to your safehouse, Agent 47. Good luck.",
            duration_sec=3.6,
            lufs=-23.0,
        ),
    ]

    results = translator.translate_batch(batch_items, use_cache=True, log_cb=logs.append)

    # 1. 3.8 Flash dahil tüm 8 ücretsiz model sırayla denenmiş olmalı
    tried_set = set(models_tried)
    assert "gemini-3.8-flash" in tried_set
    assert len(tried_set) == 8
    # Aynı modelde 10 kez 15'er saniye bekleme yapılmamış olmalı (toplam bekleme en fazla 1 kısa 3sn deneme!)
    assert sum(slept_durations) <= 3.5

    # 2. 503 hataları kesinlikle günlük kota tüketimi olarak sayılmamalı (0/1120 kullanılmış olmalı)
    snap = qm.get_snapshot()
    assert snap.total_used == 0
    assert snap.total_remaining == 1120
    assert all(m.used == 0 and m.is_available for m in snap.models)

    # 3. Otomatik ücretsiz yedek çevirmen devreye girip replikleri çevirmiş ve dramatik acting_prompt üretmiş olmalı
    assert any("ücretsiz yedek çevirmenle devam edildi" in m for m in logs)
    assert len(results) == 2
    for item_id in ("L00001", "L00002"):
        res = results[item_id]
        assert res.translated_text
        assert "Ajan 47" in res.translated_text
        assert res.acting_prompt
        for banned in BANNED_FLAT_ACTING_WORDS:
            assert banned not in res.acting_prompt.lower()
