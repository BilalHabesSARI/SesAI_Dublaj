"""Çoklu Ücretsiz Gemini Model Kayıt Defteri, 11:00 TRT Günlük Kota Takibi ve Otomatik Model Geçişi (Fallback).

Kurallar:
- Tüm kota ve model doğrulama verileri yalnızca `./_veri/gemini_kota.json` dosyasında tutulur.
- 8 ücretsiz Gemini modelinin tamamı gerçek çalışan API kimlikleriyle tanımlanmıştır:
    1. 4 Hydrogen      (20/gün)  -> gemini-4-hydrogen
    2. 3.8 Flash       (20/gün)  -> gemini-3.8-flash
    3. 3.6 Flash       (20/gün)  -> gemini-3.6-flash
    4. 3 Flash         (20/gün)  -> gemini-3.5-flash  (eski gemini-3.0-flash / gemini-3-flash otomatik düzeltilir)
    5. 2.5 Flash       (20/gün)  -> gemini-2.5-flash
    6. 3.5 Flash Lite  (500/gün) -> gemini-3.5-flash-lite
    7. 3.1 Flash Lite  (500/gün) -> gemini-3.1-flash-lite
    8. 2.5 Flash Lite  (20/gün)  -> gemini-2.5-flash-lite
  Toplam Günlük Ücretsiz Kota = 1120 istek/gün (22 satırlık toplu çeviri ile ~24.640 satır/gün).
- Sayaçlar her gün Türkiye saatiyle 11:00 (TRT / UTC+3) itibarıyla otomatik sıfırlanır.
- Seçili modelde 429, 503, 404 veya günlük limit dolumu yaşandığında öncelikle
  500 kotalı açık modellere (3.5 Flash Lite / 3.1 Flash Lite) ve ardından diğer
  müsait modellere kesintisiz geçiş yapılır.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from core.portable_env import DATA_DIR

TRT_TIMEZONE = timezone(timedelta(hours=3), name="TRT")
DAILY_RESET_HOUR_TRT: int = 11
DEFAULT_QUOTA_PATH: Path = DATA_DIR / "gemini_kota.json"


@dataclass(frozen=True)
class GeminiModelSpec:
    """Tek bir ücretsiz Gemini modelinin tanımı ve günlük kotası."""

    key: str
    api_id: str
    short_name: str
    daily_limit: int
    fallback_priority: int
    aliases: tuple[str, ...] = ()

    @property
    def ui_label(self) -> str:
        return f"{self.short_name} ({self.daily_limit}/gün) — {self.api_id}"


# Kullanıcının istediği 8 ücretsiz model (gerçek ListModels API kimlikleriyle doğrulanmış)
GEMINI_FREE_MODELS: tuple[GeminiModelSpec, ...] = (
    GeminiModelSpec(
        key="4-hydrogen",
        api_id="gemini-4-hydrogen",
        short_name="4 Hydrogen",
        daily_limit=20,
        fallback_priority=3,
        aliases=("4-hydrogen", "4 hydrogen", "gemini-4-hydrogen", "models/gemini-4-hydrogen"),
    ),
    GeminiModelSpec(
        key="3.8-flash",
        api_id="gemini-3.8-flash",
        short_name="3.8 Flash",
        daily_limit=20,
        fallback_priority=4,
        aliases=("3.8-flash", "3.8 flash", "gemini-3.8-flash", "models/gemini-3.8-flash"),
    ),
    GeminiModelSpec(
        key="3.6-flash",
        api_id="gemini-3.6-flash",
        short_name="3.6 Flash",
        daily_limit=20,
        fallback_priority=5,
        aliases=("3.6-flash", "3.6 flash", "gemini-3.6-flash", "models/gemini-3.6-flash"),
    ),
    GeminiModelSpec(
        key="3-flash",
        api_id="gemini-3.5-flash",
        short_name="3 Flash",
        daily_limit=20,
        fallback_priority=6,
        aliases=(
            "3-flash",
            "3 flash",
            "3.0-flash",
            "3.0 flash",
            "3.5-flash",
            "3.5 flash",
            "gemini-3-flash",
            "gemini-3.0-flash",
            "gemini-3-flash-preview",
            "gemini-3.5-flash",
            "models/gemini-3-flash",
            "models/gemini-3.0-flash",
            "models/gemini-3.5-flash",
        ),
    ),
    GeminiModelSpec(
        key="2.5-flash",
        api_id="gemini-2.5-flash",
        short_name="2.5 Flash",
        daily_limit=20,
        fallback_priority=7,
        aliases=("2.5-flash", "2.5 flash", "gemini-2.5-flash", "models/gemini-2.5-flash"),
    ),
    GeminiModelSpec(
        key="3.5-flash-lite",
        api_id="gemini-3.5-flash-lite",
        short_name="3.5 Flash Lite",
        daily_limit=500,
        fallback_priority=1,  # Otomatik geçişte 1. öncelik (500/gün yüksek kota)
        aliases=(
            "3.5-flash-lite",
            "3.5 flash lite",
            "gemini-3.5-flash-lite",
            "models/gemini-3.5-flash-lite",
            "gemini-3.0-flash-lite",
            "gemini-3-flash-lite",
        ),
    ),
    GeminiModelSpec(
        key="3.1-flash-lite",
        api_id="gemini-3.1-flash-lite",
        short_name="3.1 Flash Lite",
        daily_limit=500,
        fallback_priority=2,  # Otomatik geçişte 2. öncelik (500/gün yüksek kota)
        aliases=(
            "3.1-flash-lite",
            "3.1 flash lite",
            "gemini-3.1-flash-lite",
            "models/gemini-3.1-flash-lite",
        ),
    ),
    GeminiModelSpec(
        key="2.5-flash-lite",
        api_id="gemini-2.5-flash-lite",
        short_name="2.5 Flash Lite",
        daily_limit=20,
        fallback_priority=8,
        aliases=(
            "2.5-flash-lite",
            "2.5 flash lite",
            "gemini-2.5-flash-lite",
            "models/gemini-2.5-flash-lite",
        ),
    ),
)

TOTAL_DAILY_FREE_QUOTA: int = sum(m.daily_limit for m in GEMINI_FREE_MODELS)  # 1120

_SPEC_BY_API_ID: dict[str, GeminiModelSpec] = {m.api_id: m for m in GEMINI_FREE_MODELS}
_ALIAS_MAP: dict[str, GeminiModelSpec] = {}
for _spec in GEMINI_FREE_MODELS:
    _ALIAS_MAP[_spec.api_id.lower()] = _spec
    _ALIAS_MAP[_spec.key.lower()] = _spec
    _ALIAS_MAP[_spec.short_name.lower()] = _spec
    for _al in _spec.aliases:
        _ALIAS_MAP[_al.lower()] = _spec


def resolve_model_spec(name_or_id: str) -> GeminiModelSpec:
    """Verilen model adını veya eski/alias ID'yi gerçek çalışan GeminiModelSpec'e çözümler.

    Örneğin 'gemini-3.0-flash', 'gemini-3-flash' veya '3 Flash' -> 'gemini-3.5-flash'.
    Tanınmayan bir değer gelirse 500 kotalı 'gemini-3.5-flash-lite' modeline döner.
    """
    raw = (name_or_id or "").strip()
    if "—" in raw:
        raw = raw.split("—")[-1].strip()
    cleaned = raw.lower().replace("models/", "").strip()
    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]

    for spec in GEMINI_FREE_MODELS:
        if spec.api_id in cleaned or spec.short_name.lower() in cleaned:
            return spec

    return _SPEC_BY_API_ID["gemini-3.5-flash-lite"]


def normalize_gemini_model_id(name_or_id: str) -> str:
    """Eski veya hatalı model adını gerçek çalışan API model ID'sine dönüştürür."""
    return resolve_model_spec(name_or_id).api_id


def compute_quota_window_id(now: Optional[datetime] = None) -> str:
    """Türkiye saatiyle (TRT, UTC+3) her gün 11:00'de değişen kota dönem kimliğini hesaplar.

    Örnek:
      - 2026-11-23 10:59 TRT -> '2026-11-22'
      - 2026-11-23 11:00 TRT -> '2026-11-23' (otomatik sıfırlama anı)
    """
    dt = now if now is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    trt_now = dt.astimezone(TRT_TIMEZONE)
    shifted = trt_now - timedelta(hours=DAILY_RESET_HOUR_TRT)
    return shifted.strftime("%Y-%m-%d")


def next_reset_time_trt_str(now: Optional[datetime] = None) -> str:
    """Bir sonraki 11:00 TRT sıfırlama zamanını okunaklı metin olarak döndürür."""
    dt = now if now is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    trt_now = dt.astimezone(TRT_TIMEZONE)
    today_reset = trt_now.replace(hour=DAILY_RESET_HOUR_TRT, minute=0, second=0, microsecond=0)
    if trt_now >= today_reset:
        nxt = today_reset + timedelta(days=1)
    else:
        nxt = today_reset
    return nxt.strftime("%d.%m.%Y 11:00 TRT")


@dataclass
class ModelQuotaState:
    """Tek bir modelin güncel kullanım ve müsaitlik durumu."""

    api_id: str
    short_name: str
    used: int
    daily_limit: int
    exhausted: bool = False
    verified_on_server: bool = True
    last_error: str = ""

    @property
    def remaining(self) -> int:
        if self.exhausted or not self.verified_on_server:
            return 0
        return max(0, self.daily_limit - self.used)

    @property
    def is_available(self) -> bool:
        return (
            self.verified_on_server
            and (not self.exhausted)
            and (self.used < self.daily_limit)
        )


@dataclass
class QuotaSnapshot:
    """Arayüzde gösterilecek canlı kota özeti."""

    window_id: str
    active_model_id: str
    active_short_name: str
    total_used: int
    total_limit: int
    total_remaining: int
    next_reset_label: str
    models: list[ModelQuotaState] = field(default_factory=list)


class GeminiQuotaManager:
    """./_veri/gemini_kota.json üzerinde 11:00 TRT sıfırlamalı canlı kota ve otomatik model yöneticisi."""

    def __init__(
        self,
        path: Path = DEFAULT_QUOTA_PATH,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.path = path
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._listeners: list[Callable[[QuotaSnapshot], None]] = []

        self._window_id: str = ""
        self._active_model_id: str = "gemini-3.5-flash-lite"
        self._states: dict[str, ModelQuotaState] = {}
        self._server_verified_ids: set[str] = set()
        self.load()

    def add_listener(self, callback: Callable[[QuotaSnapshot], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[QuotaSnapshot], None]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify(self) -> None:
        snap = self.get_snapshot()
        for cb in list(self._listeners):
            try:
                cb(snap)
            except Exception:
                pass

    def _init_fresh_states(self, window_id: str) -> None:
        self._window_id = window_id
        self._states = {
            spec.api_id: ModelQuotaState(
                api_id=spec.api_id,
                short_name=spec.short_name,
                used=0,
                daily_limit=spec.daily_limit,
                exhausted=False,
                verified_on_server=True,
                last_error="",
            )
            for spec in GEMINI_FREE_MODELS
        }

    def _check_and_reset_if_new_day_locked(self) -> bool:
        """Saat 11:00 TRT'yi geçtiyse tüm günlük sayaçları otomatik sıfırlar."""
        cur_win = compute_quota_window_id(self._now_fn())
        if self._window_id != cur_win or not self._states:
            self._init_fresh_states(cur_win)
            self._save_locked()
            return True
        return False

    def load(self) -> QuotaSnapshot:
        with self._lock:
            cur_win = compute_quota_window_id(self._now_fn())
            self._init_fresh_states(cur_win)

            if self.path.is_file():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    saved_win = str(raw.get("window_id") or "")
                    saved_active = str(raw.get("active_model_id") or "")
                    if saved_active:
                        self._active_model_id = normalize_gemini_model_id(saved_active)

                    # Aynı 11:00 TRT günü içindeysek kayıtlı sayaçları yükle;
                    # yeni gün başladıysa (11:00 TRT geçtiyse) sıfırlanmış hali koru ve kaydet!
                    if saved_win == cur_win and isinstance(raw.get("models"), dict):
                        for api_id, st in self._states.items():
                            m_raw = raw["models"].get(api_id)
                            if isinstance(m_raw, dict):
                                st.used = max(0, int(m_raw.get("used", 0)))
                                st.exhausted = bool(m_raw.get("exhausted", False)) or (
                                    st.used >= st.daily_limit
                                )
                                st.verified_on_server = bool(
                                    m_raw.get("verified_on_server", True)
                                )
                                st.last_error = str(m_raw.get("last_error", ""))
                                if st.verified_on_server and "404" in st.last_error:
                                    st.used = 0
                                    st.exhausted = False
                                    st.last_error = ""
                    else:
                        self._save_locked()
                except Exception:
                    self._save_locked()
            else:
                self._save_locked()

        return self.get_snapshot()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window_id": self._window_id,
            "reset_schedule": "Her gün 11:00 TRT (UTC+3)",
            "active_model_id": self._active_model_id,
            "models": {
                api_id: {
                    "short_name": st.short_name,
                    "used": st.used,
                    "daily_limit": st.daily_limit,
                    "exhausted": st.exhausted,
                    "verified_on_server": st.verified_on_server,
                    "last_error": st.last_error,
                }
                for api_id, st in self._states.items()
            },
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def reset_all_quotas(self) -> QuotaSnapshot:
        """Tüm model sayaçlarını sıfırlar ve kaydeder."""
        with self._lock:
            cur_win = compute_quota_window_id(self._now_fn())
            self._init_fresh_states(cur_win)
            self._save_locked()
        self._notify()
        return self.get_snapshot()

    def set_active_model(self, model_name_or_id: str) -> str:
        spec = resolve_model_spec(model_name_or_id)
        with self._lock:
            self._check_and_reset_if_new_day_locked()
            self._active_model_id = spec.api_id
            self._save_locked()
        self._notify()
        return spec.api_id

    def verify_models_with_api(
        self,
        available_api_model_names: Iterable[str],
    ) -> dict[str, bool]:
        """ListModels sorgusundan dönen gerçek model listesiyle 8 modeli doğrular."""
        clean_set = {
            m.strip().replace("models/", "").lower()
            for m in available_api_model_names
            if m and m.strip()
        }
        if not clean_set:
            return {}

        verification: dict[str, bool] = {}
        with self._lock:
            self._check_and_reset_if_new_day_locked()
            self._server_verified_ids = clean_set
            for spec in GEMINI_FREE_MODELS:
                exists = spec.api_id.lower() in clean_set
                st = self._states[spec.api_id]
                st.verified_on_server = exists
                if not exists:
                    st.last_error = "API ListModels listesinde bulunamadı"
                elif "404" in st.last_error or "ListModels" in st.last_error:
                    st.used = 0
                    st.exhausted = False
                    st.last_error = ""
                verification[spec.api_id] = exists
            self._save_locked()
        self._notify()
        return verification

    def record_usage(self, model_name_or_id: str, count: int = 1) -> QuotaSnapshot:
        """Başarılı bir API isteği sonrası ilgili modelin günlük sayacını artırır."""
        spec = resolve_model_spec(model_name_or_id)
        with self._lock:
            self._check_and_reset_if_new_day_locked()
            st = self._states[spec.api_id]
            st.used = min(st.daily_limit, st.used + max(1, int(count)))
            if st.used >= st.daily_limit:
                st.exhausted = True
            self._active_model_id = spec.api_id
            self._save_locked()
        self._notify()
        return self.get_snapshot()

    def mark_model_exhausted(
        self,
        model_name_or_id: str,
        reason: str = "429 Kota Dolu",
    ) -> QuotaSnapshot:
        """429 günlük kota hatası alan modeli o gün (11:00 TRT'ye kadar) dolu olarak işaretler."""
        spec = resolve_model_spec(model_name_or_id)
        with self._lock:
            self._check_and_reset_if_new_day_locked()
            st = self._states[spec.api_id]
            st.exhausted = True
            st.used = st.daily_limit
            st.last_error = reason
            self._save_locked()
        self._notify()
        return self.get_snapshot()

    def select_available_model(
        self,
        preferred_model: Optional[str] = None,
        exclude_api_ids: Optional[set[str]] = None,
    ) -> Optional[GeminiModelSpec]:
        """Seçili model müsaitse onu, doluysa/hatalıysa öncelik sırasına göre (500 kotalı Lite modeller önce) sıradaki müsait modeli döndürür."""
        excluded = {x.lower() for x in (exclude_api_ids or set())}
        with self._lock:
            self._check_and_reset_if_new_day_locked()

            # 1. Kullanıcının seçtiği model müsaitse ve dışlanmadıysa onu kullan
            pref_id = (
                normalize_gemini_model_id(preferred_model)
                if preferred_model
                else self._active_model_id
            )
            if pref_id.lower() not in excluded and pref_id in self._states:
                st = self._states[pref_id]
                if st.is_available:
                    return _SPEC_BY_API_ID[pref_id]

            # 2. Otomatik Geçiş (Fallback): Öncelikle 500 kotalı 3.5 Flash Lite ve 3.1 Flash Lite,
            #    ardından 4 Hydrogen -> 3.8 Flash -> 3.6 Flash -> 3 Flash -> 2.5 Flash -> 2.5 Flash Lite
            ordered_specs = sorted(GEMINI_FREE_MODELS, key=lambda s: s.fallback_priority)
            for spec in ordered_specs:
                if spec.api_id.lower() in excluded:
                    continue
                st = self._states[spec.api_id]
                if st.is_available:
                    return spec

            return None

    def get_snapshot(self) -> QuotaSnapshot:
        with self._lock:
            self._check_and_reset_if_new_day_locked()
            states_copy = [
                ModelQuotaState(
                    api_id=st.api_id,
                    short_name=st.short_name,
                    used=st.used,
                    daily_limit=st.daily_limit,
                    exhausted=st.exhausted,
                    verified_on_server=st.verified_on_server,
                    last_error=st.last_error,
                )
                for spec in GEMINI_FREE_MODELS
                for st in [self._states[spec.api_id]]
            ]
            total_limit = sum(s.daily_limit for s in states_copy if s.verified_on_server)
            total_rem = sum(s.remaining for s in states_copy)
            total_used = total_limit - total_rem
            act_spec = resolve_model_spec(self._active_model_id)
            return QuotaSnapshot(
                window_id=self._window_id,
                active_model_id=act_spec.api_id,
                active_short_name=act_spec.short_name,
                total_used=max(0, total_used),
                total_limit=total_limit,
                total_remaining=max(0, total_rem),
                next_reset_label="11:00 TRT",
                models=states_copy,
            )
