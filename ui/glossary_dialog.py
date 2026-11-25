"""Kalıcı Çeviri Sözlüğü (Glossary) ve TTS Telaffuz Sözlüğü Düzenleyici Penceresi."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.glossary import GlossaryEntry, GlossaryManager
from core.tts_pronunciation import PronunciationRule, TTSPronunciationManager


class GlossaryDialog(QDialog):
    """./_veri/sozluk.json (Çeviri Sözlüğü) ve ./_veri/tts_telaffuz.json (TTS Telaffuz) düzenleyicisi."""

    def __init__(
        self,
        glossary_manager: GlossaryManager,
        parent: Optional[QWidget] = None,
        pronunciation_manager: Optional[TTSPronunciationManager] = None,
        initial_tab: int = 0,
    ) -> None:
        super().__init__(parent)
        self.glossary = glossary_manager
        self.pronunciation = pronunciation_manager or TTSPronunciationManager()
        self.setWindowTitle("Sözlük ve TTS Telaffuz Düzenleyici")
        self.resize(780, 540)
        self._build_ui()
        self._reload_table()
        self._reload_pron_table()
        if 0 <= initial_tab < self.tabs.count():
            self.tabs.setCurrentIndex(initial_tab)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_glossary_tab(), "📖 Çeviri Sözlüğü (Glossary)")
        self.tabs.addTab(self._build_pronunciation_tab(), "🗣️ TTS Telaffuz Sözlüğü (Kısaltmalar)")
        layout.addWidget(self.tabs, 1)

        # Alt kayıt / kapat çubuğu
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.save_close_btn = QPushButton("✅ Kaydet ve Kapat")
        self.save_close_btn.setObjectName("SuccessBtn")
        self.save_close_btn.clicked.connect(self._on_save_and_close)
        btn_row.addWidget(self.save_close_btn)
        layout.addLayout(btn_row)

    # ==========================================================================
    # SEKME 1: ÇEVİRİ SÖZLÜĞÜ (CASE-SENSITIVE GLOSSARY)
    # ==========================================================================
    def _build_glossary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        info = QLabel(
            "Oyun karakter adları, mekânlar ve özel terimler için çeviri karşılıkları tanımlayın. "
            "Eşleştirmeler kesin olarak BÜYÜK/KÜÇÜK HARF DUYARLIDIR (örn. 'Shadow' ile 'shadow' ayrı tutulur)."
        )
        info.setObjectName("MutedLabel")
        info.setWordWrap(True)
        layout.addWidget(info)

        form_row = QHBoxLayout()
        form_row.setSpacing(8)

        self.src_input = QLineEdit()
        self.src_input.setPlaceholderText("Kaynak Terim (ör. Iron Citadel)")

        self.tgt_input = QLineEdit()
        self.tgt_input.setPlaceholderText("Türkçe Karşılık (ör. Demir Kale)")

        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Not (opsiyonel, ör. Şehir adı)")

        self.add_btn = QPushButton("➕ Ekle / Güncelle")
        self.add_btn.setObjectName("PrimaryBtn")
        self.add_btn.clicked.connect(self._on_add)

        form_row.addWidget(self.src_input, 2)
        form_row.addWidget(self.tgt_input, 2)
        form_row.addWidget(self.note_input, 2)
        form_row.addWidget(self.add_btn)
        layout.addLayout(form_row)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Çeviri sözlüğünde ara...")
        self.search_input.textChanged.connect(self._apply_filter)
        self.count_label = QLabel("0 terim")
        self.count_label.setObjectName("StatusPillInfo")
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.count_label)
        layout.addLayout(search_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Kaynak Terim (Case-Sensitive)", "Türkçe Karşılık", "Açıklama / Not"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        del_row = QHBoxLayout()
        self.del_btn = QPushButton("🗑️ Seçili Terimi Sil")
        self.del_btn.setObjectName("DangerBtn")
        self.del_btn.clicked.connect(self._on_delete)
        del_row.addWidget(self.del_btn)
        del_row.addStretch()
        layout.addLayout(del_row)
        return tab

    # ==========================================================================
    # SEKME 2: TTS TELAFFUZ SÖZLÜĞÜ (./_veri/tts_telaffuz.json)
    # ==========================================================================
    def _build_pronunciation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        info = QLabel(
            "Tablodaki Türkçe dublaj metni okunaklı haliyle (örn. 'ICA') kalırken, yalnızca seslendirme motoruna "
            "(VoxCPM2 vb.) gönderilen metinde tam kelime eşleşmesiyle otomatik telaffuz dönüşümü uygulanır "
            "(örn. 'ICA' -> 'Ay-Si-Ey'). Veriler './_veri/tts_telaffuz.json' içinde saklanır."
        )
        info.setObjectName("MutedLabel")
        info.setWordWrap(True)
        layout.addWidget(info)

        form_row = QHBoxLayout()
        form_row.setSpacing(8)

        self.pron_src_input = QLineEdit()
        self.pron_src_input.setPlaceholderText("Yazılı Kısaltma / Kelime (ör. ICA)")

        self.pron_spoken_input = QLineEdit()
        self.pron_spoken_input.setPlaceholderText("TTS Okunuşu (ör. Ay-Si-Ey)")

        self.pron_note_input = QLineEdit()
        self.pron_note_input.setPlaceholderText("Not (opsiyonel)")

        self.pron_add_btn = QPushButton("➕ Ekle / Güncelle")
        self.pron_add_btn.setObjectName("PrimaryBtn")
        self.pron_add_btn.clicked.connect(self._on_pron_add)

        form_row.addWidget(self.pron_src_input, 2)
        form_row.addWidget(self.pron_spoken_input, 2)
        form_row.addWidget(self.pron_note_input, 2)
        form_row.addWidget(self.pron_add_btn)
        layout.addLayout(form_row)

        search_row = QHBoxLayout()
        self.pron_search_input = QLineEdit()
        self.pron_search_input.setPlaceholderText("TTS telaffuz kurallarında ara...")
        self.pron_search_input.textChanged.connect(self._apply_pron_filter)
        self.pron_count_label = QLabel("0 kural")
        self.pron_count_label.setObjectName("StatusPillInfo")
        search_row.addWidget(self.pron_search_input, 1)
        search_row.addWidget(self.pron_count_label)
        layout.addLayout(search_row)

        self.pron_table = QTableWidget(0, 3)
        self.pron_table.setHorizontalHeaderLabels(
            ["Yazılı Metin (Tam Kelime)", "Ses Motoru Okunuşu (TTS)", "Açıklama / Not"]
        )
        self.pron_table.setAlternatingRowColors(True)
        self.pron_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pron_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        p_header = self.pron_table.horizontalHeader()
        p_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        p_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        p_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.pron_table.verticalHeader().setVisible(False)
        layout.addWidget(self.pron_table, 1)

        del_row = QHBoxLayout()
        self.pron_del_btn = QPushButton("🗑️ Seçili Kuralı Sil")
        self.pron_del_btn.setObjectName("DangerBtn")
        self.pron_del_btn.clicked.connect(self._on_pron_delete)
        del_row.addWidget(self.pron_del_btn)
        del_row.addStretch()
        layout.addLayout(del_row)
        return tab

    def _reload_table(self) -> None:
        entries = self.glossary.list_entries()
        self.table.setRowCount(0)
        for entry in entries:
            self._append_row(entry)
        self._update_count()

    def _append_row(self, entry: GlossaryEntry) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(entry.source))
        self.table.setItem(row, 1, QTableWidgetItem(entry.target))
        self.table.setItem(row, 2, QTableWidgetItem(entry.note))

    def _update_count(self) -> None:
        self.count_label.setText(f"{self.table.rowCount()} terim")

    def _on_add(self) -> None:
        src = self.src_input.text().strip()
        tgt = self.tgt_input.text().strip()
        note = self.note_input.text().strip()
        if not src or not tgt:
            QMessageBox.warning(self, "Eksik Bilgi", "Kaynak terim ve Türkçe karşılık boş olamaz.")
            return

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == src:
                self.table.item(row, 1).setText(tgt)
                self.table.item(row, 2).setText(note)
                self.src_input.clear()
                self.tgt_input.clear()
                self.note_input.clear()
                self.src_input.setFocus()
                return

        self._append_row(GlossaryEntry(source=src, target=tgt, note=note))
        self._update_count()
        self.src_input.clear()
        self.tgt_input.clear()
        self.note_input.clear()
        self.src_input.setFocus()

    def _on_delete(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self._update_count()

    def _apply_filter(self, text: str) -> None:
        q = text.strip().lower()
        for row in range(self.table.rowCount()):
            visible = True
            if q:
                row_text = " ".join(
                    (self.table.item(row, col).text() if self.table.item(row, col) else "")
                    for col in range(3)
                ).lower()
                visible = q in row_text
            self.table.setRowHidden(row, not visible)

    def _reload_pron_table(self) -> None:
        rules = self.pronunciation.list_rules()
        self.pron_table.setRowCount(0)
        for rule in rules:
            self._append_pron_row(rule)
        self._update_pron_count()

    def _append_pron_row(self, rule: PronunciationRule) -> None:
        row = self.pron_table.rowCount()
        self.pron_table.insertRow(row)
        self.pron_table.setItem(row, 0, QTableWidgetItem(rule.source))
        self.pron_table.setItem(row, 1, QTableWidgetItem(rule.spoken))
        self.pron_table.setItem(row, 2, QTableWidgetItem(rule.note))

    def _update_pron_count(self) -> None:
        self.pron_count_label.setText(f"{self.pron_table.rowCount()} kural")

    def _on_pron_add(self) -> None:
        src = self.pron_src_input.text().strip()
        spoken = self.pron_spoken_input.text().strip()
        note = self.pron_note_input.text().strip()
        if not src or not spoken:
            QMessageBox.warning(self, "Eksik Bilgi", "Yazılı terim ve TTS okunuşu boş olamaz.")
            return

        for row in range(self.pron_table.rowCount()):
            item = self.pron_table.item(row, 0)
            if item and item.text() == src:
                self.pron_table.item(row, 1).setText(spoken)
                self.pron_table.item(row, 2).setText(note)
                self.pron_src_input.clear()
                self.pron_spoken_input.clear()
                self.pron_note_input.clear()
                self.pron_src_input.setFocus()
                return

        self._append_pron_row(PronunciationRule(source=src, spoken=spoken, note=note))
        self._update_pron_count()
        self.pron_src_input.clear()
        self.pron_spoken_input.clear()
        self.pron_note_input.clear()
        self.pron_src_input.setFocus()

    def _on_pron_delete(self) -> None:
        row = self.pron_table.currentRow()
        if row < 0:
            return
        self.pron_table.removeRow(row)
        self._update_pron_count()

    def _apply_pron_filter(self, text: str) -> None:
        q = text.strip().lower()
        for row in range(self.pron_table.rowCount()):
            visible = True
            if q:
                row_text = " ".join(
                    (self.pron_table.item(row, col).text() if self.pron_table.item(row, col) else "")
                    for col in range(3)
                ).lower()
                visible = q in row_text
            self.pron_table.setRowHidden(row, not visible)

    def _on_save_and_close(self) -> None:
        entries: list[GlossaryEntry] = []
        for row in range(self.table.rowCount()):
            src = (self.table.item(row, 0).text() if self.table.item(row, 0) else "").strip()
            tgt = (self.table.item(row, 1).text() if self.table.item(row, 1) else "").strip()
            note = (self.table.item(row, 2).text() if self.table.item(row, 2) else "").strip()
            if src and tgt:
                entries.append(GlossaryEntry(source=src, target=tgt, note=note))
        self.glossary.replace_all(entries)

        rules: list[PronunciationRule] = []
        for row in range(self.pron_table.rowCount()):
            src = (self.pron_table.item(row, 0).text() if self.pron_table.item(row, 0) else "").strip()
            spoken = (
                self.pron_table.item(row, 1).text() if self.pron_table.item(row, 1) else ""
            ).strip()
            note = (
                self.pron_table.item(row, 2).text() if self.pron_table.item(row, 2) else ""
            ).strip()
            if src and spoken:
                rules.append(PronunciationRule(source=src, spoken=spoken, note=note))
        self.pronunciation.replace_all(rules)
        self.accept()
