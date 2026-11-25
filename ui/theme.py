"""Modern, Ferah ve Göz Yormayan Koyu Tema (Dark Mode) QSS Stilleri."""

DARK_THEME_QSS = """
/* ===== Genel Pencere ve Yazı Tipi ===== */
QMainWindow, QDialog {
    background-color: #0E1219;
    color: #E6EDF5;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QWidget {
    color: #E6EDF5;
}

QLabel {
    background: transparent;
}

QLabel#AppTitle {
    font-size: 19px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 0.2px;
}

QLabel#AppSubtitle {
    font-size: 12px;
    color: #8B9BB4;
}

QLabel#SectionTitle {
    font-size: 13px;
    font-weight: 700;
    color: #CBD5E1;
    padding-bottom: 2px;
}

QLabel#MutedLabel {
    color: #8B9BB4;
    font-size: 12px;
}

QLabel#SummaryChip {
    background-color: #172030;
    color: #CBD5E1;
    border: 1px solid #28354A;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
}

QLabel#QuotaHeaderChip {
    background-color: #112622;
    color: #34D399;
    border: 1px solid #1C5E4D;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
}

QPushButton#QuotaPillAvailable {
    background-color: #12211E;
    color: #A7F3D0;
    border: 1px solid #1D4D41;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11.5px;
    font-weight: 600;
}

QPushButton#QuotaPillAvailable:hover {
    background-color: #18302B;
    border-color: #2D7A67;
}

QPushButton#QuotaPillActive {
    background-color: #15382E;
    color: #ECFDF5;
    border: 1.5px solid #10B981;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11.5px;
    font-weight: 700;
}

QPushButton#QuotaPillExhausted {
    background-color: #2A1418;
    color: #FCA5A5;
    border: 1px solid #6E232C;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11.5px;
    font-weight: 600;
}

QLabel#StatusPillOk {
    background-color: #0E2A22;
    color: #34D399;
    border: 1px solid #165947;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#StatusPillWarn {
    background-color: #2B210E;
    color: #FBBF24;
    border: 1px solid #664912;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#StatusPillInfo {
    background-color: #13223A;
    color: #60A5FA;
    border: 1px solid #1E3A5F;
    border-radius: 11px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#FooterSignature {
    color: #8B9BB4;
    font-size: 12px;
    padding: 6px 12px;
    background-color: #121720;
    border-top: 1px solid #1F2836;
}

/* ===== Sekmeli Üst Kontrol Paneli ===== */
QTabWidget#ControlTabs::pane {
    border: 1px solid #242E3E;
    border-radius: 10px;
    background-color: #141A24;
    top: -1px;
}

QTabWidget#ControlTabs QTabBar::tab {
    background-color: #11161F;
    color: #94A3B8;
    border: 1px solid #222C3C;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 16px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 12.5px;
}

QTabWidget#ControlTabs QTabBar::tab:hover {
    background-color: #18202E;
    color: #E2E8F0;
}

QTabWidget#ControlTabs QTabBar::tab:selected {
    background-color: #141A24;
    color: #60A5FA;
    border-color: #2E3D54;
}

/* ===== Kartlar ve Gruplar ===== */
QFrame#Card {
    background-color: #161D28;
    border: 1px solid #242E3E;
    border-radius: 9px;
}

QFrame#PreviewBar {
    background-color: #141A24;
    border: 1px solid #242E3E;
    border-radius: 8px;
}

/* ===== Metin Kutuları, Seçiciler ve Sayı Kutuları ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1C2432;
    color: #F1F5F9;
    border: 1px solid #2E3A4D;
    border-radius: 6px;
    padding: 5px 10px;
    min-height: 22px;
    selection-background-color: #3B82F6;
}

QSpinBox, QDoubleSpinBox {
    padding-right: 6px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #3B82F6;
    background-color: #1F2838;
}

QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #141923;
    color: #64748B;
    border-color: #202836;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #1C2432;
    color: #F1F5F9;
    border: 1px solid #334155;
    selection-background-color: #2563EB;
    outline: none;
}

/* ===== Kaydırıcılar (Sliders) ===== */
QSlider::groove:horizontal {
    height: 6px;
    background: #232D3F;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #F8FAFC;
    border: 2px solid #3B82F6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #DBEAFE;
}

QCheckBox {
    spacing: 8px;
    color: #CBD5E1;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3B4960;
    background-color: #1C2432;
}

QCheckBox::indicator:checked {
    background-color: #3B82F6;
    border-color: #3B82F6;
}

/* ===== Butonlar ===== */
QPushButton {
    background-color: #222B3B;
    color: #E2E8F0;
    border: 1px solid #323F54;
    border-radius: 7px;
    padding: 6px 13px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2A3549;
    border-color: #465773;
}

QPushButton:pressed {
    background-color: #1B2331;
}

QPushButton:disabled {
    background-color: #161B24;
    color: #556275;
    border-color: #212936;
}

QPushButton#PrimaryBtn {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
    padding: 8px 18px;
}

QPushButton#PrimaryBtn:hover {
    background-color: #1D4ED8;
}

QPushButton#PrimaryBtn:disabled {
    background-color: #1A2742;
    color: #64748B;
    border-color: #223252;
}

QPushButton#SuccessBtn {
    background-color: #059669;
    color: #FFFFFF;
    border: 1px solid #10B981;
    padding: 8px 16px;
}

QPushButton#SuccessBtn:hover {
    background-color: #047857;
}

QPushButton#SuccessBtn:disabled {
    background-color: #132A24;
    color: #58776D;
    border-color: #1A3A31;
}

QPushButton#WarningBtn {
    background-color: #B45309;
    color: #FFFBEB;
    border: 1px solid #D97706;
    padding: 8px 15px;
}

QPushButton#WarningBtn:hover {
    background-color: #92400E;
}

QPushButton#DangerBtn {
    background-color: #B91C1C;
    color: #FEF2F2;
    border: 1px solid #DC2626;
    padding: 8px 16px;
}

QPushButton#DangerBtn:hover {
    background-color: #991B1B;
}

QPushButton#DangerBtn:disabled {
    background-color: #2A171A;
    color: #7A5458;
    border-color: #3D2024;
}

/* ===== Altyapı Mod Seçici (Segmented Toggle) ===== */
QPushButton#ModeBtn {
    background-color: #171E2A;
    color: #94A3B8;
    border: 1px solid #2A3548;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton#ModeBtn:hover {
    color: #E2E8F0;
    border-color: #3E4E68;
}

QPushButton#ModeBtn[active="true"] {
    background-color: #1E3A6D;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
}

QPushButton#ModeGpuBtn {
    background-color: #171E2A;
    color: #94A3B8;
    border: 1px solid #2A3548;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton#ModeGpuBtn:hover {
    color: #E2E8F0;
    border-color: #3E4E68;
}

QPushButton#ModeGpuBtn[active="true"] {
    background-color: #37236B;
    color: #FFFFFF;
    border: 1px solid #8B5CF6;
}

/* ===== Tablo (Diyalog Listesi) ===== */
QTableWidget {
    background-color: #131822;
    alternate-background-color: #171D29;
    gridline-color: #222B3A;
    border: 1px solid #242E3E;
    border-radius: 8px;
    selection-background-color: #1E3A5F;
    selection-color: #F8FAFC;
    outline: none;
}

QTableWidget::item {
    padding: 5px 8px;
    border: none;
}

QHeaderView::section {
    background-color: #1A2230;
    color: #94A3B8;
    font-weight: 700;
    font-size: 12px;
    padding: 7px 8px;
    border: none;
    border-right: 1px solid #242E3E;
    border-bottom: 1px solid #242E3E;
}

/* ===== Canlı Günlük (Log) ===== */
QPlainTextEdit#LogConsole {
    background-color: #0B0E14;
    color: #CBD5E1;
    border: 1px solid #222B3A;
    border-radius: 8px;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 12px;
    padding: 6px;
}

/* ===== İlerleme Çubuğu ===== */
QProgressBar {
    background-color: #1A2230;
    border: 1px solid #293447;
    border-radius: 7px;
    text-align: center;
    color: #F8FAFC;
    font-weight: 700;
    font-size: 11px;
    min-height: 20px;
}

QProgressBar::chunk {
    background-color: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #2563EB, stop: 1 #10B981
    );
    border-radius: 6px;
}

/* ===== Splitter & Scrollbars ===== */
QSplitter::handle {
    background-color: #1F2836;
    height: 4px;
    margin: 2px 0;
    border-radius: 2px;
}

QScrollBar:vertical {
    background: #11161F;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2C384D;
    min-height: 26px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3E4F6B;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #11161F;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background: #2C384D;
    min-width: 26px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""
