# 🎙️ SesAI Dublaj Stüdyosu (v1.0 Beta)

![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey.svg)
![Taşınabilir](https://img.shields.io/badge/Ta%C5%9F%C4%B1nabilir-%25100%20Portable-emerald.svg)

**SesAI Dublaj Stüdyosu**, PC oyunlarındaki karakter konuşmalarını analiz eden, müzik ve efektleri atlayarak yalnızca insan konuşmalarını ayıklayan, sinematik dublaj yönetmeni istemiyle Türkçeye çeviren, karakterin kendi ses tonu ve duygusuyla dublajlayan ve SHA-256 bütünlük doğrulamasıyla geri alınabilir biçimde oyuna uygulayan **%100 taşınabilir (portable)** açık kaynaklı Windows masaüstü uygulamasıdır.

---

## 📸 Ekran Görüntüleri

> *v1.0 Beta ekran görüntüleri eklenecektir.*  
> *(Önerilen yerleşim: `docs/screenshots/ana_pencere.png` ve `docs/screenshots/hitman3_rpkg_sekmesi.png`)*

```text
+-------------------------------------------------------------------+
|  [ Ana Stüdyo Ekran Görüntüsü — 4 Sekmeli Koyu Tema & Kota Barı ] |
+-------------------------------------------------------------------+
```

---

## ✨ Öne Çıkan Özellikler

### 1. Çoklu Yapay Zeka Ses Motorları (Ücretsiz Yerel GPU, Colab Bulut veya API)
* **VoxCPM2 (48 kHz — Ücretsiz Yerel / Colab):** Karakter referans sesinden yüksek doğallıkta 48 kHz ses klonlama, `CFG`, `timesteps` ve duygu/oyunculuk yönlendirmesi (`acting_prompt`).
* **Trendyol-TTS (48 kHz):** Türkçe fonetiğe özel açık kaynak ses klonlama seçeneği.
* **Chatterbox Multilingual & Coqui XTTS v2:** Ayarlanabilir duygu şiddeti (`exaggeration`) ve tutarlılık (`cfg_weight`).
* **ElevenLabs API:** Hem **Multilingual v2 (TTS)** (hazır Voice ID veya otomatik anlık IVC klonlama) hem de orijinal tonlama, fısıltı ve bağırmayı birebir aktaran **Speech-to-Speech / Voice Changer (`eleven_multilingual_sts_v2`)**.
* **Yerel Oyunculuk Dönüştürme (Seed-VC STS):** Mikrofondan veya harici ses dosyasından kendi oyunculuğunuzu kaydedip karakterin sesine dönüştürme.

### 2. Kalite Kontrol, Akustik Senkronizasyon ve Yönetmen Araçları
* **Çoklu Aday & Whisper Hakemi:** Her replik için 1–4 arası aday ses üretir; yerel/bulut Whisper ile metin doğruluğunu ve süre uyumunu puanlayarak en iyi adayı otomatik seçer (arayüzden adaylar arası tek tıkla geçiş yapılabilir).
* **Aktif Konuşma LUFS Eşitleme + `-1.0 dBTP` True Peak Sınırlayıcı:** Uzun monologlardaki sessizlik eslerinin ortalama ses yüksekliğini aşağı çekmesini önler; yalnızca **aktif konuşma bölgesinin LUFS/RMS düzeyini** orijinal kayda (vokal ayırma açıksa ayrılmış orijinal vokale) birebir eşitler ve 4× oversampling'li `-1.0 dBTP` sınırlayıcıyla kırpılma/cızırtıyı engeller.
* **TTS Telaffuz Sözlüğü:** Tablodaki Türkçe metin okunaklı haliyle kalırken (`ICA`), yalnızca ses motoruna gönderilen metinde tam kelime eşleşmesiyle telaffuz düzeltmesi uygular (varsayılan kural: `ICA` → `Ay-Si-Ey`, arayüzden düzenlenebilir).
* **Sinematik Zamanlama Kilidi (`>= 15 sn`):** İsteğe bağlı olarak 15 saniye ve üzeri uzun monologlarda orijinal sesteki `>= 0.65 sn` dramatik esleri tespit eder ve Türkçe cümle bloklarını yumuşak geçişle orijinal zamanlamalarına yerleştirir.
* **Doğal Tempo & Akıllı Sessizlik Kırpma:** Hızlandırmadan önce baştaki, sondaki ve cümle aralarındaki gereksiz boşlukları kırpar; varsayılan üst sınır `1.05x` (`1.00x` ile hızlandırma tamamen kapatılabilir).
* **Arka Planı Koru (Vokal Ayırma & Yeniden Miks):** HTDemucs (veya FFmpeg spektral yedek) ile konuşmayı müzik/efektten ayırıp yalnızca vokali dublajlar ve arka planla yeniden miksler.
* **8 Ücretsiz Gemini Modeli & Canlı Kota Paneli:** Hız limiti (429/503) durumunda otomatik model geçişi, günlük 11:00 TRT kota sayacı, proje bazlı tablo taslağı otomatik kaydı ve **"Tümünü Yeniden Dublajla"** modu.

### 3. HITMAN 3 (RPKG) Tam Otomatik Oyun Modülü
* Uygulama içindeki **`🎯 HITMAN 3 (RPKG)`** sekmesi üzerinden oyun klasöründeki `chunk*.rpkg` arşivlerini otomatik tarar; hazır sahne profilleri (**ICA Tesisi / `chunk0`**, **Serbest Çalışan / `chunk29`**) ve diğer tüm chunk'ları listeler.
* **Tek Tıkla Ses Çıkarma:** Aynı sahnede daha önce çıkarılmış/dublajlanmış hash'leri otomatik hariç tutar; karakter/anahtar kelime, süre ve adet filtreleriyle yeni `.WWES` seslerini `vgmstream-cli.exe` ile 48 kHz `.wav` formatına çevirip doğrudan ana stüdyoya yükler.
* **Çoklu Paket Birleştirme & Gerçek `0xFFFF` Wwise Vorbis Kurulum:** Aynı sahneye ait eski ve yeni onaylı dublaj paketlerini otomatik birleştirir (çakışan hash'te en günceli alır), gerçek `0xFFFF` Wwise Vorbis `.WWES` üretip `ww2ogg` + `vgmstream-cli` + FFmpeg üçlü doğrulamasından geçirir, diğer sahnelerin `packagedefinition.txt` ayarlarını koruyarak SHA-256 doğrulamalı kurar.
* **Tek Tıkla Kaldırma:** Seçili sahne yamasını diğer kurulu sahnelere dokunmadan temiz biçimde geri alır.

### 4. %100 Taşınabilirlik ve SHA-256 Güvenliği
* Tüm ayarlar (`./_veri/`), modeller (`./_modeller/`), geçici dosyalar (`./_temp/`) ve SHA-256 yedekleri (`./_yedekler/`) yalnızca proje klasöründe tutulur; `%APPDATA%` veya Windows Kayıt Defteri kullanılmaz.
* İşlem bitiminde veya pencere kapandığında `./_temp/` otomatik temizlenir.

---

## 🛠️ Adım Adım Kurulum

### 1. Gereksinimler
* **İşletim Sistemi:** Windows 10 / 11 (64-bit)
* **Python:** [Python 3.12 (64-bit)](https://www.python.org/downloads/) *(Kurulumda "Add Python to PATH" işaretleyin)*
* **Donanım:**
  * **Bulut / API Modu (Colab Pro veya ElevenLabs):** Ekran kartı gerekmez.
  * **Yerel GPU Modu (Ücretsiz VoxCPM2 / Whisper / HTDemucs / Seed-VC):** NVIDIA CUDA destekli GPU (12 GB+ VRAM önerilir, ör. RTX 4070).

### 2. Depoyu İndirme ve Çekirdek Kurulum
```powershell
git clone https://github.com/BilalHabesSARI/SesAI_Dublaj.git
cd SesAI_Dublaj

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. *(İsteğe Bağlı)* Yerel NVIDIA CUDA GPU Modu Kurulumu
Ücretsiz yerel **VoxCPM2 (48 kHz)**, **Trendyol-TTS**, **faster-whisper**, **HTDemucs** ve **Seed-VC** için proje kökündeki kurulum betiğini çalıştırın:
```powershell
.\yerel_gpu_kur.bat
```
*(Alternatif olarak ekran kartınız yoksa `colab_motoru.ipynb` dosyasını Google Colab'e yükleyip ücretsiz/Pro bulut GPU üzerinden bağlanabilirsiniz.)*

### 4. Harici Araçların (FFmpeg ve HITMAN 3 / Wwise) Yerleştirilmesi
Lisans ve dosya boyutu kuralları gereği harici `.exe`/`.dll` araçları bu depoda barındırılmaz. Aşağıdaki resmi açık kaynak bağlantılardan indirip `./_araclar/` klasörüne yerleştirin:

```text
SesAI_Dublaj/
└── _araclar/
    ├── ffmpeg/
    │   ├── ffmpeg.exe          (Opsiyonel: yoksa imageio-ffmpeg otomatik kullanılır)
    │   └── ffprobe.exe
    ├── rpkg/                   (HITMAN 3 RPKG modülü için)
    │   ├── rpkg-cli.exe
    │   └── hash_list.txt
    └── wwise/                  (HITMAN 3 0xFFFF Wwise Vorbis paketleme/çözme için)
        ├── wav2wem.exe
        ├── vgmstream/
        │   ├── vgmstream-cli.exe
        │   └── (*.dll bağımlılıkları)
        └── ww2ogg/
            ├── ww2ogg.exe
            └── packed_codebooks_aoTuV_603.bin
```

| Araç | Ne İşe Yarar? | Resmi Kaynak |
|---|---|---|
| **FFmpeg** | Ses dönüştürme, aktif konuşma LUFS ve True Peak sınırlama | [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) *(veya `imageio-ffmpeg`)* |
| **RPKG-Tool (`rpkg-cli.exe`)** | HITMAN 3 `.rpkg` çıkarma, oluşturma ve `packagedefinition.txt` işleme | [glacier-modding/RPKG-Tool](https://github.com/glacier-modding/RPKG-Tool/releases) |
| **vgmstream (`vgmstream-cli.exe`)** | `.WWES` / `.wem` seslerini 48 kHz `.wav` çözme ve doğrulama | [vgmstream/vgmstream](https://github.com/vgmstream/vgmstream/releases) |
| **ww2ogg + aoTuV 603** | Üretilen `0xFFFF` Wwise Vorbis akışını gidiş-dönüş doğrulama | [hcs64/ww2ogg](https://github.com/hcs64/ww2ogg/releases) |
| **wav2wem (`wav2wem.exe`)** | Türkçe `.wav` dosyalarını gerçek `0xFFFF` Wwise Vorbis formatına kodlama | [pas2k/wav2wem](https://github.com/pas2k/wav2wem/releases) |

---

## 🚀 Çalıştırma ve Testler

* **Uygulamayı Başlatma:** Proje klasöründeki `baslat.bat` dosyasına çift tıklayın (veya `.\.venv\Scripts\python.exe main.py`).
* **Otomatik Test Paketini Çalıştırma:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest -v tests
  ```

---

## ⚠️ Yasal Uyarı (Legal Disclaimer)

* Bu proje **hiçbir telifli oyun dosyası, ses kaydı, oyun arşivi (`.rpkg`, `.wwes`, `.wem`, `.wav`), harici ikili araç (`.exe`, `.dll`) veya yapay zeka model ağırlığı barındırmaz**.
* **SesAI Dublaj Stüdyosu**, yalnızca kullanıcının **yasal olarak sahip olduğu** oyunlarda **ticari olmayan kişisel kullanım**, erişilebilirlik ve yerel dil denemeleri amacıyla geliştirilmiştir.
* Üçüncü parti oyunların, ses motorlarının ve açık kaynak araçların (IO Interactive / HITMAN, Audiokinetic Wwise, FFmpeg, RPKG-Tool, vgmstream, ww2ogg, VoxCPM2, Chatterbox, Seed-VC, Gemini API, ElevenLabs vb.) tüm ticari marka ve telif hakları kendi hak sahiplerine aittir.
* Oyun dosyalarında yapılan her türlü modifikasyonun ve üçüncü parti API kullanımının sorumluluğu kullanıcıya aittir; çevrimiçi/çok oyunculu koruma sistemlerine sahip oyunlarda kullanmadan önce ilgili oyunun kullanım şartlarını kontrol ediniz.

---

## 📄 Lisans ve Geliştirici

Bu proje [MIT Lisansı](LICENSE) altında açık kaynak olarak sunulmaktadır.

*Geliştirici: **Bilal Habeş SARI** 🇹🇷 — Google Antigravity kullanılarak geliştirilmiştir.*
