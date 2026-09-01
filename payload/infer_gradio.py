# F5-TTS-TW-AUDIO-WORKBENCH: 1.0
# ruff: noqa: E402
# 繁體中文介面版：僅翻譯 Gradio 顯示文字，保留模型名稱與內部邏輯。
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)
from f5_tts.model import DiT, UNetT
from pathlib import Path


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models

vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    elif isinstance(model_cfg, str):
        model_cfg = json.loads(model_cfg)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def chat_model_inference(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def load_text_from_file(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = ""
    return gr.update(value=text)


def split_text_for_sentence_pause(text):
    """Split text at sentence-ending punctuation for fixed pause insertion."""
    # Chinese punctuation can be adjacent to the next sentence; English punctuation
    # is split when followed by whitespace or end-of-text.
    parts = re.split(r"(?<=[。！？])|(?<=[.!?])(?=\s+|$)", text)
    return [part.strip() for part in parts if part and part.strip()]


# English year/date normalization helpers.
# This only rewrites date-like expressions before they are sent to the TTS model.
_SMALL_NUMBERS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_ORDINAL_SMALL = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
}
_ORDINAL_TENS = {
    20: "twentieth",
    30: "thirtieth",
    40: "fortieth",
    50: "fiftieth",
    60: "sixtieth",
    70: "seventieth",
    80: "eightieth",
    90: "ninetieth",
}


def _number_under_100(n):
    if n < 20:
        return _SMALL_NUMBERS[n]
    tens = (n // 10) * 10
    ones = n % 10
    # 不使用連字號：ninety six 比 ninety-six 對部分 TTS 更穩定
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_SMALL_NUMBERS[ones]}"


def _ordinal_under_100(n):
    if n < 20:
        return _ORDINAL_SMALL[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return _ORDINAL_TENS[tens]
    return f"{_TENS[tens]} {_ORDINAL_SMALL[ones]}"


def _year_to_english(year):
    """Convert a 1-4 digit year to a natural English year reading."""
    year = int(year)

    if year < 100:
        return _number_under_100(year)

    if year < 1000:
        first = year // 100
        last_two = year % 100
        if last_two == 0:
            return f"{_SMALL_NUMBERS[first]} hundred"
        return f"{_SMALL_NUMBERS[first]} hundred {_number_under_100(last_two)}"

    if year == 1000:
        return "one thousand"

    if 1001 <= year <= 1999:
        first_two = year // 100
        last_two = year % 100
        if last_two == 0:
            return f"{_number_under_100(first_two)} hundred"
        if last_two < 10:
            return f"{_number_under_100(first_two)} oh {_SMALL_NUMBERS[last_two]}"
        return f"{_number_under_100(first_two)} {_number_under_100(last_two)}"

    if 2000 <= year <= 2009:
        last = year % 100
        if last == 0:
            return "two thousand"
        return f"two thousand {_number_under_100(last)}"

    if 2010 <= year <= 2099:
        last_two = year % 100
        return f"twenty {_number_under_100(last_two)}"

    if 2100 <= year <= 2999:
        first_two = year // 100
        last_two = year % 100
        if last_two == 0:
            return f"{_number_under_100(first_two)} hundred"
        if last_two < 10:
            return f"{_number_under_100(first_two)} oh {_SMALL_NUMBERS[last_two]}"
        return f"{_number_under_100(first_two)} {_number_under_100(last_two)}"

    # Fallback for uncommon four-digit dates outside the normal TTS range.
    thousands = year // 1000
    remainder = year % 1000
    if remainder == 0:
        return f"{_SMALL_NUMBERS.get(thousands, str(thousands))} thousand"
    hundreds = remainder // 100
    last_two = remainder % 100
    words = [f"{_SMALL_NUMBERS.get(thousands, str(thousands))} thousand"]
    if hundreds:
        words.append(f"{_SMALL_NUMBERS[hundreds]} hundred")
    if last_two:
        words.append(_number_under_100(last_two))
    return " ".join(words)


def _era_to_letters(era):
    """Convert era abbreviations to clear letter-by-letter speech."""
    return " ".join(era.upper())


def normalize_english_year_expressions(text):
    """Rewrite date-like English expressions into pronunciation-friendly text.

    Examples:
      496–406 BCE -> four hundred ninety six to four hundred six B C E
      1945 CE      -> nineteen forty-five C E
      1945–1949    -> nineteen forty-five to nineteen forty-nine
      21st century -> twenty-first century
    """
    if not text:
        return text

    # 21st century BCE / 5th century BC, etc.
    century_pattern = re.compile(
        r"(?<!\w)(\d{1,2})(?:st|nd|rd|th)\s+century(?:\s+(BCE|BC|CE|AD))?\b",
        re.IGNORECASE,
    )

    def repl_century(match):
        number = int(match.group(1))
        if not 1 <= number <= 99:
            return match.group(0)
        era = match.group(2)
        result = f"{_ordinal_under_100(number)} century"
        if era:
            result += f" {_era_to_letters(era)}"
        return result

    text = century_pattern.sub(repl_century, text)

    # Ranges carrying one era suffix: 496–406 BCE / 1945-1949 AD.
    range_era_pattern = re.compile(
        r"(?<![\w.])(\d{3,4})\s*[\-‐‑‒–—―−﹘﹣－]\s*(\d{3,4})\s*(BCE|BC|CE|AD)\b",
        re.IGNORECASE,
    )

    def repl_range_era(match):
        start, end, era = match.groups()
        return f"{_year_to_english(start)} to {_year_to_english(end)} {_era_to_letters(era)}"

    text = range_era_pattern.sub(repl_range_era, text)

    # Individual years explicitly marked BCE/BC/CE/AD.
    single_era_pattern = re.compile(r"(?<![\w.])(\d{1,4})\s*(BCE|BC|CE|AD)\b", re.IGNORECASE)

    def repl_single_era(match):
        year, era = match.groups()
        return f"{_year_to_english(year)} {_era_to_letters(era)}"

    text = single_era_pattern.sub(repl_single_era, text)

    # Plain 3-4 digit year ranges such as 1945–1949.
    # This is intentionally limited to ranges so ordinary standalone numbers are unchanged.
    plain_range_pattern = re.compile(r"(?<![\w.])(\d{3,4})\s*[\-‐‑‒–—―−﹘﹣－]\s*(\d{3,4})(?!\w)")

    def repl_plain_range(match):
        start, end = match.groups()
        return f"{_year_to_english(start)} to {_year_to_english(end)}"

    text = plain_range_pattern.sub(repl_plain_range, text)
    return text


@lru_cache(maxsize=1000)  # NOTE. need to ensure params of infer() hashable
@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    seed,
    cross_fade_duration=0.15,
    sentence_pause=0.0,
    normalize_english_years=True,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("請提供參考音訊。")
        return gr.update(), gr.update(), ref_text

    # Set inference seed
    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("隨機種子必須介於 0～2147483647，將改用隨機種子。")
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    if not gen_text.strip():
        gr.Warning("請輸入要產生的文字，或上傳文字檔。")
        return gr.update(), gr.update(), ref_text

    if normalize_english_years:
        normalized_gen_text = normalize_english_year_expressions(gen_text)
        if normalized_gen_text != gen_text:
            show_info("已自動將英文年份／年代轉成較自然的朗讀格式。")
        gen_text = normalized_gen_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("正在載入 E2-TTS 模型……")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, tuple) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("正在載入自訂 TTS 模型……")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    if sentence_pause > 0:
        sentence_parts = split_text_for_sentence_pause(gen_text)
        generated_waves = []
        generated_specs = []
        final_sample_rate = None

        show_info(f"將文字分成 {len(sentence_parts)} 個句子，句尾固定停頓 {sentence_pause:.2f} 秒。")

        for index, sentence in enumerate(sentence_parts):
            sentence_wave, sentence_sr, sentence_spec = infer_process(
                ref_audio,
                ref_text,
                sentence,
                ema_model,
                vocoder,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                speed=speed,
                show_info=show_info,
                progress=None,
            )
            if sentence_wave is None:
                continue

            final_sample_rate = sentence_sr
            generated_waves.append(sentence_wave)
            if sentence_spec is not None:
                generated_specs.append(sentence_spec)

            # Insert real digital silence after each sentence except the last one.
            if index < len(sentence_parts) - 1:
                pause_samples = int(sentence_pause * sentence_sr)
                if pause_samples > 0:
                    generated_waves.append(np.zeros(pause_samples, dtype=sentence_wave.dtype))

        if not generated_waves or final_sample_rate is None:
            gr.Warning("沒有成功產生語音。")
            return gr.update(), gr.update(), ref_text

        final_wave = np.concatenate(generated_waves)
        combined_spectrogram = (
            np.concatenate(generated_specs, axis=1) if generated_specs else np.zeros((100, 1), dtype=np.float32)
        )
    else:
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=show_info,
            progress=gr.Progress(),
        )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        finally:
            os.unlink(temp_path)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
    save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed


# =========================
# 常用隨機種子收藏
# =========================
_FAVORITE_SEEDS_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_FAVORITE_SEEDS_FILE = os.path.join(_FAVORITE_SEEDS_DIR, "favorite_seeds.json")


def _load_favorite_seeds():
    """讀取已收藏的隨機種子。"""
    try:
        with open(_FAVORITE_SEEDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []

        result = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                seed = int(item.get("seed"))
            except (TypeError, ValueError):
                continue
            if not 0 <= seed <= 2**31 - 1:
                continue

            name = str(item.get("name", "")).strip()
            result.append({"seed": seed, "name": name})
        return result
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_favorite_seeds(items):
    """將收藏清單寫入本機 JSON，重新開程式後仍會保留。"""
    os.makedirs(_FAVORITE_SEEDS_DIR, exist_ok=True)
    with open(_FAVORITE_SEEDS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _favorite_seed_label(seed, name=""):
    name = (name or "").strip()
    return f"{seed} ｜ {name}" if name else str(seed)


def _favorite_seed_choices():
    return [_favorite_seed_label(item["seed"], item.get("name", "")) for item in _load_favorite_seeds()]


def _seed_from_favorite_label(label):
    if label is None or str(label).strip() == "":
        return None
    try:
        return int(str(label).split("｜", 1)[0].strip())
    except (TypeError, ValueError):
        return None


def save_favorite_seed(seed_value, seed_name):
    """收藏目前畫面上的 seed；相同 seed 再存一次時更新名稱，不重複新增。"""
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        gr.Warning("目前的隨機種子不是有效數字。")
        return gr.update()

    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("隨機種子必須介於 0～2147483647。")
        return gr.update()

    name = (seed_name or "").strip()
    items = _load_favorite_seeds()

    existing = None
    for item in items:
        if item["seed"] == seed:
            existing = item
            break

    if existing is not None:
        # 沒填新名稱時，保留舊名稱。
        if name:
            existing["name"] = name
        display_name = existing.get("name", "")
        gr.Info(f"已更新常用種子：{seed}")
    else:
        items.append({"seed": seed, "name": name})
        display_name = name
        gr.Info(f"已儲存常用種子：{seed}")

    _save_favorite_seeds(items)
    return gr.update(
        choices=_favorite_seed_choices(),
        value=_favorite_seed_label(seed, display_name),
    )


def use_favorite_seed(selected_label):
    """把收藏的 seed 帶回指定種子欄，並自動關閉『每次使用隨機種子』。"""
    seed = _seed_from_favorite_label(selected_label)
    if seed is None:
        gr.Warning("請先選擇一個已儲存的種子。")
        return gr.update(), gr.update()

    gr.Info(f"已套用常用種子：{seed}")
    return seed, False


def delete_favorite_seed(selected_label):
    """刪除選取的收藏 seed。"""
    seed = _seed_from_favorite_label(selected_label)
    if seed is None:
        gr.Warning("請先選擇要刪除的種子。")
        return gr.update()

    items = _load_favorite_seeds()
    new_items = [item for item in items if item["seed"] != seed]

    if len(new_items) == len(items):
        gr.Warning("找不到這個已儲存的種子。")
        return gr.update(choices=_favorite_seed_choices(), value=None)

    _save_favorite_seeds(new_items)
    gr.Info(f"已刪除常用種子：{seed}")
    return gr.update(choices=_favorite_seed_choices(), value=None)


# ============================================================
# 音訊整理工具：長音檔裁切、基礎聲音優化、Whisper 語音轉文字
# ============================================================

# 輸出位置固定放在目前 F5-TTS 安裝根目錄的「輸出檔案」。
# 標準安裝時：
# C:\AI\F5-TTS-Universal\F5-TTS-src\src\f5_tts\infer\infer_gradio.py
# -> C:\AI\F5-TTS-Universal\輸出檔案
def _audio_tool_find_install_root():
    try:
        here = Path(__file__).resolve()
        # 往上尋找同時含 F5-TTS-src 或 .venv 的安裝根目錄。
        for parent in here.parents:
            if (parent / "F5-TTS-src").exists() or (
                (parent / ".venv").exists() and (parent / "runtime").exists()
            ):
                return parent
        # 標準 source tree 的保底位置。
        if len(here.parents) >= 5:
            return here.parents[4]
    except Exception:
        pass
    return Path(os.getcwd()).resolve().parent


_AUDIO_TOOL_INSTALL_ROOT = _audio_tool_find_install_root()
_AUDIO_TOOL_OUTPUT_DIR = str(_AUDIO_TOOL_INSTALL_ROOT / "輸出檔案")
os.makedirs(_AUDIO_TOOL_OUTPUT_DIR, exist_ok=True)

# 清理舊版曾直接建立在「輸出檔案」根目錄下的空白資料夾。
# 現在正確結構是：
# 輸出檔案\日期時間_原始名稱\音訊 / 文字 / 合成
for _legacy_name in ("音訊", "文字", "合成"):
    try:
        _legacy_dir = Path(_AUDIO_TOOL_OUTPUT_DIR) / _legacy_name
        if _legacy_dir.is_dir() and not any(_legacy_dir.iterdir()):
            _legacy_dir.rmdir()
    except Exception:
        pass

# 同一次上傳的原始音訊，共用同一個「時間資料夾」，
# 並將音訊與文字分開存放。
# 例如：
# 輸出檔案\2026-09-01_23-40-12_原始檔名\
#   音訊\
#     001_優化.wav
#     002_裁切.wav
#   文字\
#     001.txt
#     002.txt
_AUDIO_TOOL_SESSION_MAP = {}


def _audio_tool_safe_stem(audio_path):
    """
    取得簡潔的原始錄音名稱。
    例如：
    林薇英文聲音英文.MP3_main_vocal.wav
    -> 林薇英文聲音英文
    """
    if not audio_path:
        return "錄音"

    stem = Path(str(audio_path)).stem.strip() or "錄音"

    # 移除檔名中殘留的副檔名字樣，例如 .MP3_main_vocal
    stem = re.sub(
        r"(?i)\.(mp3|wav|m4a|flac|ogg|aac|wma)(?=$|[_\-\s])",
        "",
        stem,
    )

    # 移除常見人聲分離 / 處理尾碼
    stem = re.sub(
        r"(?i)(?:[_\-\s]*(?:main[_\-\s]*vocal|lead[_\-\s]*vocal|"
        r"vocals?|voice|main[_\-\s]*voice|clean|cleaned|denoise[d]?|"
        r"enhanced|enhance|processed|output))+$",
        "",
        stem,
    )

    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-")
    stem = re.sub(r"[_\-]+$", "", stem)

    return (stem or "錄音")[:28]


def _audio_tool_session_label(session_dir):
    """從時間資料夾名稱取出簡化後的原始錄音名稱。"""
    try:
        name = Path(str(session_dir)).name
        m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+?)(?:_\d+)?$", name)
        if m:
            return _audio_tool_safe_stem(m.group(1))
        return _audio_tool_safe_stem(name)
    except Exception:
        return "錄音"


def _audio_tool_source_key(audio_path):
    if not audio_path:
        return None
    try:
        return str(Path(str(audio_path)).resolve())
    except Exception:
        return str(audio_path)


def _audio_tool_get_session_dir(audio_path, create=True):
    """
    每次上傳一個原始音訊，就建立一個時間分類資料夾。
    同一個原始音訊在本次程式執行期間會持續使用同一資料夾。
    """
    key = _audio_tool_source_key(audio_path)
    if not key:
        return None

    known = _AUDIO_TOOL_SESSION_MAP.get(key)
    if known and Path(known).is_dir():
        return known

    if not create:
        return None

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = _audio_tool_safe_stem(audio_path)
    base = Path(_AUDIO_TOOL_OUTPUT_DIR) / f"{stamp}_{stem}"
    folder = base
    suffix = 2
    while folder.exists():
        folder = Path(f"{base}_{suffix}")
        suffix += 1

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "音訊").mkdir(parents=True, exist_ok=True)
    (folder / "文字").mkdir(parents=True, exist_ok=True)
    (folder / "合成").mkdir(parents=True, exist_ok=True)
    _AUDIO_TOOL_SESSION_MAP[key] = str(folder)
    return str(folder)


def _audio_tool_index_from_path(path):
    if not path:
        return None

    name = Path(str(path)).name
    m = re.search(r"(?:^|_)(\d{3,})(?=_|\.|$)", name)
    return int(m.group(1)) if m else None


def _audio_tool_next_index(session_dir):
    """找目前時間資料夾下一個段落編號：001、002、003……"""
    if not session_dir:
        return 1

    highest = 0
    try:
        base = Path(session_dir)
        for subfolder in (base / "音訊", base / "文字"):
            if not subfolder.exists():
                continue
            for item in subfolder.iterdir():
                if not item.is_file():
                    continue
                idx = _audio_tool_index_from_path(item)
                if idx:
                    highest = max(highest, idx)
    except Exception:
        pass

    return highest + 1


def _audio_tool_audio_path(audio_path, kind):
    session_dir = Path(_audio_tool_get_session_dir(audio_path, create=True))
    idx = _audio_tool_next_index(str(session_dir))
    audio_dir = session_dir / "音訊"
    audio_dir.mkdir(parents=True, exist_ok=True)

    short_name = _audio_tool_safe_stem(audio_path)
    filename = f"{short_name}_{idx:03d}_{kind}.wav"
    return idx, str(audio_dir / filename)


def _audio_tool_text_path_for_audio(original_audio, audio_path):
    """
    音訊與文字分開資料夾，但使用相同「原始簡稱 + 段落編號」。
    """
    if audio_path:
        p = Path(str(audio_path))
        idx = _audio_tool_index_from_path(p)

        try:
            output_root = Path(_AUDIO_TOOL_OUTPUT_DIR).resolve()
            resolved = p.resolve()

            if idx and output_root in resolved.parents:
                session_dir = resolved.parent.parent if resolved.parent.name == "音訊" else resolved.parent
                text_dir = session_dir / "文字"
                text_dir.mkdir(parents=True, exist_ok=True)
                short_name = _audio_tool_session_label(session_dir)
                return str(text_dir / f"{short_name}_{idx:03d}.txt")
        except Exception:
            pass

    # 辨識完整原音：建立同編號原音與文字
    session_dir = Path(_audio_tool_get_session_dir(original_audio, create=True))
    idx = _audio_tool_next_index(str(session_dir))
    short_name = _audio_tool_safe_stem(original_audio)

    audio_dir = session_dir / "音訊"
    text_dir = session_dir / "文字"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    src = Path(str(original_audio))
    ext = src.suffix.lower() if src.suffix else ".wav"
    paired_audio = audio_dir / f"{short_name}_{idx:03d}_原音{ext}"

    try:
        shutil.copy2(src, paired_audio)
    except Exception:
        audio = _audio_tool_load_segment(original_audio)
        paired_audio = audio_dir / f"{short_name}_{idx:03d}_原音.wav"
        audio.export(str(paired_audio), format="wav")

    return str(text_dir / f"{short_name}_{idx:03d}.txt")


def _audio_tool_session_from_reference(ref_audio_path):
    """
    決定 TTS 合成結果要放到哪個時間資料夾。

    優先順序：
    1. 參考音訊本身就在「輸出檔案\時間資料夾\音訊」內
       → 直接沿用該時間資料夾。
    2. Gradio 若把參考音訊複製到暫存資料夾
       → 依「檔名」回頭搜尋素材庫中的原始音訊，
         找到後仍沿用原本的時間資料夾。
    3. 真的是使用者自行從外部上傳的新參考音訊
       → 才建立新的時間資料夾。
    """
    if ref_audio_path:
        try:
            p = Path(str(ref_audio_path)).resolve()
            output_root = Path(_AUDIO_TOOL_OUTPUT_DIR).resolve()

            # 情況 1：檔案本來就在工作台輸出資料夾裡。
            if output_root in p.parents:
                if p.parent.name in {"音訊", "文字", "合成"}:
                    session_dir = p.parent.parent
                else:
                    session_dir = p.parent

                if session_dir.parent == output_root:
                    (session_dir / "音訊").mkdir(parents=True, exist_ok=True)
                    (session_dir / "文字").mkdir(parents=True, exist_ok=True)
                    (session_dir / "合成").mkdir(parents=True, exist_ok=True)
                    return str(session_dir)

            # 情況 2：Gradio 可能把檔案複製到暫存路徑。
            # 這時檔名通常仍保留原本名稱，例如：
            # 林薇英文聲音英文_001_優化.wav
            target_name = p.name

            matched = []
            if output_root.is_dir():
                for session_dir in output_root.iterdir():
                    if not session_dir.is_dir():
                        continue

                    candidate = session_dir / "音訊" / target_name
                    if candidate.is_file():
                        matched.append(candidate)

            # 若同名檔案出現多次，使用最近修改的那一份。
            if matched:
                matched.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                session_dir = matched[0].parent.parent
                (session_dir / "合成").mkdir(parents=True, exist_ok=True)
                return str(session_dir)

            # 再用「原始簡稱 + 段落編號」做一次保底搜尋，
            # 可處理 Gradio 改動了一點檔名的情況。
            ref_index = _audio_tool_index_from_path(p)
            ref_stem = _audio_tool_safe_stem(p)

            if output_root.is_dir() and ref_index:
                candidates = []
                for session_dir in output_root.iterdir():
                    audio_dir = session_dir / "音訊"
                    if not audio_dir.is_dir():
                        continue

                    for item in audio_dir.iterdir():
                        if not item.is_file():
                            continue
                        if _audio_tool_index_from_path(item) != ref_index:
                            continue

                        item_stem = _audio_tool_safe_stem(item)
                        # 任一方包含另一方即可視為同來源，避免 _優化 / _裁切 尾碼影響。
                        if (
                            ref_stem in item_stem
                            or item_stem in ref_stem
                            or _audio_tool_session_label(session_dir) in ref_stem
                        ):
                            candidates.append(item)

                if candidates:
                    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    session_dir = candidates[0].parent.parent
                    (session_dir / "合成").mkdir(parents=True, exist_ok=True)
                    return str(session_dir)

        except Exception:
            pass

    # 情況 3：不是素材庫內的音訊，才建立新的時間資料夾。
    return _audio_tool_get_session_dir(ref_audio_path, create=True)


def _audio_tool_next_synth_index(session_dir):
    """合成檔另外獨立編號 001、002、003……"""
    synth_dir = Path(session_dir) / "合成"
    synth_dir.mkdir(parents=True, exist_ok=True)

    highest = 0
    for item in synth_dir.iterdir():
        if not item.is_file():
            continue
        m = re.match(r"^(\d{3,})_", item.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def _audio_tool_save_synthesized(ref_audio_path, audio_tuple, label="合成"):
    """
    將 TTS 合成結果自動存到：
    時間資料夾\合成\001_合成.wav
    """
    if not audio_tuple:
        return None

    try:
        sample_rate, wave = audio_tuple
        wave_np = np.asarray(wave)

        if wave_np.ndim > 1:
            wave_np = np.squeeze(wave_np)

        session_dir = _audio_tool_session_from_reference(ref_audio_path)
        if not session_dir:
            return None

        synth_dir = Path(session_dir) / "合成"
        synth_dir.mkdir(parents=True, exist_ok=True)

        idx = _audio_tool_next_synth_index(session_dir)
        safe_label = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(label)).strip() or "合成"
        short_name = _audio_tool_session_label(session_dir)
        out_path = synth_dir / f"{short_name}_{idx:03d}_{safe_label}.wav"

        sf.write(str(out_path), wave_np, int(sample_rate))
        return str(out_path)

    except Exception as e:
        gr.Warning(f"合成音訊自動存檔失敗：{e}")
        return None


def audio_tool_open_synth_folder(ref_audio_path):
    """直接開啟目前參考音訊對應的合成資料夾。"""
    try:
        session_dir = _audio_tool_session_from_reference(ref_audio_path)
        if not session_dir:
            return "目前沒有可開啟的合成資料夾。"

        synth_dir = Path(session_dir) / "合成"
        synth_dir.mkdir(parents=True, exist_ok=True)

        if os.name == "nt":
            os.startfile(str(synth_dir))

        return f"已開啟：{synth_dir}"
    except Exception as e:
        gr.Warning(f"無法開啟合成資料夾：{e}")
        return f"無法開啟合成資料夾：{e}"



def _audio_tool_relative_path(path):
    """素材庫顯示：時間資料夾\檔名。"""
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(_AUDIO_TOOL_OUTPUT_DIR).resolve()))
    except Exception:
        return Path(str(path)).name


def audio_tool_open_output_folder():
    """開啟輸出資料夾，讓使用者能快速找到檔案。"""
    try:
        os.makedirs(_AUDIO_TOOL_OUTPUT_DIR, exist_ok=True)
        if os.name == "nt":
            os.startfile(_AUDIO_TOOL_OUTPUT_DIR)
        else:
            gr.Info(f"輸出位置：{_AUDIO_TOOL_OUTPUT_DIR}")
        return f"已開啟輸出資料夾：{_AUDIO_TOOL_OUTPUT_DIR}"
    except Exception as e:
        gr.Warning(f"無法開啟輸出資料夾：{e}")
        return f"輸出位置：{_AUDIO_TOOL_OUTPUT_DIR}"


_AUDIO_TOOL_WHISPER_PIPELINE = None


def _audio_tool_load_segment(audio_path):
    from pydub import AudioSegment

    return AudioSegment.from_file(audio_path)


def _audio_tool_duration(audio_path):
    if not audio_path:
        return 0.0

    audio = _audio_tool_load_segment(audio_path)
    return len(audio) / 1000.0


def audio_tool_inspect(audio_path):
    """讀取音訊長度，並同步更新裁切範圍。"""
    if not audio_path:
        return (
            "尚未上傳音訊。",
            0.0,
            gr.update(minimum=0, maximum=60, value=0),
            gr.update(minimum=0, maximum=60, value=12),
        )

    try:
        audio = _audio_tool_load_segment(audio_path)
        duration = len(audio) / 1000.0
        channels = audio.channels
        sample_rate = audio.frame_rate
        end_default = min(duration, 12.0)
        slider_max = max(1.0, duration)

        session_dir = _audio_tool_get_session_dir(audio_path, create=True)
        info = (
            f"音訊長度：{duration:.2f} 秒｜"
            f"聲道：{channels}｜取樣率：{sample_rate} Hz"
            f"｜本次資料夾：{Path(session_dir).name if session_dir else '未建立'}"
        )

        if duration > 12:
            info += "｜建議裁成 5～12 秒作為 F5-TTS 參考音訊。"
        else:
            info += "｜長度適合直接作為參考音訊。"

        return (
            info,
            duration,
            gr.update(minimum=0, maximum=slider_max, value=0),
            gr.update(minimum=0, maximum=slider_max, value=end_default),
        )
    except Exception as e:
        gr.Warning(f"無法讀取音訊：{e}")
        return (
            f"讀取失敗：{e}",
            0.0,
            gr.update(),
            gr.update(),
        )


def _audio_tool_silence_threshold(audio):
    """依音檔平均音量估算自然停頓門檻。"""
    try:
        dbfs = float(audio.dBFS)
        if not np.isfinite(dbfs):
            return -45.0
        return max(-50.0, min(-28.0, dbfs - 24.0))
    except Exception:
        return -45.0


def _audio_tool_detect_speech(audio, min_silence_len=320):
    """回傳非靜音人聲區段，單位毫秒。"""
    from pydub.silence import detect_nonsilent

    if len(audio) <= 0:
        return []

    # 轉成單聲道、16k 只用於分析，時間軸不會改變。
    analysis = audio.set_channels(1).set_frame_rate(16000)
    threshold = _audio_tool_silence_threshold(analysis)

    return detect_nonsilent(
        analysis,
        min_silence_len=min_silence_len,
        silence_thresh=threshold,
        seek_step=15,
    )


def _audio_tool_pad_range(start_ms, end_ms, total_ms, pad_ms=120):
    start_ms = max(0, int(start_ms) - pad_ms)
    end_ms = min(int(total_ms), int(end_ms) + pad_ms)
    return start_ms, end_ms


def audio_tool_find_natural_phrase(audio_path, anchor_sec):
    """
    依自然停頓尋找靠近指定位置的完整語句。
    目標優先落在 5～12 秒；如果附近沒有足夠停頓，會保守取接近 12 秒。
    """
    if not audio_path:
        gr.Warning("請先上傳原始音訊。")
        return gr.update(), gr.update(), "請先上傳音訊。"

    try:
        audio = _audio_tool_load_segment(audio_path)
        total_ms = len(audio)
        if total_ms <= 0:
            return gr.update(), gr.update(), "音訊內容為空。"

        anchor_ms = int(max(0.0, min(float(anchor_sec or 0), total_ms / 1000.0)) * 1000)

        # 只分析定位點前後 45 秒，長音檔會快很多。
        window_start = max(0, anchor_ms - 45000)
        window_end = min(total_ms, anchor_ms + 45000)
        window = audio[window_start:window_end]
        segments = _audio_tool_detect_speech(window, min_silence_len=320)

        if not segments:
            start_ms = max(0, anchor_ms - 1000)
            end_ms = min(total_ms, start_ms + 12000)
            return (
                start_ms / 1000.0,
                end_ms / 1000.0,
                "附近沒有偵測到明顯自然停頓，已先取接近 12 秒的範圍，可再手動微調。",
            )

        # 轉回全音檔時間。
        segments = [(s + window_start, e + window_start) for s, e in segments]

        def distance_to_segment(seg):
            s, e = seg
            if s <= anchor_ms <= e:
                return 0
            return min(abs(anchor_ms - s), abs(anchor_ms - e))

        idx = min(range(len(segments)), key=lambda i: distance_to_segment(segments[i]))
        left = right = idx

        start_ms, end_ms = segments[idx]
        span = end_ms - start_ms

        # 太短就把相鄰語句一起納入，直到接近 5～12 秒。
        while span < 5000 and (left > 0 or right < len(segments) - 1):
            candidates = []

            if left > 0:
                s = segments[left - 1][0]
                e = end_ms
                duration = e - s
                if duration <= 12500:
                    gap = max(0, start_ms - segments[left - 1][1])
                    candidates.append(("L", duration, gap, s, e))

            if right < len(segments) - 1:
                s = start_ms
                e = segments[right + 1][1]
                duration = e - s
                if duration <= 12500:
                    gap = max(0, segments[right + 1][0] - end_ms)
                    candidates.append(("R", duration, gap, s, e))

            if not candidates:
                break

            # 優先選停頓較短、總長度較接近 8 秒的一側。
            choice = min(
                candidates,
                key=lambda x: (x[2], abs(x[1] - 8000)),
            )
            if choice[0] == "L":
                left -= 1
            else:
                right += 1
            start_ms, end_ms = choice[3], choice[4]
            span = end_ms - start_ms

        start_ms, end_ms = _audio_tool_pad_range(start_ms, end_ms, total_ms, 120)
        span = end_ms - start_ms

        warning = ""
        if span > 12500:
            # 單一連續語句本身太長，避免拿超長參考音訊。
            center = max(start_ms + 6000, min(anchor_ms, end_ms - 6000))
            start_ms = max(0, center - 6000)
            end_ms = min(total_ms, start_ms + 12000)
            start_ms = max(0, end_ms - 12000)
            warning = "附近沒有可在 12 秒內完整包住的停頓，因此已取 12 秒附近片段；建議播放確認。"
        elif span < 3500:
            warning = "找到的自然語句較短，可改用「自動挑最佳 5～12 秒」或手動加長。"
        else:
            warning = "已依自然停頓找到一段完整語句，可直接播放確認或再手動微調。"

        return start_ms / 1000.0, end_ms / 1000.0, warning

    except Exception as e:
        gr.Warning(f"自然停頓分析失敗：{e}")
        return gr.update(), gr.update(), f"自然停頓分析失敗：{e}"


def audio_tool_find_best_reference(audio_path):
    """
    從整個音檔找一段適合作為 F5-TTS 參考音訊的 5～12 秒片段。
    評分重點：自然停頓、連續人聲比例、片段長度、音量與削波。
    """
    if not audio_path:
        gr.Warning("請先上傳原始音訊。")
        return gr.update(), gr.update(), "請先上傳音訊。"

    try:
        audio = _audio_tool_load_segment(audio_path)
        total_ms = len(audio)
        if total_ms <= 0:
            return gr.update(), gr.update(), "音訊內容為空。"

        segments = _audio_tool_detect_speech(audio, min_silence_len=320)

        if not segments:
            end_ms = min(total_ms, 12000)
            return 0.0, end_ms / 1000.0, "沒有偵測到明顯人聲區段，已先選前 12 秒。"

        # 防止極長音檔造成候選過多；仍保留整段取樣。
        if len(segments) > 4000:
            step = max(1, len(segments) // 4000)
            sampled_segments = segments[::step]
        else:
            sampled_segments = segments

        prelim = []
        n = len(sampled_segments)

        for i in range(n):
            speech_ms = 0
            start_ms = sampled_segments[i][0]

            for j in range(i, min(n, i + 20)):
                s, e = sampled_segments[j]
                if j > i and s - sampled_segments[j - 1][1] > 2200:
                    break

                speech_ms += max(0, e - s)
                end_ms = e
                span = end_ms - start_ms

                if span > 12500:
                    break
                if span < 4500:
                    continue

                speech_ratio = speech_ms / max(span, 1)
                duration_score = max(0.0, 1.0 - abs(span - 8000) / 5000.0)
                ratio_score = max(0.0, 1.0 - abs(speech_ratio - 0.88) / 0.45)
                prelim_score = duration_score * 4.0 + ratio_score * 5.0

                prelim.append(
                    (prelim_score, start_ms, end_ms, speech_ratio)
                )

        if not prelim:
            # 找不到 5 秒以上候選時，從最長人聲附近取最多 12 秒。
            s, e = max(segments, key=lambda x: x[1] - x[0])
            center = (s + e) // 2
            start_ms = max(0, center - 6000)
            end_ms = min(total_ms, start_ms + 12000)
            start_ms = max(0, end_ms - 12000)
            return (
                start_ms / 1000.0,
                end_ms / 1000.0,
                "沒有找到理想的 5～12 秒自然片段，已選最長人聲附近 12 秒，請播放確認。",
            )

        # 只對前 60 名做較重的音量分析。
        prelim.sort(reverse=True, key=lambda x: x[0])
        best = None

        for prelim_score, start_ms, end_ms, speech_ratio in prelim[:60]:
            ps, pe = _audio_tool_pad_range(start_ms, end_ms, total_ms, 120)
            clip = audio[ps:pe]

            dbfs = clip.dBFS
            max_dbfs = clip.max_dBFS

            loudness_score = 0.0
            if np.isfinite(dbfs):
                # 約 -22 ~ -12 dBFS 都屬好用範圍，中心抓 -17。
                loudness_score = max(0.0, 1.0 - abs(float(dbfs) + 17.0) / 18.0)

            clipping_penalty = 2.0 if np.isfinite(max_dbfs) and max_dbfs > -0.15 else 0.0
            final_score = prelim_score + loudness_score * 2.0 - clipping_penalty

            candidate = (final_score, ps, pe, speech_ratio, dbfs, max_dbfs)
            if best is None or candidate[0] > best[0]:
                best = candidate

        _, start_ms, end_ms, speech_ratio, dbfs, max_dbfs = best
        duration = (end_ms - start_ms) / 1000.0

        detail = (
            f"已自動挑選：{duration:.2f} 秒"
            f"｜人聲比例約 {speech_ratio * 100:.0f}%"
        )
        if np.isfinite(dbfs):
            detail += f"｜平均音量 {dbfs:.1f} dBFS"
        detail += "｜已盡量讓頭尾落在自然停頓。請播放一次確認人聲與背景是否乾淨。"

        return start_ms / 1000.0, end_ms / 1000.0, detail

    except Exception as e:
        gr.Warning(f"自動挑選失敗：{e}")
        return gr.update(), gr.update(), f"自動挑選失敗：{e}"



def audio_tool_take_12_seconds(start_sec, duration_sec):
    """從目前起點向後快速取 12 秒。"""
    try:
        start = max(0.0, float(start_sec or 0))
        duration = max(0.0, float(duration_sec or 0))
        end = min(duration, start + 12.0)
        if end <= start:
            end = min(duration, 12.0)
            start = 0.0
        return start, end
    except Exception:
        return 0.0, min(float(duration_sec or 12.0), 12.0)



def _audio_tool_build_all_ranges(audio):
    """
    將整個長音檔依自然停頓切成全部段落。
    目標：
    - 優先在自然停頓切開
    - 一般段落盡量 5～12 秒
    - 不漏掉任何偵測到的人聲
    - 單一句超過 12 秒時，保底每 12 秒切一段
    回傳 [(start_ms, end_ms), ...]
    """
    total_ms = len(audio)
    if total_ms <= 0:
        return []

    speech = _audio_tool_detect_speech(audio, min_silence_len=320)

    # 完全沒有偵測到人聲：仍將整個檔案每 12 秒切開，確保全部輸出。
    if not speech:
        ranges = []
        start = 0
        while start < total_ms:
            end = min(total_ms, start + 12000)
            if end - start >= 250:
                ranges.append((start, end))
            start = end
        return ranges

    # 先把超過 12 秒的單一連續人聲拆開。
    expanded = []
    for s, e in speech:
        if e <= s:
            continue
        cur = s
        while e - cur > 12000:
            expanded.append((cur, cur + 12000))
            cur += 12000
        if e - cur >= 250:
            expanded.append((cur, e))

    if not expanded:
        return []

    # 再把太短的相鄰語句合併；加入中間停頓後仍不得超過 12 秒。
    ranges = []
    cur_s, cur_e = expanded[0]

    for s, e in expanded[1:]:
        proposed_span = e - cur_s
        current_span = cur_e - cur_s
        gap = max(0, s - cur_e)

        # 短段優先往後合併；一般最多 12 秒。
        if proposed_span <= 12000 and (
            current_span < 5000
            or gap <= 700
            or proposed_span <= 8500
        ):
            cur_e = e
        else:
            ranges.append(_audio_tool_pad_range(cur_s, cur_e, total_ms, 100))
            cur_s, cur_e = s, e

    ranges.append(_audio_tool_pad_range(cur_s, cur_e, total_ms, 100))

    # 合併後如果仍有 >12 秒，再做最後保底切分。
    final_ranges = []
    for s, e in ranges:
        cur = s
        while e - cur > 12000:
            final_ranges.append((cur, cur + 12000))
            cur += 12000
        if e - cur >= 250:
            final_ranges.append((cur, e))

    return final_ranges


def _audio_tool_optimize_segment(
    segment,
    trim_edge_silence=True,
    normalize_volume=True,
    highpass_voice=True,
    convert_f5_format=True,
):
    """對單一批次段落做與『裁切並優化』相同的保守處理。"""
    from pydub import effects
    from pydub.silence import detect_nonsilent

    audio = segment

    if trim_edge_silence and len(audio) > 0:
        base_dbfs = audio.dBFS
        if np.isfinite(base_dbfs):
            silence_thresh = max(-50.0, base_dbfs - 28.0)
            nonsilent = detect_nonsilent(
                audio,
                min_silence_len=180,
                silence_thresh=silence_thresh,
                seek_step=10,
            )
            if nonsilent:
                pad_ms = 120
                first = max(0, nonsilent[0][0] - pad_ms)
                last = min(len(audio), nonsilent[-1][1] + pad_ms)
                audio = audio[first:last]

    if highpass_voice and len(audio) > 0:
        audio = audio.high_pass_filter(70)

    if normalize_volume and len(audio) > 0 and np.isfinite(audio.dBFS):
        audio = effects.normalize(audio, headroom=1.0)

    if convert_f5_format:
        audio = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)

    if len(audio) >= 80:
        audio = audio.fade_in(20).fade_out(40)

    return audio


def audio_tool_batch_split_all(
    audio_path,
    do_optimize,
    trim_edge_silence=True,
    normalize_volume=True,
    highpass_voice=True,
    convert_f5_format=True,
):
    """
    將整個原始音檔全部切完並一次輸出所有段落。
    不只挑一個最佳片段。
    """
    if not audio_path:
        gr.Warning("請先上傳原始音訊。")
        return None, None, "請先上傳音訊。"

    try:
        audio = _audio_tool_load_segment(audio_path)
        ranges = _audio_tool_build_all_ranges(audio)

        if not ranges:
            gr.Warning("沒有找到可輸出的音訊段落。")
            return None, None, "沒有找到可輸出的音訊段落。"

        created = []
        first_path = None

        for start_ms, end_ms in ranges:
            segment = audio[start_ms:end_ms]

            if do_optimize:
                segment = _audio_tool_optimize_segment(
                    segment,
                    trim_edge_silence=trim_edge_silence,
                    normalize_volume=normalize_volume,
                    highpass_voice=highpass_voice,
                    convert_f5_format=convert_f5_format,
                )
                kind = "優化"
            else:
                kind = "裁切"

            if len(segment) < 150:
                continue

            idx, out_path = _audio_tool_audio_path(audio_path, kind)
            segment.export(out_path, format="wav")
            created.append(out_path)

            if first_path is None:
                first_path = out_path

        if not created:
            return None, None, "沒有成功輸出任何段落。"

        session_dir = Path(created[0]).parent.parent
        mode_name = "全部分段＋優化" if do_optimize else "全部分段裁切"

        status = (
            f"{mode_name}完成：共輸出 {len(created)} 段。"
            f"｜第一段：{Path(created[0]).name}"
            f"｜最後一段：{Path(created[-1]).name}"
            f"｜資料夾：{session_dir / '音訊'}"
        )

        return first_path, created[0], status

    except Exception as e:
        gr.Warning(f"全部自動分段失敗：{e}")
        return None, None, f"全部自動分段失敗：{e}"


def audio_tool_ui_batch_trim_all(audio_path):
    audio_out, file_out, status = audio_tool_batch_split_all(
        audio_path,
        False,
    )
    return (
        audio_out,
        file_out,
        status,
        _audio_tool_library_audio_update(audio_out),
        _audio_tool_library_summary(),
    )


def audio_tool_ui_batch_optimize_all(
    audio_path,
    trim_edge_silence,
    normalize_volume,
    highpass_voice,
    convert_f5_format,
):
    audio_out, file_out, status = audio_tool_batch_split_all(
        audio_path,
        True,
        trim_edge_silence,
        normalize_volume,
        highpass_voice,
        convert_f5_format,
    )
    return (
        audio_out,
        file_out,
        status,
        _audio_tool_library_audio_update(audio_out),
        _audio_tool_library_summary(),
    )


@gpu_decorator
def audio_tool_transcribe_all_segments(
    original_audio,
    language_choice,
    compute_mode,
):
    """
    將目前這次上傳所產生的所有裁切音訊逐段轉成文字，
    文字與音訊保持相同編號。
    """
    if not original_audio:
        gr.Warning("請先上傳原始音訊。")
        return "", None, "請先上傳原始音訊。", gr.update(), _audio_tool_library_summary()

    try:
        session_dir = _audio_tool_get_session_dir(original_audio, create=False)
        if not session_dir:
            return "", None, "目前尚未建立分段資料夾。", gr.update(), _audio_tool_library_summary()

        audio_dir = Path(session_dir) / "音訊"
        if not audio_dir.is_dir():
            return "", None, "目前沒有已裁切的音訊。", gr.update(), _audio_tool_library_summary()

        audio_files = [
            p for p in audio_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        ]
        audio_files.sort(key=lambda p: (_audio_tool_index_from_path(p) or 999999, p.name))

        if not audio_files:
            return "", None, "目前沒有已裁切的音訊。", gr.update(), _audio_tool_library_summary()

        pipe = _audio_tool_get_whisper_pipeline(compute_mode)
        lang_map = {
            "自動辨識": None,
            "中文": "chinese",
            "英文": "english",
        }
        language = lang_map.get(language_choice)

        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        completed = []
        latest_text = ""
        latest_txt_path = None

        for audio_file in audio_files:
            result = pipe(
                str(audio_file),
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )
            transcript = (result.get("text") or "").strip()
            if not transcript:
                continue

            txt_path = _audio_tool_text_path_for_audio(
                original_audio,
                str(audio_file),
            )
            Path(txt_path).write_text(transcript, encoding="utf-8")

            completed.append((audio_file, Path(txt_path)))
            latest_text = transcript
            latest_txt_path = txt_path

        if not completed:
            return "", None, "Whisper 沒有成功辨識任何段落。", gr.update(), _audio_tool_library_summary()

        status = (
            f"全部段落文字辨識完成：共 {len(completed)} 份。"
            f"｜文字資料夾：{Path(completed[0][1]).parent}"
        )

        return (
            latest_text,
            latest_txt_path,
            status,
            _audio_tool_library_text_update(latest_txt_path),
            _audio_tool_library_summary(),
        )

    except Exception as e:
        gr.Warning(f"全部段落轉文字失敗：{e}")
        return "", None, f"全部段落轉文字失敗：{e}", gr.update(), _audio_tool_library_summary()



def audio_tool_trim_only(audio_path, start_sec, end_sec):
    """
    只依照指定時間裁切，不做：
    - 頭尾靜音裁除
    - 音量正規化
    - 高通濾波
    - 聲道/取樣率/位元深度轉換

    為了讓輸出容易使用，仍統一另存成 WAV，
    但保留裁切後音訊本身的 channels / frame_rate / sample_width。
    """
    if not audio_path:
        gr.Warning("請先上傳原始音訊。")
        return None, None, "請先上傳音訊。"

    try:
        audio = _audio_tool_load_segment(audio_path)
        duration = len(audio) / 1000.0

        start = max(0.0, float(start_sec or 0))
        end = min(duration, float(end_sec or duration))

        if end <= start:
            gr.Warning("結束時間必須大於開始時間。")
            return None, None, "裁切範圍不正確。"

        trimmed = audio[int(start * 1000): int(end * 1000)]

        segment_index, out_path = _audio_tool_audio_path(audio_path, "裁切")
        trimmed.export(out_path, format="wav")

        processed_duration = len(trimmed) / 1000.0
        status = (
            f"只裁切完成：{processed_duration:.2f} 秒"
            f"｜未做音量正規化"
            f"｜未做低頻過濾"
            f"｜未裁頭尾靜音"
            f"｜保留原聲道 / 取樣率 / 位元深度"
            f"｜段落編號：{segment_index:03d}｜已儲存：{out_path}"
        )

        return out_path, out_path, status

    except Exception as e:
        gr.Warning(f"裁切失敗：{e}")
        return None, None, f"裁切失敗：{e}"


def audio_tool_optimize(
    audio_path,
    start_sec,
    end_sec,
    trim_edge_silence=True,
    normalize_volume=True,
    highpass_voice=True,
    convert_f5_format=True,
):
    """
    裁切並做保守型聲音整理。
    不做重度 AI 降噪，避免改變原始聲線。
    """
    if not audio_path:
        gr.Warning("請先上傳原始音訊。")
        return None, None, "請先上傳音訊。"

    try:
        from pydub import effects
        from pydub.silence import detect_nonsilent

        audio = _audio_tool_load_segment(audio_path)
        duration = len(audio) / 1000.0

        start = max(0.0, float(start_sec or 0))
        end = min(duration, float(end_sec or duration))

        if end <= start:
            gr.Warning("結束時間必須大於開始時間。")
            return None, None, "裁切範圍不正確。"

        audio = audio[int(start * 1000) : int(end * 1000)]

        # 只裁掉頭尾的長靜音，不動中間自然停頓。
        if trim_edge_silence and len(audio) > 0:
            base_dbfs = audio.dBFS
            if np.isfinite(base_dbfs):
                silence_thresh = max(-50.0, base_dbfs - 28.0)
                nonsilent = detect_nonsilent(
                    audio,
                    min_silence_len=180,
                    silence_thresh=silence_thresh,
                    seek_step=10,
                )
                if nonsilent:
                    pad_ms = 120
                    first = max(0, nonsilent[0][0] - pad_ms)
                    last = min(len(audio), nonsilent[-1][1] + pad_ms)
                    audio = audio[first:last]

        # 去掉很低頻的隆隆聲/桌面震動；70 Hz 對一般人聲相對保守。
        if highpass_voice and len(audio) > 0:
            audio = audio.high_pass_filter(70)

        # 正規化峰值，避免聲音太小或爆音；保留 1 dB headroom。
        if normalize_volume and len(audio) > 0 and np.isfinite(audio.dBFS):
            audio = effects.normalize(audio, headroom=1.0)

        # F5-TTS 參考音訊常用格式：mono / 24 kHz / 16-bit WAV。
        if convert_f5_format:
            audio = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)

        # 防止切點出現喀聲。
        if len(audio) >= 80:
            audio = audio.fade_in(20).fade_out(40)

        segment_index, out_path = _audio_tool_audio_path(audio_path, "優化")
        audio.export(out_path, format="wav")

        processed_duration = len(audio) / 1000.0
        status = (
            f"完成：{processed_duration:.2f} 秒"
            f"｜{'已裁頭尾靜音' if trim_edge_silence else '保留頭尾靜音'}"
            f"｜{'音量已正規化' if normalize_volume else '未調整音量'}"
            f"｜{'70 Hz 高通' if highpass_voice else '未做高通'}"
            f"｜{'Mono / 24 kHz WAV' if convert_f5_format else '保留原格式參數'}"
        )

        if processed_duration > 12.0:
            status += "｜注意：作為 F5-TTS 參考音訊仍建議控制在 12 秒內。"

        status += f"｜段落編號：{segment_index:03d}｜已儲存：{out_path}"
        return out_path, out_path, status

    except Exception as e:
        gr.Warning(f"音訊處理失敗：{e}")
        return None, None, f"音訊處理失敗：{e}"


def _audio_tool_get_whisper_pipeline(compute_mode="自動"):
    """延遲載入 Whisper，避免啟動 F5-TTS 時就佔用額外記憶體。"""
    global _AUDIO_TOOL_WHISPER_PIPELINE

    if _AUDIO_TOOL_WHISPER_PIPELINE is not None:
        return _AUDIO_TOOL_WHISPER_PIPELINE

    from transformers import pipeline

    mode = compute_mode or "自動"
    use_gpu = False

    if torch.cuda.is_available():
        if mode == "GPU":
            use_gpu = True
        elif mode == "自動":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # F5-TTS 模型本身已佔顯存，8GB 以上再預設讓 Whisper 也用 GPU。
                use_gpu = vram_gb >= 8.0
            except Exception:
                use_gpu = False

    if use_gpu:
        device = 0
        dtype = torch.float16
        gr.Info("Whisper 使用 GPU。")
    else:
        device = -1
        dtype = torch.float32
        if torch.cuda.is_available() and mode == "自動":
            gr.Info("顯示卡記憶體較小，Whisper 自動改用 CPU，避免與 F5-TTS 搶顯存。")
        else:
            gr.Info("Whisper 使用 CPU。")

    _AUDIO_TOOL_WHISPER_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        torch_dtype=dtype,
        chunk_length_s=30,
    )
    return _AUDIO_TOOL_WHISPER_PIPELINE


@gpu_decorator
def audio_tool_transcribe(
    original_audio,
    processed_audio,
    transcript_scope,
    language_choice,
    compute_mode,
):
    """將原音或裁切後音訊轉成文字，並輸出 txt。"""
    if transcript_scope == "完整原音檔":
        audio_path = original_audio
    else:
        audio_path = processed_audio or original_audio

    if not audio_path:
        gr.Warning("請先上傳音訊。")
        return "", None, "請先上傳音訊。"

    try:
        pipe = _audio_tool_get_whisper_pipeline(compute_mode)

        lang_map = {
            "自動辨識": None,
            "中文": "chinese",
            "英文": "english",
        }
        language = lang_map.get(language_choice)

        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        gr.Info("Whisper 正在辨識，長音檔需要較多時間。")
        result = pipe(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )

        transcript = (result.get("text") or "").strip()
        if not transcript:
            gr.Warning("Whisper 沒有辨識到文字。")
            return "", None, "沒有辨識到文字。"

        txt_path = _audio_tool_text_path_for_audio(original_audio, audio_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        status = f"辨識完成，共 {len(transcript)} 個字元。｜已儲存：{txt_path}"
        return transcript, txt_path, status

    except Exception as e:
        gr.Warning(f"Whisper 辨識失敗：{e}")
        return "", None, f"Whisper 辨識失敗：{e}"


def audio_tool_save_transcript(
    original_audio,
    processed_audio,
    current_text_file,
    transcript,
):
    """儲存修改後文字，優先保留與音訊相同的段落編號。"""
    transcript = (transcript or "").strip()
    if not transcript:
        gr.Warning("目前沒有文字可以儲存。")
        return None, "目前沒有文字可以儲存。"

    try:
        txt_path = None

        # 已經有這段的文字檔時，直接更新同一份，編號不變。
        if current_text_file:
            try:
                candidate = Path(str(current_text_file)).resolve()
                output_root = Path(_AUDIO_TOOL_OUTPUT_DIR).resolve()
                if candidate.is_file() and (
                    candidate.parent == output_root or output_root in candidate.parent.parents
                ):
                    txt_path = str(candidate)
            except Exception:
                txt_path = None

        # 若尚無文字檔，依目前處理後音訊建立相同編號文字。
        if not txt_path:
            txt_path = _audio_tool_text_path_for_audio(
                original_audio,
                processed_audio or original_audio,
            )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        return txt_path, f"已儲存修改後文字：{txt_path}"
    except Exception as e:
        gr.Warning(f"文字儲存失敗：{e}")
        return None, f"文字儲存失敗：{e}"



def audio_tool_apply_to_basic_tts(processed_audio, transcript):
    """把整理完成的音訊與文字送到基本 TTS。"""
    if not processed_audio:
        gr.Warning("請先完成「裁切＋優化」。")
        return gr.update(), gr.update()

    gr.Info("已套用到「基本 TTS」的參考音訊與參考文字。")
    return processed_audio, transcript or ""




# ============================================================
# 段落素材庫：顯示全部已輸出的音訊與文字，並可自由配對套用
# ============================================================

def _audio_tool_library_files(kind):
    """掃描所有時間資料夾中的「音訊」或「文字」子資料夾。"""
    try:
        root = Path(_AUDIO_TOOL_OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
        files = []

        target_folder_name = "音訊" if kind == "audio" else "文字"

        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue

            target_dir = session_dir / target_folder_name
            if not target_dir.is_dir():
                continue

            for item in target_dir.iterdir():
                if not item.is_file():
                    continue

                if kind == "audio" and item.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                    files.append(item)
                elif kind == "text" and item.suffix.lower() == ".txt":
                    files.append(item)

        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files
    except Exception:
        return []


def _audio_tool_library_audio_files():
    return _audio_tool_library_files("audio")


def _audio_tool_library_text_files():
    return _audio_tool_library_files("text")


def _audio_tool_library_name(path):
    return _audio_tool_relative_path(path) if path else None


def _audio_tool_library_resolve(relative_name):
    if not relative_name:
        return None
    try:
        root = Path(_AUDIO_TOOL_OUTPUT_DIR).resolve()
        path = (root / str(relative_name)).resolve()
        if path.is_file() and (path.parent == root or root in path.parent.parents):
            return str(path)
    except Exception:
        pass
    return None


def _audio_tool_library_resolve_audio(name):
    path = _audio_tool_library_resolve(name)
    if path and Path(path).suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        return path
    return None


def _audio_tool_library_resolve_text(name):
    path = _audio_tool_library_resolve(name)
    if path and Path(path).suffix.lower() == ".txt":
        return path
    return None


def _audio_tool_library_summary():
    audio_files = _audio_tool_library_audio_files()
    text_files = _audio_tool_library_text_files()

    sessions = {}

    for p in audio_files:
        session_name = p.parent.parent.name
        sessions.setdefault(session_name, {"audio": [], "text": []})["audio"].append(p)

    for p in text_files:
        session_name = p.parent.parent.name
        sessions.setdefault(session_name, {"audio": [], "text": []})["text"].append(p)

    lines = [
        f"**目前素材：音訊 {len(audio_files)} 段｜文字 {len(text_files)} 份｜時間資料夾 {len(sessions)} 個**",
        "",
    ]

    for session_name in sorted(sessions.keys(), reverse=True):
        data = sessions[session_name]
        lines.append(f"### 📁 {session_name}")

        by_index = {}

        for p in data["audio"]:
            idx = _audio_tool_index_from_path(p)
            by_index.setdefault(idx or 0, {})["audio"] = p.name

        for p in data["text"]:
            idx = _audio_tool_index_from_path(p)
            by_index.setdefault(idx or 0, {})["text"] = p.name

        for idx in sorted(by_index.keys()):
            pair = by_index[idx]
            number = f"{idx:03d}" if idx else "---"
            audio_name = pair.get("audio", "尚無音訊")
            text_name = pair.get("text", "尚無文字")
            lines.append(
                f"- **{number}**｜🎧 `音訊\\{audio_name}`｜📝 `文字\\{text_name}`"
            )

        lines.append("")

    if not sessions:
        lines.append("尚無素材。上傳音訊後會自動建立時間資料夾。")

    return "\n".join(lines)


def _audio_tool_library_audio_choices():
    return [_audio_tool_relative_path(p) for p in _audio_tool_library_audio_files()]


def _audio_tool_library_text_choices():
    return [_audio_tool_relative_path(p) for p in _audio_tool_library_text_files()]


def audio_tool_refresh_library(current_audio=None, current_text=None):
    """重新掃描輸出資料夾，保留目前選擇（若檔案仍存在）。"""
    audio_choices = _audio_tool_library_audio_choices()
    text_choices = _audio_tool_library_text_choices()

    audio_value = current_audio if current_audio in audio_choices else None
    text_value = current_text if current_text in text_choices else None

    return (
        gr.update(choices=audio_choices, value=audio_value),
        gr.update(choices=text_choices, value=text_value),
        _audio_tool_library_summary(),
    )


def audio_tool_library_preview_audio(name):
    path = _audio_tool_library_resolve_audio(name)
    if not path:
        return None
    return path


def audio_tool_library_preview_text(name):
    path = _audio_tool_library_resolve_text(name)
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except Exception:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def audio_tool_apply_library_pair(audio_name, text_name):
    """
    音訊與文字可自由配對。
    若文字未選，參考文字留空，讓基本 TTS 依原本流程自動辨識。
    """
    audio_path = _audio_tool_library_resolve_audio(audio_name)
    text_path = _audio_tool_library_resolve_text(text_name)

    if not audio_path:
        gr.Warning("請先從段落素材庫選一段音訊。")
        return gr.update(), gr.update(), "請先選擇一段音訊。"

    transcript = ""
    if text_path:
        try:
            transcript = Path(text_path).read_text(encoding="utf-8-sig").strip()
        except Exception:
            transcript = Path(text_path).read_text(encoding="utf-8", errors="replace").strip()

    audio_label = Path(audio_path).name
    text_label = Path(text_path).name if text_path else "未選文字（基本 TTS 可自動辨識）"

    gr.Info("已把所選段落套用到「基本 TTS」。")
    return (
        audio_path,
        transcript,
        f"已套用：音訊 `{audio_label}`｜文字 `{text_label}`",
    )


def _audio_tool_library_audio_update(preferred_path=None):
    choices = _audio_tool_library_audio_choices()
    preferred_name = _audio_tool_library_name(preferred_path)
    value = preferred_name if preferred_name in choices else None
    return gr.update(choices=choices, value=value)


def _audio_tool_library_text_update(preferred_path=None):
    choices = _audio_tool_library_text_choices()
    preferred_name = _audio_tool_library_name(preferred_path)
    value = preferred_name if preferred_name in choices else None
    return gr.update(choices=choices, value=value)


def audio_tool_ui_transcribe(
    original_audio,
    processed_audio,
    transcript_scope,
    language_choice,
    compute_mode,
):
    transcript, txt_file, status = audio_tool_transcribe(
        original_audio,
        processed_audio,
        transcript_scope,
        language_choice,
        compute_mode,
    )
    return (
        transcript,
        txt_file,
        status,
        _audio_tool_library_text_update(txt_file),
        _audio_tool_library_summary(),
    )


def audio_tool_ui_save_transcript(
    original_audio,
    processed_audio,
    current_text_file,
    transcript,
):
    txt_file, status = audio_tool_save_transcript(
        original_audio,
        processed_audio,
        current_text_file,
        transcript,
    )
    return (
        txt_file,
        status,
        _audio_tool_library_text_update(txt_file),
        _audio_tool_library_summary(),
    )



# ============================================================
# 音訊工作台 UI 狀態：按下哪個選項，就讓該按鈕變成主色
# ============================================================

def _audio_tool_selection_buttons(selected):
    labels = [
        "⭐ 自動挑最佳 5～12 秒",
        "🎯 從開始位置找完整語句",
        "⏱ 從開始位置取 12 秒",
    ]
    return tuple(
        gr.Button(
            value=label,
            variant="primary" if i == selected else "secondary",
        )
        for i, label in enumerate(labels)
    )


def _audio_tool_process_buttons(selected):
    labels = [
        "✂ 只裁切（不優化）",
        "✨ 裁切並優化",
    ]
    return tuple(
        gr.Button(
            value=label,
            variant="primary" if i == selected else "secondary",
        )
        for i, label in enumerate(labels)
    )


def audio_tool_ui_auto_best(audio_path):
    start, end, status = audio_tool_find_best_reference(audio_path)
    return (
        start,
        end,
        status,
        *_audio_tool_selection_buttons(0),
    )


def audio_tool_ui_auto_phrase(audio_path, start_sec):
    start, end, status = audio_tool_find_natural_phrase(audio_path, start_sec)
    return (
        start,
        end,
        status,
        *_audio_tool_selection_buttons(1),
    )


def audio_tool_ui_quick_12(start_sec, duration_sec):
    start, end = audio_tool_take_12_seconds(start_sec, duration_sec)
    status = f"已從 {start:.2f} 秒開始取 12 秒範圍；可播放確認後再裁切。"
    return (
        start,
        end,
        status,
        *_audio_tool_selection_buttons(2),
    )


def audio_tool_ui_trim_only(audio_path, start_sec, end_sec):
    audio_out, file_out, status = audio_tool_trim_only(
        audio_path,
        start_sec,
        end_sec,
    )
    return (
        audio_out,
        file_out,
        status,
        *_audio_tool_process_buttons(0),
        _audio_tool_library_audio_update(audio_out),
        _audio_tool_library_summary(),
    )


def audio_tool_ui_optimize(
    audio_path,
    start_sec,
    end_sec,
    trim_edge_silence,
    normalize_volume,
    highpass_voice,
    convert_f5_format,
):
    audio_out, file_out, status = audio_tool_optimize(
        audio_path,
        start_sec,
        end_sec,
        trim_edge_silence,
        normalize_volume,
        highpass_voice,
        convert_f5_format,
    )
    return (
        audio_out,
        file_out,
        status,
        *_audio_tool_process_buttons(1),
        _audio_tool_library_audio_update(audio_out),
        _audio_tool_library_summary(),
    )




def audio_tool_inspect_auto_only(audio_path):
    """自動分段版只需要顯示音訊資訊與總長度。"""
    result = audio_tool_inspect(audio_path)
    return result[0], result[1]


with gr.Blocks() as app_audio_tools:
    gr.Markdown("""
# 🎧 音訊工作台

長音檔直接在這裡完成：**整個音檔全部分段 → 批次轉文字 → 挑選素材 → 套用到 F5-TTS**

> 不論 38 秒、5 分鐘、1 小時或更長，都會把整個音檔全部切完，不再只挑單一片段。
""")

    audio_tool_duration_state = gr.State(0.0)

    gr.Markdown(
        "**自動存檔方式：**每次上傳原始音訊會建立一個新的「日期時間資料夾」，"
        "裡面分成「音訊」、「文字」、「合成」三個資料夾；"
        "音訊與文字使用相同段落編號。"
    )

    with gr.Row():
        audio_output_path = gr.Textbox(
            label="📁 輸出根目錄",
            value=_AUDIO_TOOL_OUTPUT_DIR,
            interactive=False,
            scale=5,
        )
        open_output_folder_btn = gr.Button(
            "📂 開啟資料夾",
            variant="secondary",
            scale=1,
        )
    output_folder_status = gr.Markdown()

    # ---------- 1. 上傳 ----------
    with gr.Group():
        gr.Markdown("## ① 上傳原始音訊")
        audio_tool_input = gr.Audio(
            label="原始音訊（可上傳長音檔）",
            type="filepath",
        )
        audio_tool_info = gr.Markdown("尚未上傳音訊。")

    # ---------- 2. 全部自動分段 ----------
    with gr.Group():
        gr.Markdown("## ② 整個音檔全部自動分段")
        gr.Markdown(
            "會從頭到尾全部處理，優先依自然停頓切分；"
            "一般每段盡量控制在 **5～12 秒**。"
        )

        with gr.Row():
            batch_trim_all_btn = gr.Button(
                "📦 全部自動分段（只裁切）",
                variant="secondary",
            )
            batch_optimize_all_btn = gr.Button(
                "📦 全部自動分段＋優化",
                variant="primary",
            )

        with gr.Accordion("優化內容（一般使用不用改）", open=False):
            with gr.Row():
                audio_trim_silence = gr.Checkbox(
                    label="整理頭尾長靜音",
                    value=True,
                )
                audio_normalize = gr.Checkbox(
                    label="音量正規化",
                    value=True,
                )
                audio_highpass = gr.Checkbox(
                    label="過濾低頻隆隆聲",
                    value=True,
                )
                audio_f5_format = gr.Checkbox(
                    label="轉 F5 建議格式（Mono / 24 kHz / 16-bit WAV）",
                    value=True,
                )
            gr.Markdown(
                "採用保守型優化，不做重度 AI 降噪，避免破壞原本聲線。"
            )

        audio_tool_status = gr.Markdown()

        with gr.Row():
            audio_tool_output = gr.Audio(
                label="第一段預覽（全部段落都已存到資料夾）",
                type="filepath",
                scale=4,
            )
            audio_tool_download = gr.File(
                label="下載目前預覽 WAV",
                scale=1,
            )

    # ---------- 3. 全部分段轉文字 ----------
    with gr.Group():
        gr.Markdown("## ③ 全部分段提取文字")

        with gr.Row():
            transcript_language = gr.Radio(
                choices=["自動辨識", "中文", "英文"],
                value="自動辨識",
                label="語言",
            )

        with gr.Accordion("Whisper 運算設定（一般使用不用改）", open=False):
            transcript_compute = gr.Dropdown(
                choices=["自動", "CPU", "GPU"],
                value="自動",
                label="Whisper 運算",
                info="自動模式會依顯存狀況選擇，顯存較小時優先 CPU，避免與 F5-TTS 搶顯存。",
            )

        transcribe_all_btn = gr.Button(
            "📝 全部已裁切段落轉文字",
            variant="primary",
        )
        transcript_status = gr.Markdown()

        transcript_output = gr.Textbox(
            label="最近完成的文字（可直接修改）",
            lines=8,
            max_lines=30,
            placeholder="全部辨識完成後，最近一段文字會顯示在這裡。",
        )

        with gr.Row():
            transcript_download = gr.File(
                label="下載目前文字 TXT",
                scale=4,
            )
            save_transcript_btn = gr.Button(
                "💾 儲存目前修改文字",
                variant="secondary",
                scale=1,
            )

    # ---------- 4. 段落素材庫 ----------
    with gr.Group():
        gr.Markdown("## ④ 段落素材庫")
        gr.Markdown(
            "所有自動分段音訊與 Whisper 文字都會保留。"
            "**同一段音訊與文字使用相同編號，也可以自由交叉配對。**"
        )

        refresh_library_btn = gr.Button(
            "🔄 重新整理全部素材",
            variant="secondary",
        )

        with gr.Row():
            with gr.Column():
                audio_library_dropdown = gr.Dropdown(
                    choices=_audio_tool_library_audio_choices(),
                    value=None,
                    label="🎧 選擇音訊段落",
                )
                audio_library_preview = gr.Audio(
                    label="音訊預覽",
                    type="filepath",
                )

            with gr.Column():
                text_library_dropdown = gr.Dropdown(
                    choices=_audio_tool_library_text_choices(),
                    value=None,
                    label="📝 選擇文字段落",
                )
                text_library_preview = gr.Textbox(
                    label="文字預覽",
                    lines=6,
                    max_lines=20,
                )

        apply_library_pair_btn = gr.Button(
            "➡ 套用所選音訊＋文字到「基本 TTS」",
            variant="primary",
        )
        library_apply_status = gr.Markdown()

        with gr.Accordion("📚 查看全部已產生檔案", open=False):
            library_all_files = gr.Markdown(_audio_tool_library_summary())

    # ---------- Events ----------
    refresh_library_btn.click(
        audio_tool_refresh_library,
        inputs=[
            audio_library_dropdown,
            text_library_dropdown,
        ],
        outputs=[
            audio_library_dropdown,
            text_library_dropdown,
            library_all_files,
        ],
    )

    audio_library_dropdown.change(
        audio_tool_library_preview_audio,
        inputs=[audio_library_dropdown],
        outputs=[audio_library_preview],
    )

    text_library_dropdown.change(
        audio_tool_library_preview_text,
        inputs=[text_library_dropdown],
        outputs=[text_library_preview],
    )

    open_output_folder_btn.click(
        audio_tool_open_output_folder,
        outputs=[output_folder_status],
    )

    audio_tool_input.change(
        audio_tool_inspect_auto_only,
        inputs=[audio_tool_input],
        outputs=[
            audio_tool_info,
            audio_tool_duration_state,
        ],
    )

    batch_trim_all_btn.click(
        audio_tool_ui_batch_trim_all,
        inputs=[audio_tool_input],
        outputs=[
            audio_tool_output,
            audio_tool_download,
            audio_tool_status,
            audio_library_dropdown,
            library_all_files,
        ],
    )

    batch_optimize_all_btn.click(
        audio_tool_ui_batch_optimize_all,
        inputs=[
            audio_tool_input,
            audio_trim_silence,
            audio_normalize,
            audio_highpass,
            audio_f5_format,
        ],
        outputs=[
            audio_tool_output,
            audio_tool_download,
            audio_tool_status,
            audio_library_dropdown,
            library_all_files,
        ],
    )

    transcribe_all_btn.click(
        audio_tool_transcribe_all_segments,
        inputs=[
            audio_tool_input,
            transcript_language,
            transcript_compute,
        ],
        outputs=[
            transcript_output,
            transcript_download,
            transcript_status,
            text_library_dropdown,
            library_all_files,
        ],
    )



with gr.Blocks() as app_tts:
    gr.Markdown("# 批量文字轉語音")
    gr.Markdown(
        "**參考來源有兩種：**  \n"
        "① 從「音訊工作台」挑選段落後一鍵套用。  \n"
        "② 直接在本頁自行上傳參考音訊、輸入參考文字。  \n"
        "本頁自行選擇的內容會直接取代先前從工作台套用的內容。"
    )
    ref_audio_input = gr.Audio(
        label="參考音訊（工作台套用／自行上傳皆可）",
        type="filepath",
    )
    with gr.Row():
        gen_text_input = gr.Textbox(
            label="要產生的文字",
            lines=10,
            max_lines=40,
            scale=4,
        )
        gen_text_file = gr.File(label="從文字檔載入要產生的文字（.txt）", file_types=[".txt"], scale=1)
    generate_btn = gr.Button("合成語音", variant="primary")
    with gr.Accordion("進階設定", open=True) as adv_settn:
        with gr.Row():
            ref_text_input = gr.Textbox(
                label="參考文字（工作台套用／自行輸入皆可）",
                info="留空時會自動辨識參考音訊內容；若自行輸入文字或上傳文字檔，將以您提供的文字為準。",
                lines=2,
                scale=4,
            )
            ref_text_file = gr.File(label="從文字檔載入參考文字（.txt）", file_types=[".txt"], scale=1)
        with gr.Row():
            randomize_seed = gr.Checkbox(
                label="每次使用隨機種子",
                info="勾選後每次生成都使用不同的隨機種子；取消勾選則使用右側指定的種子。",
                value=True,
                scale=3,
            )
            seed_input = gr.Number(
                label="目前／指定種子",
                value=0,
                precision=0,
                scale=2,
                info="生成完成後，這裡會顯示本次實際使用的種子。",
            )
            with gr.Column(scale=4):
                remove_silence = gr.Checkbox(
                    label="移除過長靜音",
                    info="若生成結果出現不需要的長時間靜音，可開啟此選項自動偵測並裁切。",
                    value=False,
                )

        with gr.Accordion("⭐ 常用隨機種子", open=False):
            gr.Markdown("聽到效果好的聲音後，可以把目前種子收藏起來；資料會保存在本機，下次開 F5-TTS 還在。")
            with gr.Row():
                favorite_seed_dropdown = gr.Dropdown(
                    choices=_favorite_seed_choices(),
                    label="已儲存的種子",
                    value=None,
                    allow_custom_value=False,
                    scale=4,
                )
                favorite_seed_name = gr.Textbox(
                    label="備註名稱（可留空）",
                    placeholder="例如：自然美式、旁白穩定、女生自然",
                    scale=3,
                )
            with gr.Row():
                save_favorite_seed_btn = gr.Button("＋ 儲存目前種子", variant="secondary")
                use_favorite_seed_btn = gr.Button("✓ 使用選取種子", variant="primary")
                delete_favorite_seed_btn = gr.Button("－ 刪除選取種子", variant="stop")
        speed_slider = gr.Slider(
            label="語速",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="調整生成語音的速度。",
        )
        nfe_slider = gr.Slider(
            label="NFE 採樣步數",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="設定去噪／生成的步數；數值越高通常越慢。",
        )
        cross_fade_duration_slider = gr.Slider(
            label="交叉淡化時間（秒）",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="設定不同語音片段銜接時的交叉淡化時間。",
        )
        sentence_pause_slider = gr.Slider(
            label="句尾停頓（秒）",
            minimum=0.0,
            maximum=1.5,
            value=0.35,
            step=0.05,
            info="遇到句號、問號、驚嘆號（. ! ? 。！？）時，分句生成並插入固定靜音。設為 0 可關閉。",
        )
        normalize_english_years = gr.Checkbox(
            label="英文年份／年代自動轉讀",
            value=True,
            info="會自動辨識各種 -／–／—／− 等年份範圍符號與年代大小寫。例如 496–406 bce 會轉成 four hundred ninety six to four hundred six B C E。",
        )

    def collapse_accordion():
        return gr.Accordion(open=False)

    # Workaround for https://github.com/SWivid/F5-TTS/issues/1239#issuecomment-3677987413
    # i.e. to set gr.Accordion(open=True) by default, then collapse manually Blocks loaded
    app_tts.load(
        fn=collapse_accordion,
        inputs=None,
        outputs=adv_settn,
    )

    audio_output = gr.Audio(label="合成語音")
    with gr.Row():
        synthesized_download = gr.File(
            label="下載合成完成 WAV",
            scale=4,
        )
        open_synth_folder_btn = gr.Button(
            "📂 開啟合成資料夾",
            variant="secondary",
            scale=1,
        )
    synthesized_status = gr.Markdown()
    spectrogram_output = gr.Image(label="頻譜圖")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        randomize_seed,
        seed_input,
        cross_fade_duration_slider,
        sentence_pause_slider,
        normalize_english_years,
        nfe_slider,
        speed_slider,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        audio_out, spectrogram_path, ref_text_out, used_seed = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            seed=seed_input,
            cross_fade_duration=cross_fade_duration_slider,
            sentence_pause=sentence_pause_slider,
            normalize_english_years=normalize_english_years,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        synth_path = _audio_tool_save_synthesized(
            ref_audio_input,
            audio_out,
            label="合成",
        )

        if synth_path:
            synth_status = f"合成完成並已自動儲存：{synth_path}"
        else:
            synth_status = "合成完成，但自動存檔未成功。"

        return (
            audio_out,
            synth_path,
            synth_status,
            spectrogram_path,
            ref_text_out,
            used_seed,
        )

    # 常用種子：新增／套用／刪除
    save_favorite_seed_btn.click(
        save_favorite_seed,
        inputs=[seed_input, favorite_seed_name],
        outputs=[favorite_seed_dropdown],
    )

    use_favorite_seed_btn.click(
        use_favorite_seed,
        inputs=[favorite_seed_dropdown],
        outputs=[seed_input, randomize_seed],
    )

    delete_favorite_seed_btn.click(
        delete_favorite_seed,
        inputs=[favorite_seed_dropdown],
        outputs=[favorite_seed_dropdown],
    )

    gen_text_file.upload(
        load_text_from_file,
        inputs=[gen_text_file],
        outputs=[gen_text_input],
    )

    ref_text_file.upload(
        load_text_from_file,
        inputs=[ref_text_file],
        outputs=[ref_text_input],
    )

    ref_audio_input.clear(
        lambda: [None, None],
        None,
        [ref_text_input, ref_text_file],
    )

    # 音訊工作台素材庫 → 基本 TTS（音訊與文字可自由配對）
    apply_library_pair_btn.click(
        audio_tool_apply_library_pair,
        inputs=[
            audio_library_dropdown,
            text_library_dropdown,
        ],
        outputs=[
            ref_audio_input,
            ref_text_input,
            library_apply_status,
        ],
    )

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            randomize_seed,
            seed_input,
            cross_fade_duration_slider,
            sentence_pause_slider,
            normalize_english_years,
            nfe_slider,
            speed_slider,
        ],
        outputs=[
            audio_output,
            synthesized_download,
            synthesized_status,
            spectrogram_output,
            ref_text_input,
            seed_input,
        ],
    )

    open_synth_folder_btn.click(
        audio_tool_open_synth_folder,
        inputs=[ref_audio_input],
        outputs=[synthesized_status],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {str} or {"name": str, "seed": int, "speed": float}
    pattern = r"(\{.*?\})"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_type_dict = {
        "name": "Regular",
        "seed": -1,
        "speed": 1.0,
    }

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                current_type_dict["text"] = text
                segments.append(current_type_dict)
        else:
            # This is type
            type_str = tokens[i].strip()
            try:  # if type dict
                current_type_dict = json.loads(type_str)
            except json.decoder.JSONDecodeError:
                type_str = type_str[1:-1]  # remove brace {}
                current_type_dict = {"name": type_str, "seed": -1, "speed": 1.0}

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # 多角色／多風格語音生成

    這裡可以產生多種說話風格，或讓不同角色使用不同的參考聲音。請依照下方格式輸入文字，或上傳相同格式的 .txt 檔案。系統會依標籤選用對應的語音類型；若沒有指定，會使用一般語氣。某個語音類型會持續套用，直到下一個語音類型標籤出現為止。
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **範例 1：使用簡單標籤** <br>
            {Regular} 你好，這是一段一般語氣。 <br>
            {Surprised} 真的嗎？我完全沒想到！ <br>
            {Sad} 我今天真的有點難過…… <br>
            {Angry} 這件事真的讓我很生氣！ <br>
            {Whisper} 我現在用小聲的方式說話。 <br>
            {Shouting} 為什麼會這樣？！
            """
        )

        gr.Markdown(
            """
            **範例 2：指定角色、種子與語速** <br>
            {"name": "Speaker1_Happy", "seed": -1, "speed": 1} 大家好，很高興見到你們。 <br>
            {"name": "Speaker2_Regular", "seed": -1, "speed": 1} 歡迎來到今天的節目。 <br>
            {"name": "Speaker1_Sad", "seed": -1, "speed": 1} 我今天的心情有一點低落。 <br>
            {"name": "Speaker2_Whisper", "seed": -1, "speed": 1} 接下來我會小聲告訴你。
            """
        )

    gr.Markdown(
        '請為每個語音類型上傳不同的參考音訊。第一個語音類型為必填；若需要更多角色或風格，可按「新增語音類型」。'
    )

    # Regular speech type (mandatory)
    with gr.Row(variant="compact") as regular_row:
        with gr.Column(scale=1, min_width=160):
            regular_name = gr.Textbox(value="Regular", label="語音類型名稱")
            regular_insert = gr.Button("插入標籤", variant="secondary")
        with gr.Column(scale=3):
            regular_audio = gr.Audio(label="一般語氣參考音訊", type="filepath")
        with gr.Column(scale=3):
            regular_ref_text = gr.Textbox(label="參考文字（一般語氣）", lines=4)
            with gr.Row():
                regular_seed_slider = gr.Slider(
                    show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="種子；-1 代表隨機"
                )
                regular_speed_slider = gr.Slider(
                    show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="調整語速"
                )
        with gr.Column(scale=1, min_width=160):
            regular_ref_text_file = gr.File(label="從文字檔載入參考文字（.txt）", file_types=[".txt"])

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_ref_text_files = [regular_ref_text_file]
    speech_type_seeds = [regular_seed_slider]
    speech_type_speeds = [regular_speed_slider]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(variant="compact", visible=False) as row:
            with gr.Column(scale=1, min_width=160):
                name_input = gr.Textbox(label="語音類型名稱")
                insert_btn = gr.Button("插入標籤", variant="secondary")
                delete_btn = gr.Button("刪除類型", variant="stop")
            with gr.Column(scale=3):
                audio_input = gr.Audio(label="參考音訊", type="filepath")
            with gr.Column(scale=3):
                ref_text_input = gr.Textbox(label="參考文字", lines=4)
                with gr.Row():
                    seed_input = gr.Slider(
                        show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed. -1 for random"
                    )
                    speed_input = gr.Slider(
                        show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="調整語速"
                    )
            with gr.Column(scale=1, min_width=160):
                ref_text_file_input = gr.File(label="從文字檔載入參考文字（.txt）", file_types=[".txt"])
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_ref_text_files.append(ref_text_file_input)
        speech_type_seeds.append(seed_input)
        speech_type_speeds.append(speed_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Global logic for all speech types
    for i in range(max_speech_types):
        speech_type_audios[i].clear(
            lambda: [None, None],
            None,
            [speech_type_ref_texts[i], speech_type_ref_text_files[i]],
        )
        speech_type_ref_text_files[i].upload(
            load_text_from_file,
            inputs=[speech_type_ref_text_files[i]],
            outputs=[speech_type_ref_texts[i]],
        )

    # Button to add speech type
    add_speech_type_btn = gr.Button("新增語音類型")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("已達語音類型數量上限，請考慮重新啟動程式。")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None, None

    # Update delete button clicks and ref text file changes
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[
                speech_type_rows[i],
                speech_type_names[i],
                speech_type_audios[i],
                speech_type_ref_texts[i],
                speech_type_ref_text_files[i],
            ],
        )

    # Text input for the prompt
    with gr.Row():
        gen_text_input_multistyle = gr.Textbox(
            label="要產生的文字",
            lines=10,
            max_lines=40,
            scale=4,
            placeholder="請在每個段落開頭加入角色名稱或情緒類型，例如：\n\n{Regular} 大家好，這是一段一般語氣。\n{Surprised} 真的嗎？我完全沒想到！\n{Sad} 我今天真的有點難過……\n{Angry} 這件事真的讓我很生氣！\n{Whisper} 我現在用小聲的方式說話。\n{Shouting} 為什麼會這樣？！",
        )
        gen_text_file_multistyle = gr.File(label="從文字檔載入要產生的文字（.txt）", file_types=[".txt"], scale=1)

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name, speech_type_seed, speech_type_speed):
            current_text = current_text or ""
            if not speech_type_name:
                gr.Warning("插入前請先輸入語音類型名稱。")
                return current_text
            speech_type_dict = {
                "name": speech_type_name,
                "seed": speech_type_seed,
                "speed": speech_type_speed,
            }
            updated_text = current_text + json.dumps(speech_type_dict) + " "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i], speech_type_seeds[i], speech_type_speeds[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("進階設定", open=True):
        with gr.Row():
            with gr.Column():
                show_cherrypick_multistyle = gr.Checkbox(
                    label="顯示候選種子挑選介面",
                    info="開啟後可顯示介面，從先前的生成結果中挑選種子。",
                    value=False,
                )
            with gr.Column():
                remove_silence_multistyle = gr.Checkbox(
                    label="移除過長靜音",
                    info="開啟後會自動偵測並裁切過長靜音。",
                    value=True,
                )

    # Generate button
    generate_multistyle_btn = gr.Button("產生多角色／多風格語音", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="合成語音")

    # Used seed gallery
    cherrypick_interface_multistyle = gr.Textbox(
        label="候選種子挑選介面",
        lines=10,
        max_lines=40,
        buttons=["copy"],  # if gradio<6.0
        interactive=False,
        visible=False,
    )

    # Logic control to show/hide the cherrypick interface
    show_cherrypick_multistyle.change(
        lambda is_visible: gr.update(visible=is_visible),
        show_cherrypick_multistyle,
        cherrypick_interface_multistyle,
    )

    # Function to load text to generate from file
    gen_text_file_multistyle.upload(
        load_text_from_file,
        inputs=[gen_text_file_multistyle],
        outputs=[gen_text_input_multistyle],
    )

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_type_name = "Regular"
        inference_meta_data = ""

        for segment in segments:
            name = segment["name"]
            seed_input = segment["seed"]
            speed = segment["speed"]
            text = segment["text"]

            if name in speech_types:
                current_type_name = name
            else:
                gr.Warning(f"找不到語音類型 {name}，將改用 Regular 作為預設。")
                current_type_name = "Regular"

            try:
                ref_audio = speech_types[current_type_name]["audio"]
            except KeyError:
                gr.Warning(f"請提供語音類型 {current_type_name} 的參考音訊。")
                return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]
            ref_text = speech_types[current_type_name].get("ref_text", "")

            if seed_input == -1:
                seed_input = np.random.randint(0, 2**31 - 1)

            # Generate or retrieve speech for this segment
            audio_out, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                text,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=0,
                speed=speed,
                show_info=print,  # no pull to top when generating
            )
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_type_name]["ref_text"] = ref_text_out
            inference_meta_data += json.dumps(dict(name=name, seed=used_seed, speed=speed)) + f" {text}\n"

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (
                [(sr, final_audio_data)]
                + [speech_types[name]["ref_text"] for name in speech_types]
                + [inference_meta_data]
            )
        else:
            gr.Warning("沒有產生任何語音。")
            return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts + [cherrypick_interface_multistyle],
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["name"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# 語音聊天
使用您的參考聲音與 AI 進行語音對話！
1. 上傳參考音訊；可選擇自行輸入逐字稿，或上傳 .txt 文字檔。
2. 載入聊天模型。
3. 使用麥克風錄下訊息，或直接輸入文字。
4. AI 會使用您的參考聲音回答。
"""
    )

    chat_model_name_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        show_info(f"正在載入聊天模型：{chat_model_name}")
        chat_model_state = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(chat_model_name)
        show_info(f"聊天模型 {chat_model_name} 已成功載入！")

        return gr.update(visible=False), gr.update(visible=True)

    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])

    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="聊天模型",
        info="輸入 Hugging Face 聊天模型名稱。",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button("載入聊天模型", variant="primary", visible=not USING_SPACES)
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model, inputs=[chat_model_name_input], outputs=[load_chat_model_btn, chat_interface_container]
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="參考音訊", type="filepath")
            with gr.Column():
                with gr.Accordion("進階設定", open=False):
                    with gr.Row():
                        ref_text_chat = gr.Textbox(
                            label="參考文字",
                            info="選填：留空時自動辨識參考音訊。",
                            lines=2,
                            scale=3,
                        )
                        ref_text_file_chat = gr.File(
                            label="從文字檔載入參考文字（.txt）", file_types=[".txt"], scale=1
                        )
                    with gr.Row():
                        randomize_seed_chat = gr.Checkbox(
                            label="每次使用隨機種子",
                            value=True,
                            info="取消勾選後會使用右側指定的種子。",
                            scale=3,
                        )
                        seed_input_chat = gr.Number(show_label=False, value=0, precision=0, scale=1)
                    remove_silence_chat = gr.Checkbox(
                        label="移除過長靜音",
                        value=True,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="系統提示詞",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(
            label="對話內容"
        )  # type="messages" hard-coded and no need to pass in since gradio 6.0

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="說出您的訊息",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="輸入您的訊息",
                    lines=1,
                )
                send_btn_chat = gr.Button("傳送訊息")
                clear_btn_chat = gr.Button("清除對話")

        # Modify process_audio_input to generate user input
        @gpu_decorator
        def process_audio_input(conv_state, audio_path, text):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return conv_state

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]
            if not text.strip():
                return conv_state

            conv_state.append({"role": "user", "content": text})
            return conv_state

        # Use model and tokenizer from state to get text response
        @gpu_decorator
        def generate_text_response(conv_state, system_prompt):
            """Generate text response from AI"""
            for single_state in conv_state:
                if isinstance(single_state["content"], list):
                    assert len(single_state["content"]) == 1 and single_state["content"][0]["type"] == "text"
                    single_state["content"] = single_state["content"][0]["text"]

            system_prompt_state = [{"role": "system", "content": system_prompt}]
            response = chat_model_inference(system_prompt_state + conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            return conv_state

        @gpu_decorator
        def generate_audio_response(conv_state, ref_audio, ref_text, remove_silence, randomize_seed, seed_input):
            """Generate TTS audio for AI response"""
            if not conv_state or not ref_audio:
                return None, ref_text, seed_input

            last_ai_response = conv_state[-1]["content"][0]["text"]
            if not last_ai_response or conv_state[-1]["role"] != "assistant":
                return None, ref_text, seed_input

            if randomize_seed:
                seed_input = np.random.randint(0, 2**31 - 1)

            audio_result, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result, ref_text_out, used_seed

        def clear_conversation():
            """Reset the conversation"""
            return [], None

        ref_text_file_chat.upload(
            load_text_from_file,
            inputs=[ref_text_file_chat],
            outputs=[ref_text_chat],
        )

        for user_operation in [audio_input_chat.stop_recording, text_input_chat.submit, send_btn_chat.click]:
            user_operation(
                process_audio_input,
                inputs=[chatbot_interface, audio_input_chat, text_input_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_text_response,
                inputs=[chatbot_interface, system_prompt_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_audio_response,
                inputs=[
                    chatbot_interface,
                    ref_audio_chat,
                    ref_text_chat,
                    remove_silence_chat,
                    randomize_seed_chat,
                    seed_input_chat,
                ],
                outputs=[audio_output_chat, ref_text_chat, seed_input_chat],
            ).then(
                lambda: [None, None],
                None,
                [audio_input_chat, text_input_chat],
            )

        # Handle clear button or system prompt change and reset conversation
        for user_operation in [clear_btn_chat.click, system_prompt_chat.change, chatbot_interface.clear]:
            user_operation(
                clear_conversation,
                outputs=[chatbot_interface, audio_output_chat],
            )


with gr.Blocks() as app_credits:
    gr.Markdown("""
# 關於與致謝

* [mrfakename](https://github.com/fakerybakery)：原始 [線上示範](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad)：初始分段生成與 Podcast 應用探索
* [jpgallegoar](https://github.com/jpgallegoar)：多語音類型生成與語音聊天功能
""")


with gr.Blocks() as app:
    gr.Markdown(
        f"""
# F5-TTS 繁體中文介面

這是 {"[F5-TTS](https://github.com/SWivid/F5-TTS) 的本機 WebUI" if not USING_SPACES else "[F5-TTS](https://github.com/SWivid/F5-TTS) 的線上示範介面"}，支援進階批次處理。此介面可使用下列 TTS 模型：

* [F5-TTS](https://arxiv.org/abs/2410.06885)（Flow Matching 零樣本語音合成模型）
* [E2 TTS](https://arxiv.org/abs/2406.18009)（全非自回歸零樣本 TTS 模型）

目前官方檢查點主要支援英文與中文。

若原始音檔很長，可先使用「音訊工作台」自動找自然片段、裁切／優化並用 Whisper 提取文字，再一鍵套用到基本 TTS。

**注意：如果沒有提供參考文字，系統會使用 Whisper 自動辨識。為獲得較佳效果，參考音訊建議控制在 12 秒以內，並確認音訊已完整上傳後再開始生成。**
"""
    )

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[(DEFAULT_TTS_MODEL, DEFAULT_TTS_MODEL), ("E2-TTS", "E2-TTS"), ("自訂模型", "Custom")], label="選擇 TTS 模型", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[(DEFAULT_TTS_MODEL, DEFAULT_TTS_MODEL), ("E2-TTS", "E2-TTS")], label="選擇 TTS 模型", value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="模型：本機路徑 | hf://使用者/儲存庫/模型檔",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="詞彙表：本機路徑 | hf://使用者/儲存庫/詞彙表檔案",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="模型設定：Dictionary 格式",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_audio_tools, app_tts, app_multistyle, app_chat, app_credits],
        ["音訊工作台", "基本 TTS", "多角色語音", "語音聊天", "關於"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="指定 WebUI 使用的連接埠")
@click.option("--host", "-H", default=None, help="指定 WebUI 主機位址")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="透過 Gradio 分享連結公開介面",
)
@click.option("--api", "-a", default=True, is_flag=True, help="允許 API 存取")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="啟動後自動在預設瀏覽器開啟介面",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("正在啟動 F5-TTS……")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        root_path=root_path,
        inbrowser=inbrowser,
        allowed_paths=[_AUDIO_TOOL_OUTPUT_DIR],
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch(allowed_paths=[_AUDIO_TOOL_OUTPUT_DIR])
