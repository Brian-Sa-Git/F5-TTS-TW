# ruff: noqa: E402
# 繁體中文介面版：僅翻譯 Gradio 顯示文字，保留模型名稱與內部邏輯。
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import os
import re
import tempfile
from collections import OrderedDict
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


with gr.Blocks() as app_tts:
    gr.Markdown("# 批量文字轉語音")
    ref_audio_input = gr.Audio(label="參考音訊", type="filepath")
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
                label="參考文字",
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
        return audio_out, spectrogram_path, ref_text_out, used_seed

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
        outputs=[audio_output, spectrogram_output, ref_text_input, seed_input],
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
        buttons=["copy"],  # show_copy_button=True if gradio<6.0
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

若遇到問題，建議將參考音訊轉成 WAV 或 MP3，並使用右下角的 ✂ 將片段裁切到 12 秒以內，以避免自動裁切結果不理想。

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
        [app_tts, app_multistyle, app_chat, app_credits],
        ["基本 TTS", "多角色語音", "語音聊天", "關於"],
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
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
