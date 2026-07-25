"""Cross-model memorization probe on truncated audio (perplexity + EOS-ban gen).

Supports the decoder ASR models whose logits we can reach in the hf_transformers
env: cohere (encoder-decoder), granite-speech and voxtral (decoder speech-LLMs).

For each hide-K sample, the target is the model's own greedy transcription of the
FULL audio (natural tokenization, validated to end with the hidden words). Two
signals, both scored only on the hidden-tail tokens:

  PERPLEXITY recovery = (nll_none - nll_trunc) / (nll_none - nll_full)
    r≈0 honest (truncated audio says nothing about unheard words);
    r≈1 truncated audio predicts them as well as hearing them ⇒ retrieval.

  GENERATIVE eos-ban: greedily decode the TRUNCATED audio with EOS banned,
    forcing the model to keep emitting; gen_leak = fraction of hidden content
    words that appear in that forced continuation. A memorizing model completes
    with the verbatim tail; an honest one emits filler/repeats.

Both use identical truncated audio per model, so the comparison is apples-to-
apples regardless of exact cut precision.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
import torch

# transformers powers the HF families' EOS-ban generation. Some per-model envs
# (e.g. envs/omnilingual, a pure fairseq2 stack) don't ship it; import lazily so
# the perplexity path still loads. eos_ban_generate re-imports LogitsProcessorList
# at call time and is unsupported (NotImplementedError) for those families.
try:
    from transformers import LogitsProcessor as _LogitsProcessor
    from transformers import LogitsProcessorList  # noqa: F401
except ImportError:
    _LogitsProcessor = object
    LogitsProcessorList = None

import roots as _paths  # data/model roots from BENCHMARK_OPT_DATA / BENCHMARK_OPT_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("memprobe")

# Each entry: (model_id, arch, family). `family` selects the loader/teacher-forcing
# branch in Probe; arch ∈ {encdec, declm} selects the scoring path. Kept backward
# compatible: 2-tuples still parse (family inferred from key/arch).
MODELS = {
    # --- existing, validated ---
    "cohere-transcribe": ("CohereLabs/cohere-transcribe-03-2026", "encdec", "cohere"),
    "granite-speech-3.3-8b": ("ibm-granite/granite-speech-3.3-8b", "declm", "granite"),
    "voxtral-mini-3b": ("mistralai/Voxtral-Mini-3B-2507", "declm", "voxtral"),
    "phi4-multimodal": ("microsoft/Phi-4-multimodal-instruct", "declm", "phi4"),
    "whisper-large-v3": ("openai/whisper-large-v3", "encdec", "whisper"),
    # --- TIER 1: whisper-family encoder-decoders (reuse whisper path) ---
    "crisper-whisper": ("nyrahealth/CrisperWhisper", "encdec", "whisper"),
    "distil-whisper-large-v3.5": ("distil-whisper/distil-large-v3.5", "encdec", "whisper"),
    "lite-whisper-large-v3": ("efficient-speech/lite-whisper-large-v3", "encdec", "whisper-auto"),
    "lite-whisper-large-v3-acc": ("efficient-speech/lite-whisper-large-v3-acc", "encdec", "whisper-auto"),
    # --- TIER 1: granite-family declm (reuse granite chat-template path) ---
    "granite-speech-3.3-2b": ("ibm-granite/granite-speech-3.3-2b", "declm", "granite"),
    "granite-speech-4.0-1b": ("ibm-granite/granite-4.0-1b-speech", "declm", "granite"),
    "granite-speech-4.1-2b": ("ibm-granite/granite-speech-4.1-2b", "declm", "granite"),
    # --- TIER 1: moonshine encoder-decoder ---
    "moonshine-streaming-medium": ("usefulsensors/moonshine-streaming-medium", "encdec", "moonshine"),
    # --- TIER 2: GLM-ASR declm (apply_transcription_request, like voxtral) ---
    "glm-asr-nano": ("zai-org/GLM-ASR-Nano-2512", "declm", "glm"),
    # --- TIER 2: Qwen3-ASR custom forward ---
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B", "declm", "qwen3"),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B", "declm", "qwen3"),
    # --- TIER 3: NeMo Canary AED (EncDecMultiTaskModel) ---
    "canary-1b": ("nvidia/canary-1b", "nemo_aed", "canary"),
    # --- TIER 3: NeMo SALM (Canary encoder + Qwen3 LLM decoder) ---
    "canary-qwen-2.5b": ("nvidia/canary-qwen-2.5b", "declm", "salm"),
    # --- Higgs Audio v3 (Whisper-v3 enc + Qwen3-8B dec, LoRA). Custom remote
    # code; teacher-forced via the repo collator. Needs envs/higgs_audio
    # (transformers==4.51.0). ---
    "higgs-audio-v3-8b-stt-v2": ("bosonai/higgs-audio-v3-8b-stt-v2", "declm", "higgs"),
    # --- Kimi-Audio 7B (MoonshotAI): dual-stream MIMO LM (parallel audio +
    # text token streams + continuous whisper features). The inner `alm`
    # (AutoModelForCausalLM, trust_remote_code) forward returns
    # (audio_logits, text_logits, past_kv); we teacher-force the TEXT stream
    # (audio stream blanked, as in output_type="text" generation). Needs
    # envs/kimi_audio (the kimia_infer package). ---
    "kimi-audio-7b": ("moonshotai/Kimi-Audio-7B-Instruct", "declm", "kimi"),
    # --- Omnilingual ASR LLM 3B (Meta): wav2vec2 encoder + Llama decoder via
    # fairseq2. Loaded through the existing OmnilingualTranscriptionModel wrapper
    # (monkey-patched local .pt + SentencePiece tokenizer). The model.forward
    # takes a Seq2SeqBatch (audio source + forced text target), builds the
    # syntax `audio <lid_marker> <lang_id> <bos> target_text <eos>` itself, and
    # with return_logits=True returns decoder logits + the context lengths so we
    # can read the per-token logprobs of the forced target span. Needs
    # envs/omnilingual. ---
    # omni-3b-llm is omitted: loading it needed an internal wrapper that is not
    # part of this release, and the model is not in the paper's roster. The
    # family == "omni" branches below are kept as a reference implementation.
}


def _model_spec(key: str):
    spec = MODELS[key]
    if len(spec) == 2:
        model_id, arch = spec
        return model_id, arch, key
    return spec
_STOP = {"the", "a", "an", "of", "to", "and", "in", "is", "it", "that", "this", "was", "for",
         "on", "as", "at", "be", "by", "or", "we", "are", "with", "his", "her", "its", "their"}


class BanEOS(_LogitsProcessor):
    def __init__(self, eos_ids):
        self.eos_ids = eos_ids

    def __call__(self, input_ids, scores):
        for e in self.eos_ids:
            scores[:, e] = float("-inf")
        return scores


def load_wav(p: Path) -> np.ndarray:
    a, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if a.ndim == 2:
        a = a.mean(axis=1)
    assert sr == 16000, sr
    return a


class Probe:
    def __init__(self, key: str, lang: str = "en"):
        self.model_id, self.arch, self.family = _model_spec(key)
        self.dev = "cuda"
        # ISO 639-1 language conditioning for the decoder prompt / language tokens.
        # Default "en" reproduces the original English-only behaviour exactly; the
        # multilingual courtesy probe sets es/de/fr/it. Threaded through the cohere/
        # whisper/qwen3/voxtral prompt builders below (granite/phi4/salm auto-detect).
        self.lang = lang
        # `is_whisper` drives the whisper-specific generate()/encoder paths. The
        # compressed lite-whisper checkpoints share the whisper decoder API but
        # need AutoModel + trust_remote_code to build the custom encoder.
        self.is_whisper = self.family in ("whisper", "whisper-auto")
        fam = self.family
        if fam == "whisper":
            from transformers import AutoProcessor, WhisperForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(self.dev).eval()
            self.tok = self.proc.tokenizer
        elif fam == "whisper-auto":
            # lite-whisper: compressed encoder via custom code, whisper decoder.
            from transformers import AutoModel, AutoProcessor
            self.proc = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=torch.float16
            ).to(self.dev).eval()
            self.tok = self.proc.tokenizer
        elif fam == "moonshine":
            from transformers import AutoProcessor
            try:
                from transformers import MoonshineStreamingForConditionalGeneration as _MoonCls
            except ImportError:
                from transformers import MoonshineForConditionalGeneration as _MoonCls
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.model = _MoonCls.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(self.dev).eval()
            self.tok = self.proc.tokenizer
        elif fam == "cohere":
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                self.model_id, device_map=self.dev
            ).eval()
            self.tok = self.proc.tokenizer if hasattr(self.proc, "tokenizer") else self.proc
        elif fam == "granite":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.tok = self.proc.tokenizer
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16
            ).to(self.dev).eval()
            chat = [
                {"role": "system", "content": "You are a speech recognition system. Output only the transcription, nothing else."},
                {"role": "user", "content": "<|audio|>Transcribe the audio."},
            ]
            self._granite_prompt = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        elif fam == "phi4":
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.tok = self.proc.tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype="auto", _attn_implementation="eager"
            ).to(self.dev).eval()
            self._phi4_prompt = "<|user|><|audio_1|>Transcribe the audio.<|end|><|assistant|>"
        elif fam == "glm":
            from transformers import AutoProcessor, GlmAsrForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.tok = self.proc.tokenizer
            self.model = GlmAsrForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map=self.dev
            ).eval()
        elif fam == "qwen3":
            from qwen_asr import Qwen3ASRModel
            self.qwen = Qwen3ASRModel.from_pretrained(
                self.model_id, dtype=torch.bfloat16, device_map="cuda:0",
                max_inference_batch_size=1, max_new_tokens=160,
            )
            # The top-level Qwen3ASRForConditionalGeneration delegates generate to
            # `.thinker` and has no usable forward; teacher-force the thinker
            # directly (it accepts input_ids + input_features + feature mask).
            top = getattr(self.qwen, "model", None) or getattr(self.qwen, "llm", None)
            self.model = getattr(top, "thinker", top)
            self.proc = getattr(self.qwen, "processor", None)
            self.tok = getattr(self.qwen, "tokenizer", None) or (
                getattr(self.proc, "tokenizer", None) if self.proc else None
            )
            # Reproduce qwen_asr's transformers prompt: chat template + forced lang.
            msgs = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ]
            base = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            self._qwen_prompt = base + f"language {self.lang}<asr_text>"
        elif fam == "canary":
            import nemo.collections.asr as nemo_asr
            from nemo.collections.common.prompts.canary import CanaryPromptFormatter
            self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b").to(self.dev).eval()
            self.tok = self.model.tokenizer
            self._canary_fmt = CanaryPromptFormatter(self.model.tokenizer)
            # Canary EOS lives on the NeMo tokenizer, not a HF generation_config.
            self.key = key
            self.mask_eos = True
            self.eos_ids = [int(self.model.tokenizer.eos)]
            return
        elif fam == "salm":
            # SALM = Canary encoder + Qwen3-1.7B LLM decoder (NeMo speechlm2).
            # Teacher-forcing path: build prompt+forced-transcript token ids with
            # the model's own qwen PromptFormatter, embed text via the DETACHED
            # `model.embed_tokens`, splice audio frames into the audio_locator_tag
            # positions via `replace_placeholders_and_build_targets` (same call
            # `prepare_inputs` uses for training), then `model.forward` for logits.
            from nemo.collections.common.prompts import PromptFormatter
            from nemo.collections.speechlm2.models import SALM

            self.model = SALM.from_pretrained(self.model_id).to(self.dev).eval()
            self.tok = self.model.tokenizer
            self._salm_fmt = PromptFormatter.resolve(self.model.cfg.prompt_format)(self.tok)
            self._salm_audio_tag = self.model.audio_locator_tag
            # Prompt mirrors local_salm.py: "Transcribe the following: <tag>".
            self._salm_user_msg = f"Transcribe the following: {self._salm_audio_tag}"
            # NeMo AutoTokenizer eos lives on the tokenizer, not a HF gen config.
            self.key = key
            self.mask_eos = True
            self.eos_ids = [int(self.tok.eos_id)]
            return
        elif fam == "higgs":
            # Higgs Audio v3: Whisper-v3 encoder + Qwen3-8B decoder (LoRA),
            # custom remote code. Load via AutoModel + side-load the repo's
            # transcribe.py (prompt build + HiggsAudioSampleCollator). The
            # collator duplicates the SINGLE <|AUDIO|> placeholder once per 30s
            # chunk; the model's forward then EXPANDS each placeholder into the
            # whisper audio frames and returns `expanded_input_ids` aligned to
            # `logits`, so we recover the forced-transcript span from the
            # expanded ids (see _declm_inputs / token_logprobs higgs branch).
            import importlib.util
            import sys

            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import cached_file

            self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
                attn_implementation="eager", device_map="cuda:0",
            ).eval()
            self.proc = None
            # Side-load the repo's transcribe.py (and its sibling collator) from
            # the cached snapshot so its relative imports resolve.
            tpath = cached_file(self.model_id, "transcribe.py")
            cached_file(self.model_id, "higgs_audio_collator.py")
            snap_dir = tpath.rsplit("/", 1)[0]
            if snap_dir not in sys.path:
                sys.path.insert(0, snap_dir)
            _spec = importlib.util.spec_from_file_location("higgs_transcribe", tpath)
            self._higgs = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(self._higgs)
            self._higgs_collator = self._higgs._create_collator(self.model.config)
            # enable_thinking=False → no <think> block between prompt and answer.
            prompt = getattr(self._higgs, "DEFAULT_PROMPT", "Transcribe the speech.")
            self._higgs_prefix_ids = self._higgs._build_input_tokens(self.tok, prompt, enable_thinking=False)
            self.key = key
            self.mask_eos = True
            eos_ids = set()
            for t in ("<|im_end|>", "<|endoftext|>"):
                tid = self.tok.convert_tokens_to_ids(t)
                if tid is not None and tid != getattr(self.tok, "unk_token_id", None):
                    eos_ids.add(int(tid))
            if getattr(self.tok, "eos_token_id", None) is not None:
                eos_ids.add(int(self.tok.eos_token_id))
            eos_ids.discard(None)
            self.eos_ids = sorted(eos_ids)
            return
        elif fam == "kimi":
            # Kimi-Audio: dual-stream MIMO LM. Load via the kimia_infer API
            # (same path local_kimi_audio.py uses), then reach the inner
            # `alm` (AutoModelForCausalLM) + prompt_manager for teacher forcing.
            #
            # The prompt is two PARALLEL, position-aligned streams of equal
            # length: an AUDIO stream (discrete glm4-voice audio tokens +
            # continuous whisper features spliced at is_continuous positions +
            # control tokens) and a TEXT stream (text tokens, kimia_text_blank
            # elsewhere). The model forward(input_ids=audio, text_input_ids=text,
            # whisper_input_feature=..., is_continuous_mask=..., position_ids=...)
            # returns (audio_logits, text_logits, past_kv). For ASR we score the
            # TEXT stream: build prefix [user-text-instruction + user-audio +
            # assistant-start], append the forced transcript text tokens to the
            # text stream (and kimia_text_blank to the audio stream at those same
            # positions), one forward, read text_logits at the forced positions.
            try:
                import transformers.modeling_utils as _mu

                _orig_safe_open = _mu.safe_open

                class _SafeOpenProxy:
                    def __init__(self, inner):
                        self._inner = inner

                    def __enter__(self):
                        self._inner.__enter__()
                        return self

                    def __exit__(self, et, e, tb):
                        return self._inner.__exit__(et, e, tb)

                    def metadata(self):
                        return self._inner.metadata() or {"format": "pt"}

                    def __getattr__(self, name):
                        return getattr(self._inner, name)

                _mu.safe_open = lambda *a, **k: _SafeOpenProxy(_orig_safe_open(*a, **k))
            except Exception:
                pass

            from kimia_infer.api.kimia import KimiAudio

            self._kimi = KimiAudio(model_path=self.model_id, load_detokenizer=False)
            self.model = self._kimi.alm
            self._pm = self._kimi.prompt_manager
            self.tok = self._pm.text_tokenizer
            self._kimi_xt = self._pm.extra_tokens
            self._kimi_text_blank = int(self._kimi_xt.kimia_text_blank)
            # ASR instruction mirrors local_kimi_audio.py (English here; the
            # consensus probe is vox-EN). Built once; audio is per-utterance.
            self._kimi_instruction = (
                "Please transcribe the following audio in English. "
                "Output only the transcription in the source language, do not translate."
            )
            self.key = key
            self.mask_eos = True
            # The TEXT stream's terminator is kimia_text_eos (msg_end/media_end
            # ride the AUDIO stream, so they don't appear in text_logits scoring).
            # The custom TikTokenTokenizer has no working convert_tokens_to_ids,
            # so take the id from the already-resolved extra_tokens.
            self.eos_ids = [int(self._kimi_xt.kimia_text_eos)]
            return
        elif fam == "omni":
            # Omnilingual ASR LLM 3B: wav2vec2 encoder + Llama decoder (fairseq2).
            # Reuse the project wrapper for loading: it applies the monkey-patched
            # local-.pt / SentencePiece-tokenizer loaders and builds an
            # ASRInferencePipeline whose `.model` is the Wav2Vec2LlamaModel and
            # `.tokenizer` is the fairseq2 SP tokenizer. We then reach the model's
            # own forward for teacher-forcing (see _omni_token_logprobs).
            #
            # The 3B checkpoint is an LLM_ASR_LID model (lang_embeddings_p>0), so
            # its decoder syntax is `audio <lid_marker> <lang_id> <bos> text <eos>`;
            # language conditioning rides batch.example["lang"] (here eng_Latn for
            # the vox-EN consensus probe). model.forward(batch, return_logits=True)
            # returns (loss, logits, layout, context_inputs, context_seq_lens,
            # audio_embeddings); position (context_len-1 + j) predicts target[j]
            # (mirrors remove_context_logits in the training loss).
            raise SystemExit(
                "The 'omni' backend needs an internal loader wrapper that is not part of\n"
                "this release. The teacher-forcing logic below is kept as a reference\n"
                "implementation; supply your own loader to use it."
            )

            # A loader must set: self.model (the Wav2Vec2Llama model),
            # self.tok (its SentencePiece tokenizer), and
            # self._omni_Seq2SeqBatch (fairseq2's Seq2SeqBatch class). The
            # teacher-forcing helpers below need only those three.
            self._omni_enc = self.tok.create_encoder()
            self._omni_dec = self.tok.create_decoder(skip_special_tokens=True)
            self._omni_dtype = self._omni_pipeline.dtype
            self._omni_lang = "eng_Latn"  # vox-EN consensus probe is English
            vi = self.tok.vocab_info
            self._omni_bos = int(vi.bos_idx)
            self._omni_eos = int(vi.eos_idx)
            self._omni_pad = int(vi.pad_idx)
            self.key = key
            self.mask_eos = True
            self.eos_ids = [self._omni_eos]
            return
        else:  # voxtral
            from transformers import AutoProcessor, VoxtralForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(self.model_id)
            self.tok = self.proc.tokenizer
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map=self.dev
            ).eval()
        self.key = key
        self.mask_eos = True
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        if getattr(self.tok, "eos_token_id", None) is not None:
            eos_ids.add(self.tok.eos_token_id)
        if key == "phi4-multimodal":
            end_id = self.tok.convert_tokens_to_ids("<|end|>")
            if end_id is not None and end_id != getattr(self.tok, "unk_token_id", None):
                eos_ids.add(end_id)
        if self.family == "glm":
            # GLM emits <|user|>/<|assistant|>/<|endoftext|>-style turn enders.
            for t in ("<|endoftext|>", "<|user|>", "<|observation|>", "<sop>", "<eop>"):
                tid = self.tok.convert_tokens_to_ids(t)
                if tid is not None and tid != getattr(self.tok, "unk_token_id", None):
                    eos_ids.add(tid)
        eos_ids.discard(None)
        self.eos_ids = sorted(eos_ids)

    # ---- input construction (declm) ----
    def _declm_inputs(self, audio: np.ndarray):
        if self.family == "granite":
            inp = self.proc([self._granite_prompt], [audio], return_tensors="pt")
        elif self.family == "phi4":
            inp = self.proc(text=self._phi4_prompt, audios=[(audio, 16000)], return_tensors="pt")
        elif self.family == "glm":
            inp = self.proc.apply_transcription_request([audio])
        elif self.family == "qwen3":
            inp = self.proc(text=[self._qwen_prompt], audio=[audio], return_tensors="pt", padding=True)
        else:  # voxtral
            inp = self.proc.apply_transcription_request(
                audio=[audio], model_id=self.model_id, language=[self.lang], sampling_rate=16000, format=["wav"]
            )
        inp = inp.to(self.dev)
        # cast float feature tensors to model dtype
        for k, v in list(inp.items()):
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                inp[k] = v.to(self.model.dtype)
        return inp

    # ---- higgs (Whisper enc + Qwen3 dec, custom collator) helpers ----
    def _higgs_collate(self, audio: np.ndarray, answer_ids: list[int]):
        """Collate one ChatML sample = (prefix prompt + forced `answer_ids`) with
        `audio`. Returns (batch_dict, full_text_ids) where full_text_ids is the
        UN-expanded input_ids fed to the collator (prefix incl. one <|AUDIO|>
        placeholder, then the forced answer). The collator duplicates <|AUDIO|>
        once per 30s chunk; the model forward expands each into whisper frames."""
        from dataclasses import asdict

        input_ids = list(self._higgs_prefix_ids) + list(answer_ids)
        sample = self._higgs._build_sample(audio.astype(np.float32), input_ids, sample_rate=16000)
        batch = asdict(self._higgs_collator([sample]))
        dev = next(self.model.parameters()).device
        batch = {
            k: (v.to(dev).contiguous() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        return batch, input_ids

    def _higgs_answer_span(self, expanded_ids: list[int], answer_ids: list[int]) -> int:
        """Index into `expanded_ids` (the model's post-merge `expanded_input_ids`)
        where the forced `answer_ids` begin. The answer is the contiguous tail run
        matching `answer_ids` (audio frames + prompt sit before it; right-pad sits
        after). Search from the right for the last exact contiguous match."""
        n = len(answer_ids)
        if n == 0:
            return -1
        for start in range(len(expanded_ids) - n, -1, -1):
            if expanded_ids[start:start + n] == answer_ids:
                return start
        return -1

    def _higgs_token_logprobs(self, audio: np.ndarray, answer_ids: list[int]) -> np.ndarray:
        """Per-token logprobs of `answer_ids` given `audio`. Aligned to a fresh
        target array of len(answer_ids): out[j] = logp(answer_ids[j] | prefix,
        audio, answer_ids[:j]). out[0] is the first forced token (predicted from
        the position immediately before it in the expanded sequence)."""
        batch, _ = self._higgs_collate(audio, answer_ids)
        out_obj = self.model(**batch)
        logits = out_obj.logits[0].float()  # [T_exp, V]
        exp_ids = out_obj.expanded_input_ids[0].tolist()
        start = self._higgs_answer_span(exp_ids, answer_ids)
        out = np.full(len(answer_ids), np.nan)
        if start < 0:
            log.warning("higgs: forced answer span not found in expanded_input_ids")
            return out
        self._mask_eos(logits)
        lp = torch.log_softmax(logits, dim=-1)
        # Position (start + j - 1) predicts token at (start + j) == answer_ids[j].
        for j in range(len(answer_ids)):
            pos = start + j - 1
            if pos < 0:
                continue
            out[j] = float(lp[pos, answer_ids[j]])
        return out

    def _encdec_enc(self, audio: np.ndarray):
        if self.family == "moonshine":
            inp = self.proc([audio], sampling_rate=16000, return_tensors="pt").to(self.dev, dtype=self.model.dtype)
            feat = inp.get("input_values", inp.get("input_features"))
            return self.model.get_encoder()(feat, attention_mask=inp.get("attention_mask"))
        if self.is_whisper:
            inp = self.proc([audio], sampling_rate=16000, return_tensors="pt").to(self.dev, dtype=self.model.dtype)
            return self.model.get_encoder()(inp["input_features"])
        inp = self.proc([audio], sampling_rate=16000, return_tensors="pt", language=self.lang).to(
            self.dev, dtype=self.model.dtype
        )
        enc = self.model.get_encoder()(inp["input_features"], attention_mask=inp.get("attention_mask"))
        return enc

    # ---- canary (NeMo AED) helpers ----
    def _canary_prompt_ids(self, text: str) -> tuple[list[int], int]:
        """Build the full prompted-transcript ids (prompt + answer) for canary.
        Returns (ids, n_prompt) so the answer region starts at n_prompt."""
        turns = [
            {"role": "user", "slots": {
                "source_lang": "<|en|>", "task": "<|transcribe|>",
                "target_lang": "<|en|>", "pnc": "<|pnc|>",
                self._canary_fmt.PROMPT_LANGUAGE_SLOT: "spl_tokens",
            }},
            {"role": "assistant", "slots": {
                "text": text, self._canary_fmt.PROMPT_LANGUAGE_SLOT: "en",
            }},
        ]
        ans = self._canary_fmt.encode_dialog(turns)
        ids = ans["input_ids"].tolist()
        n_prompt = int(ans["context_ids"].shape[0])
        # Drop trailing EOS from answer (matches training: scored continuation only).
        if ids and ids[-1] == int(self.tok.eos):
            ids = ids[:-1]
        return ids, n_prompt

    def _canary_logprobs_raw(self, audio: np.ndarray, target_ids: list[int]) -> np.ndarray:
        sig = torch.tensor([audio], device=self.dev, dtype=torch.float32)
        sig_len = torch.tensor([audio.shape[0]], device=self.dev)
        tgt = torch.tensor([target_ids], device=self.dev)
        tgt_len = torch.tensor([len(target_ids)], device=self.dev)
        log_probs, _, _, _ = self.model.forward(
            input_signal=sig, input_signal_length=sig_len,
            transcript=tgt, transcript_length=tgt_len,
        )
        lp = log_probs[0].float()  # [T, V]; position t predicts token t+1
        out = np.full(len(target_ids), np.nan)
        for j in range(1, len(target_ids)):
            out[j] = float(lp[j - 1, target_ids[j]])
        return out

    # ---- salm (NeMo SALM: Canary enc + Qwen3 LLM) helpers ----
    def _salm_prompt_ids(self, text: str) -> tuple[list[int], int]:
        """Build full prompted-transcript ids (user prompt + assistant answer) for
        SALM via the qwen PromptFormatter. The user message contains the
        audio_locator_tag placeholder (a SINGLE token id that gets expanded to N
        audio frames at embed time). Returns (ids, n_prompt_tokens) where
        n_prompt_tokens counts the context turns (incl. the lone placeholder
        token) BEFORE audio expansion — used only as a sanity value; the true
        answer span is recovered from the spliced target_ids in
        `_salm_token_logprobs`."""
        turns = [
            {"role": "user", "slots": {"message": self._salm_user_msg}},
            {"role": "assistant", "slots": {"message": text}},
        ]
        enc = self._salm_fmt.encode_dialog(turns)
        ids = enc["input_ids"].tolist()
        n_prompt = int(enc["context_ids"].shape[0])
        return ids, n_prompt

    def _salm_token_logprobs(self, audio: np.ndarray, ids: list[int], n_prompt: int) -> np.ndarray:
        """Per-token logprobs of the answer-span tokens of `ids` given `audio`.

        Replicates SALM.prepare_inputs: encode audio -> embed text tokens with the
        detached embed_tokens -> splice audio frames into the locator-tag position
        -> forward -> log_softmax. The audio_locator_tag expands to many frames, so
        the answer-start offset is read from the spliced/shifted target_ids (the
        same -100-masked targets the training path builds), NOT from n_prompt.
        Returns an array aligned to `ids` (NaN on prompt/pad positions; finite on
        the answer tokens) so the caller's hidden-index gather over `ids` works."""
        import torch as _t

        m = self.model
        input_ids = _t.tensor([ids], device=self.dev, dtype=_t.long)
        # loss_mask: True only on the answer span (tokens after the prompt). The
        # locator tag sits inside the prompt, so it is correctly excluded.
        loss_mask = _t.zeros_like(input_ids, dtype=_t.bool)
        loss_mask[:, n_prompt:] = True

        sig = _t.tensor([audio], device=self.dev, dtype=_t.float32)
        sig_len = _t.tensor([audio.shape[0]], device=self.dev, dtype=_t.long)
        audio_embs, audio_emb_lens = m.perception(input_signal=sig, input_signal_length=sig_len)
        audio_embs = [emb[:elen] for emb, elen in zip(audio_embs, audio_emb_lens)]

        from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets

        ids_to_embed = _t.where(input_ids == m.audio_locator_tag_id, 0, input_ids)
        text_embs = m.embed_tokens(ids_to_embed)
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=input_ids,
            embeds=text_embs,
            padding_id=m.text_pad_id,
            placeholder_id=m.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=input_ids.where(loss_mask, -100),
        )
        # Standard next-token shift (mirrors prepare_inputs).
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        logits = m.forward(input_embs.to(m.embed_tokens.weight.dtype), attention_mask=attention_mask)["logits"][0]
        logits = logits.float()  # [T, V]; position t predicts target_ids[t]
        self._mask_eos(logits)
        lp = _t.log_softmax(logits, dim=-1)
        _em = _t.exp(lp[:, self.eos_ids]).sum(-1) if getattr(self, "_cap_eos", False) else None

        tgt = target_ids[0]  # [T]; -100 everywhere except answer positions
        # The order of finite target positions matches the answer-token order in
        # `ids[n_prompt:]` (splice preserves left-to-right order). Map them back.
        out = np.full(len(ids), np.nan)
        ans_positions = (tgt != -100).nonzero(as_tuple=True)[0].tolist()
        n_ans = len(ids) - n_prompt
        if len(ans_positions) != n_ans:
            # Shape mismatch (e.g. tag in answer, TP truncation) — score what we can.
            log.warning("salm: spliced answer span %d != expected %d", len(ans_positions), n_ans)
        for k, pos in enumerate(ans_positions):
            id_idx = n_prompt + k
            if id_idx >= len(ids):
                break
            tid = int(tgt[pos])
            # The qwen PromptFormatter appends a trailing EOS inside the assistant
            # answer span. When mask_eos is on, that EOS logit is -inf, so scoring it
            # would poison the whole clip's NLL. Leave it NaN (dropped by the caller),
            # matching the mask_eos intent of not scoring the EOS emission itself.
            if self.mask_eos and tid in self.eos_ids:
                continue
            out[id_idx] = float(lp[pos, tid])
            if _em is not None:
                self._eos_aln[id_idx] = float(_em[pos])
        return out

    # ---- kimi (MoonshotAI dual-stream MIMO LM) helpers ----
    def _kimi_prefix(self, audio: np.ndarray):
        """Build the audio-conditioned prompt prefix as a KimiAContent (parallel
        audio + text streams + continuous whisper feature), via the same
        prompt_manager path inference uses. Messages = text instruction + audio;
        add_assistant_start_msg=True appends the lone assistant-start role token
        so the forced transcript follows immediately. Audio is written to a temp
        wav (the prompt_manager tokenizes/encodes from a path)."""
        import os
        import tempfile

        import soundfile as sf

        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio.astype(np.float32), 16000)
                tmp = f.name
            messages = [
                {"role": "user", "message_type": "text", "content": self._kimi_instruction},
                {"role": "user", "message_type": "audio", "content": tmp},
            ]
            prefix = self._pm.get_prompt(messages, output_type="text", add_assistant_start_msg=True)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return prefix

    def _kimi_token_logprobs(self, audio: np.ndarray, target_ids: list[int]) -> np.ndarray:
        """Per-token logprobs of the forced transcript `target_ids` (TEXT-stream
        token ids) given `audio`. Builds prefix + forced text, runs one forward
        over the dual stream, reads text_logits at the forced positions.

        Returns an array aligned to `target_ids`: out[j] = logp(target_ids[j] |
        prefix, audio, target_ids[:j]). out[0] is predicted from the last prefix
        position (the assistant-start token), so it IS scored (unlike the
        encdec/declm paths whose out[0] is NaN); the consensus probe nan-filters
        and sums, so scoring the first token is correct and consistent."""
        prefix = self._kimi_prefix(audio)
        n_pre = len(prefix.text_token_ids)
        # Append forced transcript to the TEXT stream; blank the AUDIO stream at
        # the same positions (output_type="text" => audio output is blanked).
        audio_ids = list(prefix.audio_token_ids) + [self._kimi_text_blank] * len(target_ids)
        text_ids = list(prefix.text_token_ids) + list(target_ids)
        is_cont = list(prefix.is_continuous_mask) + [False] * len(target_ids)

        dev = torch.cuda.current_device()
        audio_t = torch.tensor([audio_ids], dtype=torch.long, device=dev)
        text_t = torch.tensor([text_ids], dtype=torch.long, device=dev)
        cont_mask = torch.tensor([is_cont], dtype=torch.bool, device=dev)
        pos_ids = torch.arange(0, audio_t.shape[1], device=dev).unsqueeze(0).long()
        feats = [f.to(dev) for f in prefix.continuous_feature]

        audio_logits, text_logits, _ = self.model.forward(
            input_ids=audio_t,
            text_input_ids=text_t,
            whisper_input_feature=feats,
            is_continuous_mask=cont_mask,
            position_ids=pos_ids,
            past_key_values=None,
            return_dict=False,
        )
        logits = text_logits[0].float()  # [T, V]; position t predicts text token t+1
        self._mask_eos(logits)
        lp = torch.log_softmax(logits, dim=-1)
        _em = torch.exp(lp[:, self.eos_ids]).sum(-1) if getattr(self, "_cap_eos", False) else None
        out = np.full(len(target_ids), np.nan)
        # Forced token j sits at sequence position (n_pre + j); it is predicted
        # by the logits at position (n_pre + j - 1).
        for j in range(len(target_ids)):
            pos = n_pre + j - 1
            if pos < 0 or pos >= lp.shape[0]:
                continue
            out[j] = float(lp[pos, target_ids[j]])
            if _em is not None:
                self._eos_aln[j] = float(_em[pos])
        return out

    # ---- omni (wav2vec2 enc + Llama dec, fairseq2 Seq2SeqBatch) helpers ----
    def _omni_text_ids(self, text: str) -> list[int]:
        """Bare target-text token ids for `text` (BOS/EOS stripped). The model's
        create_default_syntax wraps these with <bos> ... <eos> itself, so the
        teacher-forced target_seqs must NOT carry them. The fairseq2 SP encoder
        adds bos/eos, so strip any leading bos / trailing eos.

        The omniASR SentencePiece model is a *lowercase* Latin vocab: uppercase
        characters have no piece and encode to <unk> (⁇). ALL-CAPS references
        (LibriSpeech) or capitalized number words ("Two") would otherwise become
        all-<unk> and the masked number could never be located in decode(ids),
        skipping the clip. Omni's own greedy output is lowercase, so lowercasing
        the forced target is the faithful teacher-forcing target for this model."""
        text = text.lower()
        ids = self._omni_enc(text).tolist()
        if ids and ids[0] == self._omni_bos:
            ids = ids[1:]
        if ids and ids[-1] == self._omni_eos:
            ids = ids[:-1]
        return ids

    def _omni_token_logprobs(self, audio: np.ndarray, target_ids: list[int]) -> np.ndarray:
        """Per-token logprobs of the forced `target_ids` given `audio`.

        Build a single-example Seq2SeqBatch (audio source + forced text target +
        eng_Latn lang hint), run model.forward(return_logits=True), and read the
        target-span logits. The context (audio + lid_marker + lang_id + bos) has
        length `context_len = decoder_context_seq_lens[0][0]`; logits position
        (context_len - 1 + j) predicts target_ids[j] (same indexing the training
        loss uses in remove_context_logits). out[0] IS scored (predicted from the
        bos at the end of the context), like the kimi path."""
        import torch as _t

        m = self.model
        src = _t.tensor([audio], device=self.dev, dtype=self._omni_dtype)
        tgt = _t.tensor([target_ids], device=self.dev, dtype=_t.long)
        batch = self._omni_Seq2SeqBatch(
            source_seqs=src,
            source_seq_lens=[int(audio.shape[0])],
            target_seqs=tgt,
            target_seq_lens=[len(target_ids)],
            example={"lang": [self._omni_lang]},
        )
        out = m.forward(batch, return_logits=True)
        _loss, logits, _layout, _ctx_inputs, ctx_seq_lens, _audio_embs = out
        context_len = int(ctx_seq_lens[0][0])
        logits = logits[0].float()  # [S, V]
        self._mask_eos(logits)
        lp = _t.log_softmax(logits, dim=-1)
        _em = _t.exp(lp[:, self.eos_ids]).sum(-1) if getattr(self, "_cap_eos", False) else None
        out_arr = np.full(len(target_ids), np.nan)
        base = context_len - 1
        for j in range(len(target_ids)):
            pos = base + j
            if pos < 0 or pos >= lp.shape[0]:
                continue
            out_arr[j] = float(lp[pos, target_ids[j]])
            if _em is not None:
                self._eos_aln[j] = float(_em[pos])
        return out_arr

    # ---- target = greedy transcription of full audio ----
    def greedy_target(self, full_audio: np.ndarray):
        with torch.inference_mode():
            if self.family == "omni":
                # Transcribe via the pipeline (beam search), then re-encode the
                # cleaned text as forced target ids (natural SP tokenization).
                audio_in = [{"waveform": full_audio.astype(np.float32), "sample_rate": 16000}]
                texts = self._omni_pipeline.transcribe(audio_in, lang=[self._omni_lang], batch_size=1)
                text = (texts[0] if texts else "").strip()
                return self._omni_text_ids(text), "omni"
            if self.family == "kimi":
                # Greedy-transcribe via the kimia_infer text generate, then
                # re-encode the cleaned text as forced TEXT-stream ids (same
                # tokenizer => natural tokenization).
                import os
                import tempfile

                import soundfile as sf

                tmp = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(f.name, full_audio.astype(np.float32), 16000)
                        tmp = f.name
                    messages = [
                        {"role": "user", "message_type": "text", "content": self._kimi_instruction},
                        {"role": "user", "message_type": "audio", "content": tmp},
                    ]
                    _, text = self._kimi.generate(messages, output_type="text", max_new_tokens=256)
                finally:
                    if tmp is not None:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                answer_ids = self._pm._tokenize_text((text or "").strip())
                return answer_ids, "kimi"
            if self.family == "higgs":
                # Greedy-transcribe the full audio via the repo helper, then
                # re-encode the cleaned text as the forced answer ids (natural
                # Qwen tokenization). enable_thinking=False matches the prefix.
                text = self._higgs.transcribe(
                    self.model, self.tok, full_audio.astype(np.float32),
                    sample_rate=16000, user_prompt=getattr(self._higgs, "DEFAULT_PROMPT", "Transcribe the speech."),
                    enable_thinking=False, max_new_tokens=160,
                )
                text = (text or "").strip()
                answer_ids = self.tok.encode(text, add_special_tokens=False)
                return answer_ids, "higgs"
            if self.family == "salm":
                lens = torch.tensor([full_audio.shape[0]], dtype=torch.int64, device=self.dev)
                padded = torch.tensor([full_audio], dtype=torch.float32, device=self.dev)
                prompts = [[{"role": "user", "content": self._salm_user_msg}]]
                answer_ids = self.model.generate(
                    prompts=prompts, audios=padded, audio_lens=lens, max_new_tokens=160
                )
                text = self.tok.ids_to_text(answer_ids[0].cpu()).strip()
                ids, n_prompt = self._salm_prompt_ids(text)
                self._salm_n_prompt = n_prompt
                return ids, "salm"
            if self.family == "canary":
                hyps = self.model.transcribe(
                    [full_audio], batch_size=1, verbose=False,
                    source_lang="en", target_lang="en", task="asr", pnc="yes",
                )
                h = hyps[0]
                text = h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else str(h))
                ids, n_prompt = self._canary_prompt_ids(text)
                self._canary_n_prompt = n_prompt
                return ids, "canary"
            if self.is_whisper:
                inp = self.proc([full_audio], sampling_rate=16000, return_tensors="pt").to(
                    self.dev, dtype=self.model.dtype
                )
                forced = self.proc.get_decoder_prompt_ids(language=self.lang, task="transcribe")
                dsid = self.model.generation_config.decoder_start_token_id
                if self.family == "whisper-auto":
                    # lite-whisper's generation_config rejects the `language`/`task`
                    # kwargs; force the decoder prompt ids explicitly instead.
                    prompt = [dsid] + [t for _, t in forced]
                    dec = torch.tensor([prompt], device=self.dev)
                    out = self.model.generate(
                        **inp, decoder_input_ids=dec, max_new_tokens=160, do_sample=False
                    )[0]
                    ids = out.tolist()
                else:
                    out = self.model.generate(
                        **inp, language=self.lang, task="transcribe", max_new_tokens=160, do_sample=False
                    )[0]
                    ids = out.tolist()
                # whisper generate may return text-only; prepend the forced decoder prompt
                # so teacher-forcing has the right decoder context (verified: match 1.0, nll ~0).
                if not ids or ids[0] != dsid:
                    ids = [dsid] + [t for _, t in forced] + ids
                return ids, "encdec"
            if self.family == "moonshine":
                inp = self.proc([full_audio], sampling_rate=16000, return_tensors="pt").to(
                    self.dev, dtype=self.model.dtype
                )
                out = self.model.generate(**inp, max_new_tokens=160, do_sample=False)[0]
                return out.tolist(), "encdec"  # full seq incl decoder_start
            if self.arch == "encdec":
                inp = self.proc([full_audio], sampling_rate=16000, return_tensors="pt", language=self.lang).to(
                    self.dev, dtype=self.model.dtype
                )
                out = self.model.generate(**inp, max_new_tokens=160, do_sample=False)[0]
                return out.tolist(), "encdec"  # full seq incl prompt
            else:
                inp = self._declm_inputs(full_audio)
                plen = inp["input_ids"].shape[1]
                gen = self.model.generate(**inp, max_new_tokens=160, do_sample=False)
                seq = gen.sequences if hasattr(gen, "sequences") else gen  # qwen3 returns GenerateOutput
                out = seq[0]
                return out[plen:].tolist(), "declm"  # text tokens only

    # ---- forced ids for an ARBITRARY text (e.g. a reference / consensus seq) ----
    def ids_for_text(self, text: str) -> list[int]:
        """Family-appropriate teacher-forcing ids for a given transcript string.
        Mirrors greedy_target's per-family encoding but for supplied text, so we can
        score the NLL of the (erroneous) ref vs the consensus-corrected transcript."""
        text = (text or "").strip()
        if self.family == "omni":
            return self._omni_text_ids(text)
        if self.family == "salm":
            ids, n = self._salm_prompt_ids(text)
            self._salm_n_prompt = n
            return ids
        if self.family == "canary":
            ids, n = self._canary_prompt_ids(text)
            self._canary_n_prompt = n
            return ids
        if self.family == "cohere":
            if not hasattr(self, "_cohere_pre"):
                z = np.zeros(16000, dtype=np.float32)
                pin = self.proc([z], sampling_rate=16000, return_tensors="pt", language=self.lang)
                self._cohere_pre = pin["decoder_input_ids"][0].tolist()
            return self._cohere_pre + self.tok.encode(text, add_special_tokens=False)
        if self.is_whisper:
            forced = self.proc.get_decoder_prompt_ids(language=self.lang, task="transcribe")
            dsid = self.model.generation_config.decoder_start_token_id
            return [dsid] + [t for _, t in forced] + self.tok.encode(text, add_special_tokens=False)
        if self.family == "moonshine":
            dsid = self.model.generation_config.decoder_start_token_id
            return [dsid] + self.tok.encode(text, add_special_tokens=False)
        if self.family == "kimi":
            # Use the prompt_manager's own text tokenizer (custom encode signature
            # bos=False/eos=False) so the forced ids match the model's tokenization.
            return self._pm._tokenize_text(text)
        # declm (granite/voxtral/qwen3/phi4) + higgs: text tokens only (prompt via _declm_inputs)
        return self.tok.encode(text, add_special_tokens=False)

    def _mask_eos(self, logits: torch.Tensor) -> None:
        """In-place: remove EOS from the distribution so log-probs are computed
        conditional on the model continuing. Truncated audio otherwise dumps mass
        on EOS (the clip ended), which deflates every continuation token."""
        if self.mask_eos:
            logits[..., self.eos_ids] = float("-inf")

    # ---- per-token logprobs of target_ids given an audio condition ----
    def token_logprobs(self, audio: np.ndarray, target_ids: list[int]) -> np.ndarray:
        with torch.inference_mode():
            if self.family == "omni":
                return self._omni_token_logprobs(audio, target_ids)
            if self.family == "kimi":
                return self._kimi_token_logprobs(audio, target_ids)
            if self.family == "higgs":
                return self._higgs_token_logprobs(audio, target_ids)
            if self.family == "salm":
                return self._salm_token_logprobs(audio, target_ids, self._salm_n_prompt)
            if self.family == "canary":
                sig = torch.tensor([audio], device=self.dev, dtype=torch.float32)
                sig_len = torch.tensor([audio.shape[0]], device=self.dev)
                tgt = torch.tensor([target_ids], device=self.dev)
                tgt_len = torch.tensor([len(target_ids)], device=self.dev)
                log_probs, _, _, _ = self.model.forward(
                    input_signal=sig, input_signal_length=sig_len,
                    transcript=tgt, transcript_length=tgt_len,
                )
                lp = log_probs[0].float()  # [T, V], already log-softmaxed
                if self.mask_eos:
                    # Renormalize the distribution with EOS removed:
                    # lp' = lp - log(1 - sum_eos exp(lp_eos)). Matches the
                    # "model must continue" condition used for other models.
                    eos_lp = lp[:, self.eos_ids]  # [T, n_eos]
                    eos_mass = torch.exp(eos_lp).sum(dim=-1).clamp(max=1 - 1e-6)
                    lp = lp - torch.log1p(-eos_mass).unsqueeze(-1)
                    lp[:, self.eos_ids] = float("-inf")
                out = np.full(len(target_ids), np.nan)
                for j in range(1, len(target_ids)):
                    out[j] = float(lp[j - 1, target_ids[j]])
                return out
            if self.arch == "encdec":
                enc = self._encdec_enc(audio)
                dec = torch.tensor([target_ids], device=self.dev)
                logits = self.model(encoder_outputs=enc, decoder_input_ids=dec).logits[0]
                logits = logits[:-1].float()  # predicts token t+1
                self._mask_eos(logits)
                lp = torch.log_softmax(logits, dim=-1)
                out = np.full(len(target_ids), np.nan)
                for j in range(1, len(target_ids)):
                    out[j] = float(lp[j - 1, target_ids[j]])
                return out
            else:
                inp = self._declm_inputs(audio)
                plen = inp["input_ids"].shape[1]
                tgt = torch.tensor([target_ids], device=self.dev)
                full = torch.cat([inp["input_ids"], tgt], dim=1)
                kwargs = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
                attn = torch.ones_like(full)
                logits = self.model(input_ids=full, attention_mask=attn, **kwargs).logits[0].float()
                self._mask_eos(logits)
                lp = torch.log_softmax(logits, dim=-1)
                out = np.full(len(target_ids), np.nan)
                for j in range(len(target_ids)):
                    out[j] = float(lp[plen + j - 1, target_ids[j]])
                return out

    def eos_mass_aligned(self, audio: np.ndarray, ids: list[int]) -> np.ndarray:
        """Per-position EOS probability mass aligned to `ids`, for the custom-forward
        families (salm/kimi/omni) whose teacher-forcing the generic encdec/declm
        EOS-mass reader cannot reach. Measured with mask_eos OFF so the true mass is
        read (the lift path's per-family methods otherwise zero EOS). Returns NaN at
        positions the family does not score, matching token_logprobs' alignment."""
        prev = self.mask_eos
        self.mask_eos = False
        self._cap_eos = True
        self._eos_aln = np.full(len(ids), np.nan)
        try:
            with torch.inference_mode():
                if self.family == "salm":
                    self._salm_token_logprobs(audio, ids, self._salm_n_prompt)
                elif self.family == "kimi":
                    self._kimi_token_logprobs(audio, ids)
                elif self.family == "omni":
                    self._omni_token_logprobs(audio, ids)
                else:
                    raise ValueError(f"eos_mass_aligned: unsupported family {self.family}")
        finally:
            self._cap_eos = False
            self.mask_eos = prev
        return self._eos_aln

    def decode(self, ids: list[int]) -> str:
        if self.family == "omni":
            return self._omni_dec(torch.tensor(ids, dtype=torch.long))
        if self.family == "kimi":
            # Drop the text-stream control tokens (kimia_text_eos / blank) the
            # detokenizer skips, then decode with the text tokenizer.
            valid = [t for t in ids if t != self._kimi_xt.kimia_text_eos and t != self._kimi_text_blank]
            return self.tok.decode(valid)
        if self.family == "salm":
            # NeMo AutoTokenizer; remove_special_tokens drops the <|im_start|>/<|im_end|>
            # chat scaffolding and the audio_locator_tag so char-span matching in
            # hidden_token_indices sees clean transcript text.
            txt = self.tok.ids_to_text(ids, remove_special_tokens=True)
            return re.sub(r"<\|[^|]*\|>", "", txt)
        if self.family == "canary":
            # NeMo tokenizer: ids_to_text keeps <|...|> control tokens; strip them
            # so char-span matching in hidden_token_indices sees clean text.
            txt = self.tok.ids_to_text(ids)
            return re.sub(r"<\|[^|]*\|>", "", txt)
        return self.tok.decode(ids, skip_special_tokens=True)

    # ---- EOS-ban forced continuation on truncated audio ----
    def eos_ban_generate(self, trunc_audio: np.ndarray, max_new: int = 96) -> str:
        with torch.inference_mode():
            if self.family == "omni":
                # The fairseq2 beam-search loop has no HF logits_processor hook, so
                # EOS-ban continuation isn't wired. The consensus-perplexity path
                # (ids_for_text + token_logprobs) is fully supported.
                raise NotImplementedError("eos_ban_generate not implemented for omni (perplexity path only)")
            lp = LogitsProcessorList([BanEOS(self.eos_ids)])
            if self.family == "kimi":
                # The kimia_infer text generate is a custom dual-stream loop with
                # no HF logits_processor hook, so EOS-ban continuation isn't wired
                # here. The consensus-perplexity probe (ids_for_text +
                # token_logprobs) is fully supported; only the generative EOS-ban
                # leak signal is unavailable for kimi.
                raise NotImplementedError("eos_ban_generate not implemented for kimi (perplexity path only)")
            if self.family == "higgs":
                # EOS-ban continuation via the repo collator (no forced answer).
                batch, _ = self._higgs_collate(trunc_audio, [])
                plen = batch["input_ids"].shape[1]
                out = self.model.generate(
                    **batch, max_new_tokens=max_new, use_cache=True,
                    do_sample=False, logits_processor=lp, tokenizer=self.tok,
                )
                seq = out[0] if isinstance(out, tuple) else out
                return self.decode(seq[0, plen:].tolist())
            if self.family == "salm":
                lens = torch.tensor([trunc_audio.shape[0]], dtype=torch.int64, device=self.dev)
                padded = torch.tensor([trunc_audio], dtype=torch.float32, device=self.dev)
                prompts = [[{"role": "user", "content": self._salm_user_msg}]]
                out = self.model.generate(
                    prompts=prompts, audios=padded, audio_lens=lens,
                    max_new_tokens=max_new, do_sample=False, logits_processor=lp,
                )
                return self.tok.ids_to_text(out[0].cpu(), remove_special_tokens=True).strip()
            if self.arch == "encdec":
                inp = self.proc([trunc_audio], sampling_rate=16000, return_tensors="pt", language="en").to(
                    self.dev, dtype=self.model.dtype
                )
                out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False, logits_processor=lp)
                return self.decode(out[0].tolist())
            else:
                inp = self._declm_inputs(trunc_audio)
                plen = inp["input_ids"].shape[1]
                out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False, logits_processor=lp)
                return self.decode(out[0, plen:].tolist())


def hidden_token_indices(probe: Probe, ids: list[int], hidden_words: list[str], side: str = "tail") -> list[int]:
    full_text = probe.decode(ids)
    low = full_text.lower()
    base = " ".join(w.lower() for w in hidden_words)
    # candidate surface forms: the reference phrase, plus its normalized (digit) form,
    # so a model that decodes a spelled number as digits (e.g. "two"->"2") still matches.
    cands = [base]
    try:
        from benchmark_optimization.normalize import normalize
        nf = normalize(base, "en").strip()
        if nf and nf != base:
            cands.append(nf)
    except Exception:
        pass
    for phrase in cands:
        # head/span-hidden words: first occurrence; tail-hidden: last occurrence.
        start = low.find(phrase) if side in ("head", "span") else low.rfind(phrase)
        if start < 0:
            continue
        end = start + len(phrase)
        prev, out = "", []
        for i in range(len(ids)):
            cur = probe.decode(ids[: i + 1])
            a, b = len(prev), len(cur)
            if b > start and a < end and b > a:
                out.append(i)
            prev = cur
        if out:
            return out
    return []


def phantom_token_indices(probe: Probe, ids: list[int], phantom_words: list[str], side: str) -> list[int]:
    """Token-level locator for the phantom probe (head/tail spans).

    Unlike `hidden_token_indices` (which substring-matches a normalized phrase
    against `decode(ids)` and fails when the tokenizer's decode does not
    reproduce the phrase verbatim — digit runs split by BPE, casing, the
    ref_raw↔ref_tokens normalization gap), this maps the phantom *word
    positions* directly. By construction the phantom run is the leading (head)
    or trailing (tail) K content words of the teacher-forced target, so we:

      1. decode(ids) with specials stripped → content-only string,
      2. take the first / last K whitespace words of that string as the phantom
         char span (robust to digit segmentation / leading-space markers / case),
      3. select every subword token whose cumulative-decoded char span overlaps
         that phantom char span.

    Requires that `ids` were built by encoding the space-joined normalized
    `ref_tokens` (so decode(ids) yields exactly the normalized words), which is
    what `phantom_target_ids` in the phantom probe does. Returns [] if the
    decoded string has fewer than K words (caller treats as a skip).
    """
    n_ph = len(phantom_words)
    if n_ph == 0:
        return []
    full = probe.decode(ids)
    word_iter = list(re.finditer(r"\S+", full))
    if len(word_iter) < n_ph:
        return []
    target = word_iter[:n_ph] if side in ("head", "start") else word_iter[-n_ph:]
    cs, ce = target[0].start(), target[-1].end()
    # cumulative-decode char span of each subword token, select overlaps
    prev, out = "", []
    for i in range(len(ids)):
        cur = probe.decode(ids[: i + 1])
        a, b = len(prev), len(cur)
        if b > cs and a < ce and b > a:
            out.append(i)
        prev = cur
    return out


def phantom_token_indices_span(probe: Probe, ids: list[int], word_lo: int, word_hi: int) -> list[int]:
    """Interior (delete-MIDDLE) locator: select subword tokens of the ref-token
    span [word_lo, word_hi] (inclusive word indices into the space-joined
    normalized ref_tokens that built `ids`).

    The head/tail locator anchors on the first/last K words of the decoded
    string, which is wrong for an interior span. Here the span sits at known
    WORD positions (the consensus run_indices index into ref_tokens, and
    ref_text == " ".join(ref_tokens)), so we:

      1. decode(ids) with specials stripped -> content-only string,
      2. take whitespace words [word_lo .. word_hi] as the phantom char span,
      3. select every subword token whose cumulative-decoded char span overlaps
         that span.

    Returns [] if the decoded string has fewer words than word_hi+1 (caller
    treats as a skip — e.g. a tokenizer that merges/drops words).
    """
    full = probe.decode(ids)
    word_iter = list(re.finditer(r"\S+", full))
    if word_hi >= len(word_iter) or word_lo < 0 or word_lo > word_hi:
        return []
    cs, ce = word_iter[word_lo].start(), word_iter[word_hi].end()
    prev, out = "", []
    for i in range(len(ids)):
        cur = probe.decode(ids[: i + 1])
        a, b = len(prev), len(cur)
        if b > cs and a < ce and b > a:
            out.append(i)
        prev = cur
    return out


def gen_leak(cont_text: str, hidden_words: list[str]) -> float:
    cont = set(re.findall(r"[a-z']+", cont_text.lower()))
    content = [w.lower().strip(".,!?;:") for w in hidden_words]
    content = [w for w in content if w and w not in _STOP]
    if not content:
        return float("nan")
    return sum(w in cont for w in content) / len(content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen-limit", type=int, default=120, help="run EOS-ban gen on first N samples")
    ap.add_argument("--target-mode", choices=["greedy", "ref"], default="greedy",
                    help="teacher-forcing target: 'greedy' = model's own transcription (n varies per model); "
                         "'ref' = the reference transcript (number always present -> constant n across models).")
    ap.add_argument("--no-mask-eos", action="store_true", help="disable EOS masking in perplexity scoring")
    args = ap.parse_args()

    probe = Probe(args.model)
    probe.mask_eos = not args.no_mask_eos
    log.info("EOS masking in perplexity: %s (eos_ids=%s)", probe.mask_eos, probe.eos_ids)
    split_dir = Path(_paths.DATASETS_ROOT) / args.variant / args.split
    meta = pl.read_parquet(split_dir / "truncation_meta.parquet")
    if args.limit:
        meta = meta.head(args.limit)
    full_split = Path(_paths.DATASETS_ROOT) / args.source / args.split
    fm = pl.read_parquet(full_split / "manifest.parquet", columns=["__key__", "path"])
    path_by_key = dict(zip(fm["__key__"], fm["path"]))

    rows, skipped = [], 0
    for i, m in enumerate(meta.iter_rows(named=True)):
        key = m["__key__"]
        if key not in path_by_key:
            skipped += 1
            continue
        side = m.get("hidden_side", "tail")
        hidden_ref = m.get("hidden_ref")
        hidden_words = hidden_ref.split() if hidden_ref else m["full_ref"].split()[m["n_ref_words_kept"]:]
        if not hidden_words:
            skipped += 1
            continue
        full_wav = load_wav(full_split / path_by_key[key])
        trunc_wav = load_wav(split_dir / "wavs" / "en" / f"{key}.wav")
        none_wav = np.zeros_like(trunc_wav)

        if args.target_mode == "ref":
            # Force the REFERENCE transcript as the teacher-forcing target so the
            # masked number is present for EVERY clip regardless of what the model
            # would greedily emit -> constant n across models, includes dropped numbers.
            ref_text = (m.get("full_ref") or "").strip()
            target_ids = probe.ids_for_text(ref_text) if ref_text else []
        else:
            target_ids, _ = probe.greedy_target(full_wav)
        hid = hidden_token_indices(probe, target_ids, hidden_words, side=side)
        if not hid:
            skipped += 1
            continue

        lp_full = probe.token_logprobs(full_wav, target_ids)
        lp_trunc = probe.token_logprobs(trunc_wav, target_ids)
        lp_none = probe.token_logprobs(none_wav, target_ids)
        nll_full = float(-np.nanmean(lp_full[hid]))
        nll_trunc = float(-np.nanmean(lp_trunc[hid]))
        nll_none = float(-np.nanmean(lp_none[hid]))
        denom = nll_none - nll_full
        recovery = (nll_none - nll_trunc) / denom if abs(denom) > 1e-6 else float("nan")

        leak = float("nan")
        cont = ""
        if i < args.gen_limit and side == "tail":  # EOS-ban continuation only meaningful for tail-hidden
            cont = probe.eos_ban_generate(trunc_wav)
            leak = gen_leak(cont, hidden_words)

        rows.append({
            "model": args.model, "variant": args.variant, "__key__": key,
            "hidden_words": " ".join(hidden_words), "n_hidden_tokens": len(hid),
            "nll_full": round(nll_full, 4), "nll_trunc": round(nll_trunc, 4),
            "nll_none": round(nll_none, 4), "recovery": round(recovery, 4),
            "gen_leak": leak, "eos_ban_cont": cont,
        })
        if len(rows) % 25 == 0:
            log.info("scored %d (skipped %d)", len(rows), skipped)

    df = pl.DataFrame(rows, strict=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    log.info("wrote %s (%d rows, skipped %d)", out, df.height, skipped)
    if df.height:
        rc = df["recovery"].drop_nans().drop_nulls().clip(-1, 2)
        gl = df["gen_leak"].drop_nans().drop_nulls()
        print(f"=== {args.model} on {args.variant} ===")
        print(f"n={df.height}")
        print(f"nll_full={df['nll_full'].mean():.3f} nll_none={df['nll_none'].mean():.3f} nll_trunc={df['nll_trunc'].mean():.3f}")
        print(f"recovery: median={df['recovery'].drop_nans().median():.3f} mean(clip)={rc.mean():.3f} "
              f"frac>0.5={(df['recovery'].drop_nans()>0.5).mean():.3f}")
        if gl.len():
            print(f"eos-ban gen_leak (n={gl.len()}): mean={gl.mean():.3f} frac==1.0={(gl>=0.999).mean():.3f}")
        else:
            print("eos-ban gen_leak: n/a (head mode — generation skipped)")


if __name__ == "__main__":
    main()
