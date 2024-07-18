#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import json
import time
import copy
import torch
import random
import string
import logging
import os.path
import numpy as np
from tqdm import tqdm

from funasr.utils.misc import deep_update
from funasr.register import tables
from funasr.utils.load_utils import load_bytes
from funasr.download.file import download_from_url
from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.utils.timestamp_tools import timestamp_sentence_en
from funasr.download.download_from_hub import download_model
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.utils.vad_utils import merge_vad
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils import export_utils
from funasr.utils import misc

try:
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend
except:
    pass


def prepare_data_iterator(data_in, input_len=None, data_type=None, key=None):
    """ """
    data_list = []
    key_list = []
    filelist = [".scp", ".txt", ".json", ".jsonl", ".text"]

    chars = string.ascii_letters + string.digits
    if isinstance(data_in, str):
        if data_in.startswith("http://") or data_in.startswith("https://"):  # url
            data_in = download_from_url(data_in)

    if isinstance(data_in, str) and os.path.exists(
        data_in
    ):  # wav_path; filelist: wav.scp, file.jsonl;text.txt;
        _, file_extension = os.path.splitext(data_in)
        file_extension = file_extension.lower()
        if file_extension in filelist:  # filelist: wav.scp, file.jsonl;text.txt;
            with open(data_in, encoding="utf-8") as fin:
                for line in fin:
                    key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                    if data_in.endswith(".jsonl"):  # file.jsonl: json.dumps({"source": data})
                        lines = json.loads(line.strip())
                        data = lines["source"]
                        key = data["key"] if "key" in data else key
                    else:  # filelist, wav.scp, text.txt: id \t data or data
                        lines = line.strip().split(maxsplit=1)
                        data = lines[1] if len(lines) > 1 else lines[0]
                        key = lines[0] if len(lines) > 1 else key

                    data_list.append(data)
                    key_list.append(key)
        else:
            if key is None:
                # key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                key = misc.extract_filename_without_extension(data_in)
            data_list = [data_in]
            key_list = [key]
    elif isinstance(data_in, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):  # mutiple inputs
            data_list_tmp = []
            for data_in_i, data_type_i in zip(data_in, data_type):
                key_list, data_list_i = prepare_data_iterator(
                    data_in=data_in_i, data_type=data_type_i
                )
                data_list_tmp.append(data_list_i)
            data_list = []
            for item in zip(*data_list_tmp):
                data_list.append(item)
        else:
            # [audio sample point, fbank, text]
            data_list = data_in
            key_list = []
            for data_i in data_in:
                if isinstance(data_i, str) and os.path.exists(data_i):
                    key = misc.extract_filename_without_extension(data_i)
                else:
                    key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                key_list.append(key)

    else:  # raw text; audio sample point, fbank; bytes
        if isinstance(data_in, bytes):  # audio bytes
            data_in = load_bytes(data_in)
        if key is None:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
        data_list = [data_in]
        key_list = [key]

    return key_list, data_list


class AutoModel:

    def __init__(self, **kwargs):

        log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
        logging.basicConfig(level=log_level)

        if not kwargs.get("disable_log", True):
            tables.print()

        model, kwargs = self.build_model(**kwargs)

        # if vad_model is not None, build vad model else None
        vad_model = kwargs.get("vad_model", None)
        vad_kwargs = {} if kwargs.get("vad_kwargs", {}) is None else kwargs.get("vad_kwargs", {})
        if vad_model is not None:
            logging.info("Building VAD model.")
            vad_kwargs["model"] = vad_model
            vad_kwargs["model_revision"] = kwargs.get("vad_model_revision", "master")
            vad_kwargs["device"] = kwargs["device"]
            vad_model, vad_kwargs = self.build_model(**vad_kwargs)

        # if punc_model is not None, build punc model else None
        punc_model = kwargs.get("punc_model", None)
        punc_kwargs = {} if kwargs.get("punc_kwargs", {}) is None else kwargs.get("punc_kwargs", {})
        if punc_model is not None:
            logging.info("Building punc model.")
            punc_kwargs["model"] = punc_model
            punc_kwargs["model_revision"] = kwargs.get("punc_model_revision", "master")
            punc_kwargs["device"] = kwargs["device"]
            punc_model, punc_kwargs = self.build_model(**punc_kwargs)

        # if spk_model is not None, build spk model else None
        spk_model = kwargs.get("spk_model", None)
        spk_kwargs = {} if kwargs.get("spk_kwargs", {}) is None else kwargs.get("spk_kwargs", {})
        if spk_model is not None:
            logging.info("Building SPK model.")
            spk_kwargs["model"] = spk_model
            spk_kwargs["model_revision"] = kwargs.get("spk_model_revision", "master")
            spk_kwargs["device"] = kwargs["device"]
            spk_model, spk_kwargs = self.build_model(**spk_kwargs)
            self.cb_model = ClusterBackend().to(kwargs["device"])
            spk_mode = kwargs.get("spk_mode", "punc_segment")
            if spk_mode not in ["default", "vad_segment", "punc_segment"]:
                logging.error("spk_mode should be one of default, vad_segment and punc_segment.")
            self.spk_mode = spk_mode

        self.kwargs = kwargs
        self.model = model
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs
        self.punc_model = punc_model
        self.punc_kwargs = punc_kwargs
        self.spk_model = spk_model
        self.spk_kwargs = spk_kwargs
        self.model_path = kwargs.get("model_path")

    def build_model(self, **kwargs):
        assert "model" in kwargs
        if "model_conf" not in kwargs:
            logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
            kwargs = download_model(**kwargs)

        set_all_random_seed(kwargs.get("seed", 0))

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"
            kwargs["batch_size"] = 1
        kwargs["device"] = device

        torch.set_num_threads(kwargs.get("ncpu", 4))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

        # build frontend
        frontend = kwargs.get("frontend", None)
        kwargs["input_size"] = None
        if frontend is not None:
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = (
                frontend.output_size() if hasattr(frontend, "output_size") else None
            )
        kwargs["frontend"] = frontend
        # build model
        model_class = tables.model_classes.get(kwargs["model"])
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf, vocab_size=vocab_size)
        model.to(device)

        # init_param
        init_param = kwargs.get("init_param", None)
        if init_param is not None:
            if os.path.exists(init_param):
                logging.info(f"Loading pretrained params from {init_param}")
                load_pretrained_model(
                    model=model,
                    path=init_param,
                    ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                    oss_bucket=kwargs.get("oss_bucket", None),
                    scope_map=kwargs.get("scope_map", []),
                    excludes=kwargs.get("excludes", None),
                )
            else:
                print(f"error, init_param does not exist!: {init_param}")

        # fp16
        if kwargs.get("fp16", False):
            model.to(torch.float16)
        elif kwargs.get("bf16", False):
            model.to(torch.bfloat16)
        return model, kwargs

    def __call__(self, *args, **cfg):
        kwargs = self.kwargs
        deep_update(kwargs, cfg)
        res = self.model(*args, kwargs)
        return res

    def generate(self, input, input_len=None, **cfg):
        if self.vad_model is None:
            return self.inference(input, input_len=input_len, **cfg)

        else:
            return self.inference_with_vad(input, input_len=input_len, **cfg)

    def inference(self, input, input_len=None, model=None, kwargs=None, key=None, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        deep_update(kwargs, cfg)
        model = self.model if model is None else model
        model.eval()

        batch_size = kwargs.get("batch_size", 1)
        # if kwargs.get("device", "cpu") == "cpu":
        #     batch_size = 1

        key_list, data_list = prepare_data_iterator(
            input, input_len=input_len, data_type=kwargs.get("data_type", None), key=key
        )

        speed_stats = {}
        asr_result_list = []
        num_samples = len(data_list)
        disable_pbar = self.kwargs.get("disable_pbar", False)
        pbar = (
            tqdm(colour="blue", total=num_samples, dynamic_ncols=True) if not disable_pbar else None
        )
        time_speech_total = 0.0
        time_escape_total = 0.0
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}

            if (end_idx - beg_idx) == 1 and kwargs.get("data_type", None) == "fbank":  # fbank
                batch["data_in"] = data_batch[0]
                batch["data_lengths"] = input_len

            time1 = time.perf_counter()
            with torch.no_grad():
                res = model.inference(**batch, **kwargs)
                if isinstance(res, (list, tuple)):
                    results = res[0] if len(res) > 0 else [{"text": ""}]
                    meta_data = res[1] if len(res) > 1 else {}
            time2 = time.perf_counter()

            asr_result_list.extend(results)

            # batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
            batch_data_time = meta_data.get("batch_data_time", -1)
            time_escape = time2 - time1
            speed_stats["load_data"] = meta_data.get("load_data", 0.0)
            speed_stats["extract_feat"] = meta_data.get("extract_feat", 0.0)
            speed_stats["forward"] = f"{time_escape:0.3f}"
            speed_stats["batch_size"] = f"{len(results)}"
            speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
            description = f"{speed_stats}, "
            if pbar:
                pbar.update(1)
                pbar.set_description(description)
            time_speech_total += batch_data_time
            time_escape_total += time_escape

        if pbar:
            # pbar.update(1)
            pbar.set_description(f"rtf_avg: {time_escape_total/time_speech_total:0.3f}")
        torch.cuda.empty_cache()
        return asr_result_list

    def inference_with_vad(self, input, input_len=None, **cfg):
        kwargs = self.kwargs
        # step.1: compute the vad model
        deep_update(self.vad_kwargs, cfg)
        beg_vad = time.time()
        res = self.inference(
            input, input_len=input_len, model=self.vad_model, kwargs=self.vad_kwargs, **cfg
        )
        end_vad = time.time()
            
        #  FIX(gcf): concat the vad clips for sense vocie model for better aed
        if kwargs.get("merge_vad", False):
            for i in range(len(res)):
                res[i]["value"] = merge_vad(res[i]["value"], kwargs.get("merge_length", 15000))

        # step.2 compute asr model
        model = self.model
        deep_update(kwargs, cfg)
        batch_size = max(int(kwargs.get("batch_size_s", 300)) * 1000, 1)
        batch_size_threshold_ms = int(kwargs.get("batch_size_threshold_s", 60)) * 1000
        kwargs["batch_size"] = batch_size

        key_list, data_list = prepare_data_iterator(
            input, input_len=input_len, data_type=kwargs.get("data_type", None)
        )
        results_ret_list = []
        time_speech_total_all_samples = 1e-6

        beg_total = time.time()
        pbar_total = (
            tqdm(colour="red", total=len(res), dynamic_ncols=True)
            if not kwargs.get("disable_pbar", False)
            else None
        )

        all_speech = []
        max_len_in_batch = 0
        for i in range(len(res)):
            key = res[i]["key"]
            vadsegments = res[i]["value"]
            input_i = data_list[i]
            fs = kwargs["frontend"].fs if hasattr(kwargs["frontend"], "fs") else 16000
            speech = load_audio_text_image_video(input_i, fs=fs, audio_fs=kwargs.get("fs", 16000))
            speech_lengths = len(speech)
            n = len(vadsegments)
            data_with_index = [(vadsegments[i], i) for i in range(n)]

            speech_j, _ = slice_padding_audio_samples(
                speech, speech_lengths, data_with_index
            )
            assert len(speech_j) == n
            for speech in speech_j:
                max_len_in_batch = max(max_len_in_batch, len(speech))
            all_speech.extend(speech_j)

        if max_len_in_batch == 0:  # all silence, return empty result
            return [{"key": "", "text": "", "timestamp": []}]

        all_result = []
        stride = batch_size_threshold_ms * 16 // max_len_in_batch
        for i in range(0, len(all_speech), stride):
            batch_speech = all_speech[i:min(i + stride, len(all_speech))]
            results = self.inference(
                batch_speech, input_len=None, model=model, kwargs=kwargs, **cfg
            )
            num_results = len(results)
            if len(batch_speech) != num_results:
                assert len(batch_speech) >= num_results
                for _ in range(len(batch_speech) - num_results):
                    results.append({"key": "", "text": "", "timestamp": []})
            all_result.extend(results)

        assert len(all_result) == sum([len(r['value']) for r in res])
        beg_idx = 0
        for i in range(len(res)):
            key = res[i]["key"]
            vadsegments = res[i]["value"]
            n = len(vadsegments)
            sentence_result = {}

            for j, segment_result in enumerate(all_result[beg_idx:beg_idx + n]):
                for k, v in segment_result.items():
                    if k.startswith("timestamp"):
                        if k not in sentence_result:
                            sentence_result[k] = []
                        for t in segment_result[k]:
                            t[0] += vadsegments[j][0]
                            t[1] += vadsegments[j][0]
                        sentence_result[k].extend(segment_result[k])
                    elif k == "spk_embedding":
                        if k not in sentence_result:
                            sentence_result[k] = segment_result[k]
                        else:
                            sentence_result[k] = torch.cat([sentence_result[k], segment_result[k]], dim=0)
                    elif "text" in k:
                        if k not in sentence_result:
                            sentence_result[k] = segment_result[k]
                        else:
                            sentence_result[k] += " " + segment_result[k]
                    else:
                        if k not in sentence_result:
                            sentence_result[k] = segment_result[k]
                        else:
                            sentence_result[k] += segment_result[k]

            return_raw_text = kwargs.get("return_raw_text", False)
            if self.punc_model is not None:
                if not len(sentence_result["text"].strip()):
                    if return_raw_text:
                        sentence_result["raw_text"] = ""
                else:
                    deep_update(self.punc_kwargs, cfg)
                    punc_res = self.inference(
                        sentence_result["text"], model=self.punc_model, kwargs=self.punc_kwargs, **cfg
                    )
                    raw_text = copy.copy(sentence_result["text"])
                    if return_raw_text:
                        sentence_result["raw_text"] = raw_text
                    sentence_result["text"] = punc_res[0]["text"]
            else:
                raw_text = None

            # TODO(xcsong): add spk_model
            sentence_result["key"] = key
            results_ret_list.append(sentence_result)
            beg_idx += n

        # end_total = time.time()
        # time_escape_total_all_samples = end_total - beg_total
        # print(f"rtf_avg_all: {time_escape_total_all_samples / time_speech_total_all_samples:0.3f}, "
        #                      f"time_speech_all: {time_speech_total_all_samples: 0.3f}, "
        #                      f"time_escape_all: {time_escape_total_all_samples:0.3f}")
        return results_ret_list

    def export(self, input=None, **cfg):
        """

        :param input:
        :param type:
        :param quantize:
        :param fallback_num:
        :param calib_num:
        :param opset_version:
        :param cfg:
        :return:
        """

        device = cfg.get("device", "cpu")
        model = self.model.to(device=device)
        kwargs = self.kwargs
        deep_update(kwargs, cfg)
        kwargs["device"] = device
        del kwargs["model"]
        model.eval()

        type = kwargs.get("type", "onnx")

        key_list, data_list = prepare_data_iterator(
            input, input_len=None, data_type=kwargs.get("data_type", None), key=None
        )

        with torch.no_grad():

            if type == "onnx":
                export_dir = export_utils.export_onnx(model=model, data_in=data_list, **kwargs)
            else:
                export_dir = export_utils.export_torchscripts(
                    model=model, data_in=data_list, **kwargs
                )

        return export_dir
