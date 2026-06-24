import ray
import uuid
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    process_validation_metrics,
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.trainer.ppo.tool_adaptation import inject_tools_to_batch
from verl.utils.debug import marked_timer


##############################################################################
#### Replace the original classes/functions with verl-tool customized ones ####
import verl.experimental.agent_loop
from verl_tool.agent_loop import AgentLoopManager
import verl.trainer.ppo.ray_trainer
from .reward import compute_reward, compute_reward_async
from verl_tool.workers.rollout.vllm_rollout.vllm_async_server import VerlToolvLLMHttpServer
import verl.workers.rollout.vllm_rollout.vllm_async_server
from .metric_util import compute_data_metrics, process_validation_metrics
verl.experimental.agent_loop.AgentLoopManager = AgentLoopManager
verl.trainer.ppo.ray_trainer.compute_reward = compute_reward
verl.trainer.ppo.ray_trainer.compute_reward_async = compute_reward_async
verl.trainer.ppo.ray_trainer.compute_data_metrics = compute_data_metrics
verl.trainer.ppo.ray_trainer.process_validation_metrics = process_validation_metrics
verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer = VerlToolvLLMHttpServer
##############################################################################


def _normalize_dump_list(value, expected_len: int) -> list | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == expected_len:
        return value
    return None


def _build_validation_dump_reward_infos(
    reward_extra_infos_dict: dict[str, list],
    data_source_lst: list,
    expected_len: int,
) -> dict[str, list]:
    reward_extra_infos_to_dump = dict(reward_extra_infos_dict)
    if data_source_lst:
        data_sources = np.concatenate(data_source_lst, axis=0)
        normalized_data_sources = _normalize_dump_list(data_sources, expected_len)
        if normalized_data_sources is not None:
            reward_extra_infos_to_dump.setdefault("data_source", normalized_data_sources)
    return reward_extra_infos_to_dump


class AgentRayPPOTrainer(RayPPOTrainer):
    
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        val_only_mode = bool(self.config.trainer.get("val_only", False))
        # Validation-only runs should keep the same skill adaptation behavior as
        # training, while remaining fully independent from train updates.

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        pending_skill_paths_to_cleanup: list[str] = []

        for val_batch_idx, test_data in enumerate(self.val_dataloader, start=1):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            tools_prompt = self.tool_adaptation_manager.get_tools_prompt_for_validation()
            if tools_prompt:
                test_batch = inject_tools_to_batch(
                    test_batch,
                    tools_prompt,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_prompt_length=self.config.data.max_prompt_length,
                    truncation=self.config.data.get("truncation", "left"),
                    apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_attention_mask = test_output_gen_batch.batch["attention_mask"][:, test_output_gen_batch.batch["prompts"].shape[1]:]
            output_texts = [self.tokenizer.decode(ids[output_attention_mask[i]==1], skip_special_tokens=False) for i, ids in enumerate(output_ids)]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch.meta_info["skill_names"] = self.tool_adaptation_manager.get_current_skill_names()
            test_batch.meta_info["global_steps"] = self.global_steps

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = None
            if "reward_extra_info" in result:
                reward_extra_info = result["reward_extra_info"]
                if isinstance(reward_extra_info, dict):
                    batch_new_skills = reward_extra_info.get("batch_new_skills", {})
                    if isinstance(batch_new_skills, dict):
                        for info in batch_new_skills.values():
                            if not isinstance(info, dict):
                                continue
                            path_str = info.get("path")
                            if isinstance(path_str, str) and path_str.strip():
                                pending_skill_paths_to_cleanup.append(path_str)

                    batch_size = len(scores)
                    for key, value in reward_extra_info.items():
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        if isinstance(value, (list, tuple)):
                            if len(value) == batch_size:
                                reward_extra_infos_dict[key].extend(value)
                            else:
                                continue
                        else:
                            continue

            if val_only_mode:
                # Persist per-batch updates so later validation batches can use
                # newly selected or newly generated skills in this eval run.
                val_skill_step = self.global_steps * 100000 + val_batch_idx
                self.tool_adaptation_manager.process_reward_extra_info(
                    reward_extra_info=reward_extra_info,
                    global_step=val_skill_step,
                    write_immediately=True,
                )
                    
            tool_interact_info = test_batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if isinstance(tool_interact_info, list) and len(tool_interact_info) == len(scores):
                for tool_interact in tool_interact_info:
                    if isinstance(tool_interact, dict) and "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_dict["tool_interact_info"].extend(tool_interact_info)

                traj_stop_reason = test_batch.non_tensor_batch.get("traj_stop_reason", None)
                if isinstance(traj_stop_reason, np.ndarray):
                    traj_stop_reason = traj_stop_reason.tolist()
                if isinstance(traj_stop_reason, list) and len(traj_stop_reason) == len(scores):
                    reward_extra_infos_dict["traj_stop_reason"].extend(traj_stop_reason)

                verl_tool_metrics = test_batch.non_tensor_batch.get("verl_tool_metrics", None)
                if isinstance(verl_tool_metrics, np.ndarray):
                    verl_tool_metrics = verl_tool_metrics.tolist()
                if isinstance(verl_tool_metrics, list) and len(verl_tool_metrics) == len(scores):
                    reward_extra_infos_dict["verl_tool_metrics"].extend(verl_tool_metrics)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            self._cleanup_all_pending_dirs(reason="validate_batch_end")

        self._cleanup_validation_pending_skill_paths(pending_skill_paths_to_cleanup)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            reward_extra_infos_to_dump = _build_validation_dump_reward_infos(
                reward_extra_infos_dict=reward_extra_infos_dict,
                data_source_lst=data_source_lst,
                expected_len=len(sample_scores),
            )
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=val_data_dir,
            )
        if "tool_interact_info" in reward_extra_infos_dict:
            # remove if after dump
            reward_extra_infos_dict.pop("tool_interact_info")
        if "traj_stop_reason" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("traj_stop_reason")
        if "verl_tool_metrics" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("verl_tool_metrics")

        if "acc" not in reward_extra_infos_dict:
            acc_source = None
            if "is_correct" in reward_extra_infos_dict:
                acc_source = reward_extra_infos_dict["is_correct"]
            elif "accuracy" in reward_extra_infos_dict:
                acc_source = reward_extra_infos_dict["accuracy"]

            if acc_source is not None:
                acc_values: list[bool] = []
                for value in acc_source:
                    if isinstance(value, (bool, np.bool_)):
                        acc_values.append(bool(value))
                        continue
                    if isinstance(value, str):
                        normalized = value.strip().lower()
                        if normalized in {"true", "1", "yes"}:
                            acc_values.append(True)
                            continue
                        if normalized in {"false", "0", "no", ""}:
                            acc_values.append(False)
                            continue
                    try:
                        acc_values.append(bool(float(value)))
                    except (TypeError, ValueError):
                        acc_values.append(False)
                reward_extra_infos_dict["acc"] = acc_values

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs_attention_masks = batch.batch['attention_mask'][:, :batch.batch['prompts'].shape[1]]
            outputs_attention_masks = batch.batch['attention_mask'][:, batch.batch['prompts'].shape[1]:]
            inputs = [self.tokenizer.decode(batch.batch["prompts"][i][inputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["prompts"].shape[0])]
            outputs = [self.tokenizer.decode(batch.batch["responses"][i][outputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["responses"].shape[0])]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            batch_size = len(scores)

            reward_extra_infos_to_dump = {}
            for key, value in reward_extra_infos_dict.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)) and len(value) == batch_size:
                    reward_extra_infos_to_dump[key] = list(value)
            if "request_id" in batch.non_tensor_batch:
                request_id = batch.non_tensor_batch["request_id"]
                if isinstance(request_id, np.ndarray):
                    request_id = request_id.tolist()
                if isinstance(request_id, list) and len(request_id) == batch_size:
                    reward_extra_infos_to_dump.setdefault("request_id", request_id)

            data_source = batch.non_tensor_batch.get("data_source", None)
            if isinstance(data_source, np.ndarray):
                data_source = data_source.tolist()
            if isinstance(data_source, list) and len(data_source) == batch_size:
                reward_extra_infos_to_dump.setdefault("data_source", data_source)

            tool_interact_info = batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if isinstance(tool_interact_info, list) and len(tool_interact_info) == batch_size:
                for tool_interact in tool_interact_info:
                    if isinstance(tool_interact, dict) and "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_to_dump["tool_interact_info"] = tool_interact_info

                traj_stop_reason = batch.non_tensor_batch.get("traj_stop_reason", None)
                if isinstance(traj_stop_reason, np.ndarray):
                    traj_stop_reason = traj_stop_reason.tolist()
                if isinstance(traj_stop_reason, list) and len(traj_stop_reason) == batch_size:
                    reward_extra_infos_to_dump["traj_stop_reason"] = traj_stop_reason

                verl_tool_metrics = batch.non_tensor_batch.get("verl_tool_metrics", None)
                if isinstance(verl_tool_metrics, np.ndarray):
                    verl_tool_metrics = verl_tool_metrics.tolist()
                if isinstance(verl_tool_metrics, list) and len(verl_tool_metrics) == batch_size:
                    reward_extra_infos_to_dump["verl_tool_metrics"] = verl_tool_metrics

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
