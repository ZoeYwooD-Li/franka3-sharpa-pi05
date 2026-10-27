"""Add optional single-GPU LoRA support to an installed Sharpa T-Rex checkout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str,
                 previous: str | None = None) -> str:
    if new in source:
        return source
    if previous is not None and previous in source:
        if source.count(previous) != 1:
            raise RuntimeError(f"Unexpected prior {label}; inspect before patching")
        return source.replace(previous, new, 1)
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream {label}; inspect before patching")
    return source.replace(old, new, 1)


def install(repo: Path) -> None:
    repo = repo.resolve()
    trainer = repo / "scripts" / "train.py"
    modeling = repo / "qwen_vla" / "modeling_vla.py"
    mot = repo / "qwen_vla" / "modeling_qwen3vl_mot.py"
    source = trainer.read_text(encoding="utf-8")
    if 'choices=["json", "lerobot", "sharpa"]' not in source:
        raise RuntimeError("Install the v3 Sharpa adapter before LoRA")
    backup = trainer.with_suffix(".py.pre_sharpa_lora")
    if not backup.exists():
        shutil.copy2(trainer, backup)

    source = replace_once(source,
        "from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds\n",
        "from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds\n"
        "from qwen_vla.sharpa_lora import inject_lora, save_lora\n",
        "LoRA import")
    source = replace_once(source,
        '        sd = accelerator.get_state_dict(model)\n'
        '        torch.save(sd, os.path.join(save_dir, "model.pt"))',
        '        if args.lora_rank > 0:\n'
        '            save_lora(accelerator.unwrap_model(model), save_dir, args.lora_config)\n'
        '        else:\n'
        '            sd = accelerator.get_state_dict(model)\n'
        '            torch.save(sd, os.path.join(save_dir, "model.pt"))',
        "checkpoint storage")
    source = replace_once(source,
        '                "model_path": args.model_path,',
        '                "model_path": args.model_path,\n'
        '                "lora_rank": args.lora_rank,\n'
        '                "lora_alpha": args.lora_alpha,\n'
        '                "lora_dropout": args.lora_dropout,',
        "checkpoint training args")
    source = replace_once(source,
        '        self.world_size   = dist.get_world_size()',
        '        self.world_size   = dist.get_world_size() if dist.is_initialized() else 1',
        "single-GPU metric size")
    source = replace_once(source,
        '        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:\n'
        '            dist.all_reduce(t, op=dist.ReduceOp.SUM)',
        '        for t in [self.total_loss, self.action_loss, self.tactile_loss, self.flare_loss]:\n'
        '            if dist.is_initialized():\n'
        '                dist.all_reduce(t, op=dist.ReduceOp.SUM)',
        "single-GPU metric reduction")
    source = replace_once(source,
        '    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu\n'
        '    accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps)',
        '    if accelerator.state.deepspeed_plugin is not None:\n'
        '        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu\n'
        '        accelerator.state.deepspeed_plugin.deepspeed_config["train_batch_size"] = (\n'
        '            args.train_bsz_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps)',
        "optional DeepSpeed")
    source = replace_once(source,
        'def train(args):\n'
        '    accelerator = Accelerator(',
        'def train(args):\n'
        '    if args.lora_rank > 0 and args.lora_sdpa_math:\n'
        '        # On RTX 5090 with torch 2.7/cu128, the fused SDPA backward\n'
        '        # produced non-finite expert gradients in this training path.\n'
        '        torch.backends.cuda.enable_flash_sdp(False)\n'
        '        torch.backends.cuda.enable_mem_efficient_sdp(False)\n'
        '        torch.backends.cuda.enable_math_sdp(True)\n'
        '    accelerator = Accelerator(',
        "math SDPA for LoRA",
        previous='def train(args):\n'
                 '    if args.lora_rank > 0:\n'
                 '        torch.backends.cuda.enable_flash_sdp(False)\n'
                 '        torch.backends.cuda.enable_mem_efficient_sdp(False)\n'
                 '        torch.backends.cuda.enable_math_sdp(True)\n'
                 '    accelerator = Accelerator(')
    source = replace_once(source,
        '        // dist.get_world_size()',
        '        // accelerator.num_processes',
        "single-GPU step count")
    source = replace_once(source,
        '    num_training_steps = (\n'
        '        int(len(dataloader) * args.n_epochs)\n'
        '        // accelerator.gradient_accumulation_steps\n'
        '        // accelerator.num_processes\n'
        '    )',
        '    # max_train_batches counts batches processed by each rank.\n'
        '    # Convert the unsharded loader length before applying that cap.\n'
        '    per_rank_batches = (\n'
        '        (len(dataloader) + accelerator.num_processes - 1)\n'
        '        // accelerator.num_processes\n'
        '    ) * args.n_epochs\n'
        '    if args.max_train_batches > 0:\n'
        '        per_rank_batches = min(per_rank_batches, args.max_train_batches)\n'
        '    num_training_steps = max(1,\n'
        '        per_rank_batches\n'
        '        // accelerator.gradient_accumulation_steps\n'
        '    )',
        "bounded learning-rate schedule",
        previous='    total_train_batches = int(len(dataloader) * args.n_epochs)\n'
                 '    if args.max_train_batches > 0:\n'
                 '        total_train_batches = min(total_train_batches, args.max_train_batches)\n'
                 '    num_training_steps = max(1,\n'
                 '        total_train_batches\n'
                 '        // accelerator.gradient_accumulation_steps\n'
                 '        // accelerator.num_processes\n'
                 '    )')
    source = replace_once(source,
        '    global_step = 0\n'
        '    T_per_frame = args.n_flare_tokens_per_frame',
        '    global_step = 0\n'
        '    last_saved_step = -1\n'
        '    T_per_frame = args.n_flare_tokens_per_frame',
        "periodic checkpoint state")
    source = replace_once(source,
        '        model, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(\n'
        '            model, optimizer, dataloader, val_dataloader, lr_scheduler)\n'
        '    else:\n'
        '        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(\n'
        '            model, optimizer, dataloader, lr_scheduler)',
        '        model, optimizer, dataloader, val_dataloader = accelerator.prepare(\n'
        '            model, optimizer, dataloader, val_dataloader)\n'
        '    else:\n'
        '        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)',
        "one scheduler step per optimizer update")
    source = replace_once(source,
        '            global_step += 1\n'
        '            if args.max_train_batches > 0 and global_step >= args.max_train_batches:',
        '            global_step += 1\n'
        '            if args.save_steps > 0 and global_step % args.save_steps == 0:\n'
        '                accelerator.wait_for_everyone()\n'
        '                save_checkpoint(model, processor, accelerator, args,\n'
        '                                epoch, global_step, dataset.stats_data)\n'
        '                last_saved_step = global_step\n'
        '            if args.max_train_batches > 0 and global_step >= args.max_train_batches:',
        "periodic checkpoint save")
    source = replace_once(source,
        '        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1 or short_run_done:',
        '        if (global_step != last_saved_step and\n'
        '                ((epoch + 1) % args.save_freq == 0 or\n'
        '                 epoch == args.n_epochs - 1 or short_run_done)):',
        "avoid duplicate final checkpoint")
    source = replace_once(source,
        '        else:\n'
        '            param.requires_grad = True\n\n'
        '    # Keep the embedded VQ-VAE',
        '        else:\n'
        '            param.requires_grad = True\n\n'
        '    if args.lora_rank > 0:\n'
        '        args.lora_config = inject_lora(\n'
        '            model, args.lora_rank, args.lora_alpha, args.lora_dropout,\n'
        '            gradient_checkpointing=bool(args.lora_gradient_checkpointing))\n'
        '        accelerator.print(f"LoRA: {len(args.lora_config[\'target_modules\'])} linear layers, "\n'
        '                          f"{args.lora_config[\'trainable_parameters\']:,} trainable parameters")\n\n'
        '    # Keep the embedded VQ-VAE',
        "LoRA injection")
    source = replace_once(source,
        '    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")',
        '    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")\n'
        '    parser.add_argument("--lora_rank", type=int, default=0, help="0: full fine-tuning; >0: train only LoRA matrices")\n'
        '    parser.add_argument("--lora_alpha", type=float, default=16.0)\n'
        '    parser.add_argument("--lora_dropout", type=float, default=0.05)\n'
        '    parser.add_argument("--lora_gradient_checkpointing", type=int, default=1)\n'
        '    parser.add_argument("--lora_sdpa_math", type=int, default=1)',
        "LoRA command-line options",
        previous='    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")\n'
                 '    parser.add_argument("--lora_rank", type=int, default=0, help="0: full fine-tuning; >0: train only LoRA matrices")\n'
                 '    parser.add_argument("--lora_alpha", type=float, default=16.0)\n'
                 '    parser.add_argument("--lora_dropout", type=float, default=0.05)\n'
                 '    parser.add_argument("--lora_gradient_checkpointing", type=int, default=1)')
    source = replace_once(source,
        '    parser.add_argument("--save_freq", type=int, default=50)',
        '    parser.add_argument("--save_freq", type=int, default=50)\n'
        '    parser.add_argument("--save_steps", type=int, default=0, '
        'help="Save every N training batches; 0 disables periodic saves")',
        "periodic checkpoint option")
    source = replace_once(source,
        '                if args.max_grad_norm > 0:\n'
        '                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)',
        '                if args.lora_rank > 0:\n'
        '                    bad_grads = []\n'
        '                    for grad_name, grad_param in model.named_parameters():\n'
        '                        if grad_param.grad is not None and not torch.isfinite(grad_param.grad).all():\n'
        '                            bad_grads.append(grad_name)\n'
        '                    if bad_grads:\n'
        '                        raise FloatingPointError(\n'
        '                            f"Non-finite LoRA gradients before clipping: "\n'
        '                            f"{len(bad_grads)} tensors; first={bad_grads[:12]}")\n'
        '                if args.max_grad_norm > 0:\n'
        '                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)',
        "non-finite LoRA gradient guard",
        previous='                if args.lora_rank > 0:\n'
                 '                    for grad_name, grad_param in model.named_parameters():\n'
                 '                        if grad_param.grad is not None and not torch.isfinite(grad_param.grad).all():\n'
                 '                            raise FloatingPointError(f"Non-finite LoRA gradient before clipping: {grad_name}")\n'
                 '                if args.max_grad_norm > 0:\n'
                 '                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)')

    model_source = modeling.read_text(encoding="utf-8")
    model_source = replace_once(model_source,
        '            flat = tactile_deform.view(-1, C, Hd, Wd).to(dtype)\n'
        '            feats = self.deform_encoder(flat).view(Bd, nf, -1)',
        '            # The frozen deform encoder can remain fp32 while the tactile\n'
        '            # tokens use bf16, including outside Accelerator autocast.\n'
        '            flat = tactile_deform.view(-1, C, Hd, Wd).to(\n'
        '                next(self.deform_encoder.parameters()).dtype)\n'
        '            feats = self.deform_encoder(flat).view(Bd, nf, -1)',
        "deform encoder input dtype")
    mot_source = mot.read_text(encoding="utf-8")
    mot_source = replace_once(mot_source,
        '            if self.gradient_checkpointing and self.training:\n'
        '                layer_out = torch.utils.checkpoint.checkpoint(',
        '            # A cached tactile forward mutates the KV cache; replaying it\n'
        '            # during checkpoint backward would append keys twice.\n'
        '            if self.gradient_checkpointing and self.training and not use_cache:\n'
        '                layer_out = torch.utils.checkpoint.checkpoint(',
        "KV-safe gradient checkpointing")

    shutil.copy2(Path(__file__).with_name("sharpa_lora.py"), repo / "qwen_vla" / "sharpa_lora.py")
    model_backup = modeling.with_suffix(".py.pre_sharpa_lora")
    if not model_backup.exists():
        shutil.copy2(modeling, model_backup)
    mot_backup = mot.with_suffix(".py.pre_sharpa_lora")
    if not mot_backup.exists():
        shutil.copy2(mot, mot_backup)
    trainer.write_text(source, encoding="utf-8")
    modeling.write_text(model_source, encoding="utf-8")
    mot.write_text(mot_source, encoding="utf-8")
    print("Installed optional LoRA and single-GPU path; backups:", backup, model_backup, mot_backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    install(parser.parse_args().repo)
