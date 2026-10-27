"""Install into an existing official T-Rex checkout; require exact known import."""
import argparse, shutil
from pathlib import Path

def install(repo):
    repo=Path(repo); path=repo/'scripts/train.py'
    source=path.read_text(encoding='utf-8')
    old='''    if getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)'''
    new='''    if getattr(args, "data_format", "json") == "sharpa":
        from qwen_vla.sharpa_dataset import SharpaDataset
        dataset = SharpaDataset(args, processor, accelerator)
    elif getattr(args, "data_format", "json") == "lerobot":
        from qwen_vla.lerobot_dataset import TRexLeRobotDataset
        dataset = TRexLeRobotDataset(args, processor, accelerator)'''
    if old not in source and new not in source:
        raise RuntimeError('Upstream train.py changed; inspect its dataset factory before installing')
    stats_anchor='''        with open(os.path.join(save_dir, "stats_data.json"), "w") as f:
            json.dump(stats_data, f, indent=2)'''
    metadata_patch='''
        # Persist the Sharpa action reference with every trained checkpoint.
        if getattr(args, "data_format", "json") == "sharpa":
            shutil.copy2(os.path.join(args.lerobot_root, "adapter_metadata.json"),
                         os.path.join(save_dir, "adapter_metadata.json"))
            shutil.copytree(os.path.join(args.lerobot_root, "robot"),
                            os.path.join(save_dir, "robot"), dirs_exist_ok=True)
'''
    if '# Persist the Sharpa action reference' not in source:
        if stats_anchor not in source:raise RuntimeError('Upstream checkpoint saving changed; inspect before installing')
        source=source.replace(stats_anchor,stats_anchor+metadata_patch)
    old_guard='''    if args.data_format == "lerobot":
        if not args.lerobot_root:
            parser.error("--data_format lerobot requires --lerobot_root")
    elif not args.data_path:
        parser.error("--data_format json requires --data_path")'''
    new_guard='''    if args.data_format in ("lerobot", "sharpa"):
        if not args.lerobot_root:
            parser.error("--data_format lerobot/sharpa requires --lerobot_root")
    elif not args.data_path:
        parser.error("--data_format json requires --data_path")'''
    if old_guard not in source and new_guard not in source:
        raise RuntimeError('Upstream CLI validation changed; inspect before installing')
    source=source.replace(old_guard,new_guard)
    parser_anchor='    parser.add_argument("--n_epochs", type=int, default=200)'
    if '--max_train_batches' not in source:
        if parser_anchor not in source:raise RuntimeError('Upstream epoch argument changed')
        source=source.replace(parser_anchor,parser_anchor+'\n    parser.add_argument("--max_train_batches", type=int, default=0, help="Stop after this many training batches; 0 disables the short-run limit")')
        loop_anchor='''            global_step += 1

        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:'''
        loop_new='''            global_step += 1
            if args.max_train_batches > 0 and global_step >= args.max_train_batches:
                break

        short_run_done = args.max_train_batches > 0 and global_step >= args.max_train_batches
        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1 or short_run_done:'''
        if loop_anchor not in source:raise RuntimeError('Upstream training loop changed')
        source=source.replace(loop_anchor,loop_new)
        end_anchor='''                            epoch, global_step, dataset.stats_data)


if __name__ == "__main__":'''
        end_new='''                            epoch, global_step, dataset.stats_data)
        if short_run_done:
            break


if __name__ == "__main__":'''
        if end_anchor not in source:raise RuntimeError('Upstream checkpoint tail changed')
        source=source.replace(end_anchor,end_new)
    source=source.replace('''    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)''','''    if args.use_tactile_deform and args.deform_encoder_ckpt:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)''')
    if new not in source:
        if 'choices=["json", "lerobot"]' not in source:
            raise RuntimeError('Upstream data_format parser changed; inspect before installing')
        backup=path.with_suffix('.py.pre_sharpa')
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(path,backup)
    here=Path(__file__).resolve().parent
    for name in ('sharpa_dataset.py','sharpa_core.py'):
        shutil.copy2(here/name,repo/'qwen_vla'/name)
    path.write_text(source.replace(old,new).replace('choices=["json", "lerobot"]','choices=["json", "lerobot", "sharpa"]'),encoding='utf-8')
    print('Installed --data_format sharpa. Backup: scripts/train.py.pre_sharpa')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True)
    install(p.parse_args().repo)
