import os
import torch


def save_checkpoint(args, model, global_step, val_loss=None, tokenizer=None):
    output_dirs = []
    if args.save_all_checkpoints:
        output_dirs = [os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))]

    if args.save_last_checkpoint:
        output_dirs.append(os.path.join(args.output_dir, 'checkpoint-last'))

    if args.save_best_checkpoint and val_loss is not None:
        # Get the metric for best checkpoint
        val_metric = val_loss[args.best_checkpoint_metric] if isinstance(val_loss, dict) else val_loss
        prev_best = getattr(save_checkpoint, "best", val_metric)

        is_better = (lambda x, y: x >= y) if args.maximize_best_checkpoint_metric else (lambda x, y: x <= y)
        if is_better(val_metric, prev_best):
            save_checkpoint.best = val_metric
            output_dirs.append(os.path.join(args.output_dir, 'checkpoint-best'))

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            for fn in os.listdir(output_dir):  # clean the directory
                os.remove(os.path.join(output_dir, fn))

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    return output_dirs
