import argparse


def add_general_args(parser):
    group = parser.add_argument_group("General")

    group.add_argument("--model_type", default=None, type=str, required=True, help="Model type.")
    group.add_argument("--model_name_or_path", default=None, type=str,
                       help="Path to pre-trained model or shortcut name.")
    group.add_argument("--config_name", default="", type=str,
                       help="Pretrained config name or path if not the same as model_name.")

    group.add_argument("--data_dir", default=None, type=str,
                       help="The input data dir. Should contain the .json files for the task."
                            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    group.add_argument("--output_dir", default=None, type=str, required=True,
                       help="The output directory where the model checkpoints and predictions will be written.", )
    group.add_argument("--cache_dir", default="", type=str,
                       help="Where do you want to store the pre-trained models downloaded from s3")

    group.add_argument("--do_train", action="store_true", help="Whether to run training.")
    group.add_argument("--evaluate_during_training", action="store_true",
                       help="Run evaluation during training at each logging step.")
    group.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    group.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    group.add_argument("--verbose", action="store_true", help="If true, print more warnings for debugging. ")
    return group


def add_training_args(parser):
    group = parser.add_argument_group("Training")
    group.add_argument("--train_file", default=None, type=str,
                       help="The input training file. If a data dir is specified, will look for the file there"
                            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    group.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    group.add_argument('--evaluate_steps', default=100, type=int,
                       help="Evaluate every X steps during training.")
    return group


def add_evaluate_args(parser):
    group = parser.add_argument_group("Evaluate")
    group.add_argument("--predict_file", default=None, type=str,
                       help="The input evaluation file. If a data dir is specified, will look for the file there"
                            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    group.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    group.add_argument("--eval_all_checkpoints", action="store_true",
                       help="(deprecated) Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    return group


def add_preprocess_args(parser):
    group = parser.add_argument_group("Dataset preprocessing")
    group.add_argument("--max_seq_length", default=384, type=int,
                       help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
    group.add_argument("--doc_stride", default=128, type=int,
                       help="When splitting up a long document into chunks, how much stride to take between chunks.")
    group.add_argument("--overwrite_cache", action="store_true",
                       help="Overwrite the cached training and evaluation sets")
    group.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    return group


def add_tokenizer_args(parser):
    group = parser.add_argument_group("Tokenizer")
    group.add_argument("--tokenizer_name", default="", type=str,
                       help="Pretrained tokenizer name or path if not the same as model_name")
    group.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument("--resume_prev_optim", action="store_true", help="Restore from previous optimizer states.")

    group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    group.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    group.add_argument("--adam_beta1", default=0.9, type=float, help="beta_1 for Adam optimizer.")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="beta_2 for Adam optimizer.")
    group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    group.add_argument("--num_train_epochs", default=3.0, type=float,
                       help="Total number of training epochs to perform.")
    group.add_argument("--max_steps", default=-1, type=int,
                       help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    group.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    group.add_argument("--warmup_ratio", default=0.0, type=float, help="Ratio of warmup over total number of steps.")
    group.add_argument("--early_stop_patience", default=-1, type=int,
                       help="Early stop if the metric did not improve for certain number of steps.")
    return group


def add_squad_args(parser):
    group = parser.add_argument_group("SQuAD dataset arguments")

    group.add_argument("--version_2_with_negative", action="store_true",
                       help="If true, the SQuAD examples contain some that do not have an answer.")
    group.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                       help="If null_score - best_non_null is greater than the threshold predict null.")
    group.add_argument("--max_query_length", default=64, type=int,
                       help="The maximum number of tokens for the question. Questions longer than this will "
                            "be truncated to this length.")
    group.add_argument("--n_best_size", default=20, type=int,
                       help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    group.add_argument("--max_answer_length", default=30, type=int,
                       help="The maximum length of an answer that can be generated. This is needed because the start "
                            "and end predictions are not conditioned on one another.")
    group.add_argument("--verbose_logging", action="store_true",
                       help="If true, all of the warnings related to data processing will be printed. "
                            "A number of warnings are expected for a normal SQuAD evaluation.")
    return group


def add_logging_args(parser):
    group = parser.add_argument_group("Logging")
    group.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    group.add_argument("--tb_logging_dir", type=str, default=None, help="Log output directory for tensorboard.")
    group.add_argument("--system_log_path", type=str, default=None, help="Log file path to record system outputs.")
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    group.add_argument("--overwrite_output_dir", action="store_true",
                       help="Overwrite the content of the output directory")
    group.add_argument("--save_all_checkpoints", action='store_true', help="Save all the checkpoint.")
    group.add_argument("--save_last_checkpoint", action='store_true',
                       help="Only save the last checkpoint, overwrite all previous one to save disk space.")
    group.add_argument("--save_best_checkpoint", action='store_true', help="Save the best checkpoint.")
    group.add_argument("--best_checkpoint_metric", type=str, default="f1",
                       help="The metric for finding the best checkpoint.")
    group.add_argument("--maximize_best_checkpoint_metric", action='store_true',
                       help="True if the goal is to maximize the metric (e.g., accuracy); "
                            "False otherwise (e.g., perplexity).")
    return group


def add_environment_args(parser):
    group = parser.add_argument_group("Environment")
    group.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    group.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    group.add_argument("--fp16", action="store_true",
                       help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    group.add_argument("--fp16_opt_level", type=str, default="O1",
                       help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    # group.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    # group.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    return group


def get_parser():
    parser = argparse.ArgumentParser()
    add_general_args(parser)
    args, _ = parser.parse_known_args()

    if args.do_train:
        add_training_args(parser)
        add_optimization_args(parser)

    if args.do_eval or (args.do_train and args.evaluate_during_training):
        add_evaluate_args(parser)

    add_checkpoint_args(parser)
    add_logging_args(parser)
    add_preprocess_args(parser)
    add_tokenizer_args(parser)
    add_environment_args(parser)

    return parser


def get_squad_parser():
    parser = get_parser()
    add_squad_args(parser)
    return parser
