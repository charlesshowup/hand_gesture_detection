def calc_steps(epoch, batch_size, num_samples):
    num_steps = (num_samples / batch_size) * epoch
    return int(num_steps)

num_samples = 1800
batch_size = 16
total_steps = calc_steps(epoch=200, batch_size=batch_size, num_samples=num_samples)
# total_steps = 200000
# warmup_steps = 4000
warmup_steps = calc_steps(epoch=4, batch_size=batch_size, num_samples=num_samples)
print('total_steps:%d' % total_steps)
print('warmup_steps:%d' % warmup_steps)

params = {
    "used_gpus": '1',
    "is_train_from_begining": True,
    "train_and_validate": True,  # FDDB validation
    "save_summary": True,
    "save_ckpt_every_n_steps": 5000,
    "model_params": {
        "model_dir": "checkpoint_train",

        "weight_decay": 1e-3,
        "score_threshold": 0.05, "iou_threshold": 0.3, "max_boxes": 200,

        # multi loss
        "use_multi_loss": True,
        # if use_multi_loss == False, use config below
        "localization_loss_weight": 1.0,
        "classification_loss_weight": 1.0,
        "landmark_loss_weight": 0.5,
        "quality_loss_weight": 1.0,
        "blur_loss_weight": 1.0,
        "occlude_loss_weight": 1.0,
        "use_diou_loss": False,
        "use_stitcher": False,

        # online hard example mining
        "loss_to_use": "classification",
        "loc_loss_weight": 0.0, "cls_loss_weight": 1.0, "lmk_loss_weight": 0.0,
        "num_hard_examples": 500, "nms_threshold": 0.99,
        "max_negatives_per_positive": 3.0, "min_negatives_per_image": 30,

        # learning rate decay
        "use_cosine_decay": True,
        # if use_cosine_decay == True, use config below
        "learning_rate_base": 0.004,
        "total_steps": total_steps,
        "warmup_learning_rate": 0.000004,
        "warmup_steps": warmup_steps,
        #"hold_base_rate_steps": 10000,
        "hold_base_rate_steps": 0,
        # if use_cosine_decay == False, use config below
        #"lr_boundaries": [20000, 100000, 200000, 300000],
        #"lr_values": [0.0001, 0.00006, 0.00003, 0.00001, 0.00001],
        "lr_boundaries": [10000, 30000, 40000],
        "lr_values": [0.0001, 0.00006, 0.00003, 0.00001],

        # each loss setup
        "use_class_label_smoothing": True,
        "class_label_smoothing": 0.1,
        "use_landmark_wing_loss": True,
        "use_occlude_label_smoothing": True,
        "occlude_label_smoothing": 0.1,

        # if continue train with no aug
        "is_continue_train_with_no_aug": False,

        # fine_tune
        "is_fine_tune_landmark": False  # always false let the func to change it
    },

    "input_pipeline_params": {
        "image_size": [384, 384],
        "batch_size": batch_size,
        "train_dataset": "/Users/chenjiayi/Downloads/hand_gesture/dataset/train",
        # "train_dataset": "/home/chenjy531/Desktop/data/trans/HAGRID_tfrecord/3d_train",
        "val_dataset": "./data/val_shards",
        "num_steps": total_steps,
        "use_bbox_only": False  # tfrecords has no landmarks, quality etc...
    },

    "quantization_params": {
        "use_quantization_model": False,
        "is_train_fake_model": True,
        "is_restore_from_float_ckpt": True,
        "is_full_fixed_model": True,
        "is_weight_use_LSQ": True,
        "quantization_weight_bit_num": 8,
        "quantization_act_bit_num": 4,
    }
}


