# yapf:disable

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='uncategorized')),
        # dict(type='WandbLoggerHook'),
    ])
# log_config = dict(
#     interval=100,
#     hooks=[
#             dict(type='TextLoggerHook'),
#             dict(type='WandbLoggerHook',
#                  init_kwargs={'project': 'veuirfdf'},
#                  interval=100,
#                  log_checkpoint=True,
#                  log_checkpoint_metadata=True,
#                  num_eval_images=100)]
# )

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
