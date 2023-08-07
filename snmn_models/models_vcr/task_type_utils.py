from .config import cfg

def get_name_prefix(index):
    if index == 0:
        return 'question'
    elif index == 1:
        return 'answer'
    elif index == 2:
        return 'rationale'
    else:
        raise ValueError(f'Invalid index {index}')

# def get_name_prefix(index):
#     if cfg.MODEL.VCR_TASK_TYPE == 'Q_2_A' and 0 <= index <= 1:
#         return 'question' if index == 0 else f'answer'
#     elif (cfg.MODEL.VCR_TASK_TYPE == 'QA_2_R' or cfg.MODEL.VCR_TASK_TYPE == 'Q_2_AR') and 0 <= index <= 2:
#         return 'question' if index == 0 else f'answer' if index == 1 else f'rationale'
#     else:
#         raise ValueError(f'Invalid VCR_TASK_TYPE: {cfg.MODEL.VCR_TASK_TYPE}')
