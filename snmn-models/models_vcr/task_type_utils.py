from .config import cfg

def get_name_prefix(index):
    if cfg.MODEL.VCR_TASK_TYPE == 'Q_2_A':
        return 'question' if index == 0 else f'answer{index}'
    elif cfg.MODEL.VCR_TASK_TYPE == 'QA_2_R' or cfg.MODEL.VCR_TASK_TYPE == 'Q_2_AR':
        return 'question' if index == 0 else f'answer{index}' if index == 1 else f'rationale{index}'
    else:
        raise ValueError(f'Invalid VCR_TASK_TYPE: {cfg.MODEL.VCR_TASK_TYPE}')
