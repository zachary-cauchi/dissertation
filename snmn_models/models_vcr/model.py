import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from .config import cfg
from . import controller, nmn, input_unit, output_unit, vis


class Model:
    def __init__(self,
                 question_seq_batch,
                 all_answers_seq_batch,
                 all_rationales_seq_batch,
                 question_length_batch,
                 all_answers_length_batch,
                 all_rationales_length_batch,
                 bert_question_embeddings_batch,
                 bert_answer_embeddings_batch,
                 bert_rationale_embeddings_batch,
                 image_feat_batch,
                 num_vocab, num_choices, module_names, is_training,
                 use_cudnn_lstm=True,
                 scope='model', reuse=None):
        """
        Neual Module Networks v4 (the whole model)

        Input:
            question_seq_batch: [S, N], tf.int32
            question_length_batch: [N], tf.int32
            image_feat_batch: [N, H, W, C], tf.float32
        """

        with tf.variable_scope(scope, reuse=reuse):
            self.T_ctrl = cfg.MODEL.T_CTRL
            # Question + answer + rationale input so 3 total input sequences.
            if cfg.MODEL.VCR_TASK_TYPE == 'Q_2_A':
                self.seq_in_count = 2
            else:
                self.seq_in_count = 3

            # Input unit
            lstm_seq, lstm_encodings, embed_seq = input_unit.build_input_unit(
                question_seq_batch, all_answers_seq_batch, all_rationales_seq_batch, question_length_batch, all_answers_length_batch, all_rationales_length_batch, bert_question_embeddings_batch, bert_answer_embeddings_batch, bert_rationale_embeddings_batch, num_vocab, self.seq_in_count, reuse=reuse, use_cudnn_lstm=use_cudnn_lstm)
            kb_batch = input_unit.build_kb_batch(image_feat_batch, reuse=reuse)

            # Controller and NMN
            num_module = len(module_names)
            self.controller = controller.Controller(
                lstm_seq, lstm_encodings, embed_seq, question_length_batch, all_answers_length_batch, all_rationales_length_batch, num_module, self.seq_in_count, reuse=reuse)
            self.c_list = self.controller.c_list
            self.module_logits = self.controller.module_logits
            self.module_probs = self.controller.module_probs
            self.module_prob_list = self.controller.module_prob_list
            self.nmn = nmn.NMN(
                kb_batch, self.c_list, module_names, self.module_prob_list, reuse=reuse)

            # Output unit
            if cfg.MODEL.BUILD_VQA:
                vqa_scores = output_unit.build_output_unit_vqa(
                    lstm_encodings, self.nmn.mem_last, num_choices,
                    apply_dropout=is_training, reuse=reuse)
                
                if cfg.TRAIN.SOLVER.USE_SPARSE_SOFTMAX_LABELS:
                    self.vqa_scores = tf.reshape(vqa_scores, (tf.shape(vqa_scores)[0] // num_choices, num_choices))
                else:
                    self.vqa_scores = vqa_scores

            if cfg.MODEL.BUILD_LOC:
                loc_scores, bbox_offset, bbox_offset_fcn = \
                    output_unit.build_output_unit_loc(
                        lstm_encodings['question'], kb_batch, self.nmn.att_last)
                self.loc_scores = loc_scores
                self.bbox_offset = bbox_offset
                self.bbox_offset_fcn = bbox_offset_fcn

            # Reconstruction loss
            if cfg.MODEL.REC.USE_REC_LOSS:
                rec_inputs = (self.module_logits if cfg.MODEL.REC.USE_LOGITS
                              else self.module_probs)
                if cfg.MODEL.REC.USE_TXT_ATT:
                    rec_inputs = tf.concat(
                        [rec_inputs, tf.stack(self.c_list)], axis=-1)
                self.rec_loss = output_unit.build_output_unit_rec(
                    rec_inputs, all_answers_seq_batch, embed_seq, question_length_batch,
                    num_vocab)
            else:
                self.rec_loss = tf.convert_to_tensor(0.)

            self.params = [ v for v in tf.trainable_variables(scope) ]
            
            l1_regs = []
            l2_regs = []
            
            for v in self.params:
                if 'weights' in v.op.name:
                    l1_regs.append(tf.reduce_sum(tf.abs(v), name=f'l1_{v.op.name}'))
                    l2_regs.append(tf.nn.l2_loss(v, name=f'l2_{v.op.name}'))

            self.l1_reg = tf.add_n(l1_regs, name='l1_reg')
            self.l2_reg = tf.add_n(l2_regs, name='l2_reg')
            self.reg_rho = cfg.TRAIN.L1_L2_RHO

            # Use a rho of 1. for lasso (l1) regularisation only, use a rho of 0. for ridge (l2) regularisation only.
            self.elastic_net_reg =  tf.add(self.reg_rho * self.l1_reg, (1 - self.reg_rho) * self.l2_reg, name='elastic_net_reg')

            # tensors for visualization
            self.vis_outputs = {
                'txt_att':  # [N, T, S]
                tf.transpose(  # [S, N, T] -> [N, T, S]
                    tf.concat(self.controller.cv_list, axis=2), (1, 2, 0), name='txt_att'),
                'att_stack':  # [N, T, H, W, L]
                tf.stack(self.nmn.att_stack_list, axis=1, name='att_stack'),
                'stack_ptr':  # [N, T, L]
                tf.stack(self.nmn.stack_ptr_list, axis=1, name='stack_ptr'),
                'module_prob':  # [N, T, D]
                tf.stack(self.module_prob_list, axis=1, name='module_prob')
            }
            
            if cfg.MODEL.BUILD_VQA:
                self.vis_outputs['vqa_scores'] = self.vqa_scores
            if cfg.MODEL.BUILD_LOC:
                self.vis_outputs['loc_scores'] = self.loc_scores
                self.vis_outputs['bbox_offset'] = self.bbox_offset

    def bbox_offset_loss(self, bbox_ind_batch, bbox_offset_batch):
        if cfg.MODEL.BBOX_REG_AS_FCN:
            N = tf.shape(self.bbox_offset_fcn)[0]
            B = tf.shape(self.bbox_offset_fcn)[1]  # B = H*W
            bbox_offset_flat = tf.reshape(self.bbox_offset_fcn, to_T([N*B, 4]))
            slice_inds = tf.range(N) * B + bbox_ind_batch
            bbox_offset_sliced = tf.gather(bbox_offset_flat, slice_inds)
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(bbox_offset_sliced, bbox_offset_batch))
        else:
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(self.bbox_offset, bbox_offset_batch))

        return loss_bbox_offset

    def vis_batch_vqa(self, *args):
        vis.vis_batch_vqa(self, *args)

    def vis_batch_loc(self, *args):
        vis.vis_batch_loc(self, *args)
