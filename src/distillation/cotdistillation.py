import torch.nn.functional as F

def cot_loss(teacher_reasoning_ids, student_logits):
    # teacher_reasoning_ids: [batch, seq] target tokens (teacher reasoning)
    # student_logits: [batch, seq, vocab]
    return F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_reasoning_ids.view(-1),
        ignore_index=-100,
    )