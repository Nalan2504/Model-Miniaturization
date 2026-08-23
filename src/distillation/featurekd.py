import torch
import torch.nn.functional as F

def feature_kd_loss(teacher_hidden, student_hidden):
    # teacher_hidden, student_hidden: [batch, seq, hidden_dim]
    return F.mse_loss(student_hidden, teacher_hidden)