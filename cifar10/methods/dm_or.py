import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMethod
from .utils import gen_centers, gen_reference
from model import VelocityNet
from eval.get_data import get_data
from eval.sgd import eval_sgd
from eval.knn import eval_knn
from scipy.optimize import linear_sum_assignment

class DM(BaseMethod):
    def __init__(self, cfg, centers):
        super().__init__(cfg)
        self.cfg = cfg
        self.centers = centers 
        self.v_net = VelocityNet(cfg.emb)

    def ode_solve(self, z, steps):
        """ 连续性方程的数值实现：将数据从 t=0 演化到 t=1 """
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = torch.full((z.shape[0], 1), i * dt).to(z.device)
            z = z + self.v_net(z, t_curr) * dt
        return z

    def forward(self, samples):
        """ 核心训练逻辑：独立 OT 匹配 + 非匀速 Flow Matching + 显式对齐 """
        x1, x2 = samples
        x1, x2 = x1.cuda(), x2.cuda()

        z0_1 = F.normalize(self.head(self.model(x1)), p=2, dim=1)
        z0_2 = F.normalize(self.head(self.model(x2)), p=2, dim=1)
        batch_size = z0_1.shape[0]

        z0_avg = F.normalize(z0_1 + z0_2, p=2, dim=1)          

        with torch.no_grad():
            r_candidates = gen_reference(self.centers, batch_size, self.cfg.eps)
            sim_matrix = torch.mm(z0_avg, r_candidates.t())
            
            cost_matrix = -sim_matrix
            
            _, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            col_ind = torch.tensor(col_ind).to(z0_1.device)
            r = r_candidates[col_ind]

        t1 = torch.rand(batch_size, 1).to(z0_1.device)
        t2 = torch.rand(batch_size, 1).to(z0_2.device)


        s_t1 = t1
        s_t2 = t2
        

        s_dot1 = 1
        s_dot2 = 1

        zt_1 = (1 - s_t1) * z0_1 + s_t1 * r
        zt_2 = (1 - s_t2) * z0_2 + s_t2 * r

        loss_fm1 = F.mse_loss(self.v_net(zt_1, t1), s_dot1 * (r - z0_1))
        loss_fm2 = F.mse_loss(self.v_net(zt_2, t2), s_dot2 * (r - z0_2))

        loss_fm = (loss_fm1 + loss_fm2) / 2

        loss_align = F.mse_loss(z0_1, z0_2)

        return loss_fm + self.cfg.lambda_param * loss_align

        """ 重写评估函数：在 z1 空间（经过 ODE 纠偏后）进行测试 """




