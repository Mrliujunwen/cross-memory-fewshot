import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *
from .ResNet import ResNet
from .cross_memory_attention import CrossMemoryAtt
from .SeqAttention import SeqAttention
from .fusion import iAFF

class TransformerDecoder(nn.Module):
    def __init__(self, args, dim, mlp_hidden_dim, norm_layer):
        super().__init__()
        self.args = args
        self.qkv = nn.Linear(dim, dim)
        self.norm = norm_layer(dim)

    def forward(self, tgt, mem):
        if self.args.dataset == 'fc100':
            query = self.normalize_feature(self.qkv(tgt))
            key = self.normalize_feature(self.qkv(mem))
            value = self.normalize_feature(self.qkv(mem))
        else:
            query = self.qkv(self.norm(tgt))
            key = self.qkv(self.norm(mem))
            value = self.qkv(self.norm(mem))

        score = torch.einsum('i b c, i d c -> i b d', query, key).softmax(dim=-1) 
        tgt = torch.einsum('i b c, i c d -> i b d', score, value)  
        return score, tgt

    def normalize_feature(self, x):
        return F.normalize(x, p=2, dim=2, eps=1e-12)


class sampleFormer(nn.Module):
    def __init__(self, args, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(sampleFormer, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=args.dp2)
        self.transformer_decoder = TransformerDecoder(args, dim, mlp_hidden_dim, norm_layer)
        self.sa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=args.dp3)

    def forward(self, proto, query, resize=False):
        mem = self.encoder_layer(proto).transpose(0,1)
        tgt, _ = self.sa(query, query, query) 
        score, tgt = self.transformer_decoder((tgt+query).transpose(0,1), mem)
        score = score.squeeze(0)
        return score, tgt

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1280, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.encoder_dim = 640

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CSM(nn.Module):
    '''
    Cross-Scale Memory (CSM) Network
    '''
    def __init__(self, args, mode='meta'):
        super(CSM, self).__init__()
        self.mode = mode
        self.args = args
        self.depth = args.N
        self.encoder = ResNet(args=args)
        dim = 640
        self.r = RelationNetwork(64, 8)
        self.conv = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1, stride=1, padding=0)
        self.atten1 = SeqAttention(1280, 1, 1, sqa_type="linear", residual_mode=True)
        self.fu = iAFF()
        self.conv2 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=1)
        self.fc1 = nn.Linear(640, 64)
        self.fc2 = nn.Linear(64, 1)
        if self.mode == 'pre_train':
            if args.dataset == 'cub':
                self.fc = nn.Linear(dim, self.args.num_class)
            else:
                self.fc = nn.Linear(dim, self.args.num_class)

        self.trans = nn.ModuleList([sampleFormer(args, dim=dim, num_heads=args.head_metric, mlp_ratio=4, qkv_bias=True, qk_scale=None, \
                                                    drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm) for i in
                                    range(self.depth)])
        self.patchFormer = nn.MultiheadAttention(embed_dim=640, num_heads=self.args.head_enc, dropout=self.args.dp4)
        self.agent=CrossMemoryAtt(dim=640)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            z_support_save = support
            z_query = query
            support_orig = support
            query_orig = query

            support_orig = support_orig.squeeze(0)
            B_s, C, H, W = support_orig.shape
            B_q, _, _, _ = query_orig.shape
            support_orig = support_orig.reshape(B_s, C, -1).permute(2, 0, 1)
            query_orig = query_orig.reshape(B_q, C, -1).permute(2,0,1)

            support_orig=self.agent(support_orig.squeeze(0).permute(1,0,2)).permute(1,0,2)
            support_orig = self.args.lamda1*support.squeeze(0) + (1-self.args.lamda1)*support_orig.permute(1,2,0).reshape(B_s, C, H, W)
            cpsupport_orig=support_orig.repeat(15,1,1,1)
            cpsupport_orig=cpsupport_orig.reshape(B_q, C, -1).permute(2,0,1)
            c_query_orig = torch.cat((query_orig,cpsupport_orig),dim=2).permute(1,2,0).view(75,-1,5,5)
            c_query_orig = self.conv(c_query_orig).reshape(B_q, C, -1).permute(2,0,1)
            query_orig=self.agent(c_query_orig.permute(1,0,2)).permute(1,0,2)
            query_orig = self.args.lamda1*query + (1-self.args.lamda1)*query_orig.permute(1,2,0).reshape(B_q, C, H, W)

            noise_dimensions = (1, 5, 640, 5, 5)
            random_noise = torch.randn(noise_dimensions).cuda()
            z_support_save = z_support_save+random_noise
            z_proto_ext = z_support_save.squeeze(0).repeat(75, 1, 1, 1, 1)
            z_query_ext = z_query.repeat(5, 1, 1, 1, 1)
            z_query_ext = torch.transpose(z_query_ext, 0, 1)
            re = torch.cat((z_proto_ext, z_query_ext), 2).reshape(-1, 1280, 5, 5)
            score=self.r(re).view(-1,5)

            return self.emd_forward_1shot(support_orig, query_orig, score), torch.div(score, self.args.tau)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    def get_weight_vector(self, A, B):
        M = A.shape[0]
        N = B.shape[0]
        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])
        A = A.unsqueeze(1)
        B = B.unsqueeze(0)
        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)
        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        del M, N, A, B
        return combination

    def emd_forward_1shot(self, proto, query, score):
        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)
        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, score, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, score, solver='qpth')
        del proto, weight_1, weight_2, query, similarity_map
        return logits

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
        return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)
        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        del way, num_query, query, proto, feature_size
        return similarity_map

    def encode(self, x, dense=True):
        if x.shape.__len__() == 5:
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x
        else:
            x = self.encoder(x)
            if not dense:
                x = F.adaptive_avg_pool2d(x, 1)
            return x