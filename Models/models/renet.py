import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *
from .CIFE import ResNet
# from .agentattention import AgentAttention
from .gaiatent import AgentAttention
from .SeqAttention import SeqAttention
from .fusion import iAFF
from .vmambablock import SS2D


class TransformerDecoder(nn.Module):
    def __init__(self, args, dim, mlp_hidden_dim, norm_layer):
        super().__init__()
        self.args = args
        self.qkv = nn.Linear(dim, dim)
        self.norm = norm_layer(dim)

    def forward(self, tgt, mem):  # torch.Size([1, nq, c]) torch.Size([1, ns, c])
        if self.args.dataset == 'fc100':
            query = self.normalize_feature(self.qkv(tgt))  # torch.Size([1, nq, c])
            key = self.normalize_feature(self.qkv(mem))  # torch.Size([1, ns, c])
            value = self.normalize_feature(self.qkv(mem))
        else:
            query = self.qkv(self.norm(tgt))  # torch.Size([1, nq, c])
            key = self.qkv(self.norm(mem))  # torch.Size([1, ns, c])
            value = self.qkv(self.norm(mem))

        score = torch.einsum('i b c, i d c -> i b d', query, key).softmax(dim=-1)
        tgt = torch.einsum('i b c, i c d -> i b d', score, value)
        # del query, key
        return score, tgt

    def normalize_feature(self, x):
        return F.normalize(x, p=2, dim=2, eps=1e-12)


class sampleFormer(nn.Module):
    """docstring for sampleFormer Block"""

    def __init__(self, args, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(sampleFormer, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=args.dp2)
        # QS-Decoder
        self.transformer_decoder = TransformerDecoder(args, dim, mlp_hidden_dim, norm_layer)
        self.sa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=args.dp3)

    def forward(self, proto, query, resize=False):  # torch.Size([ns, c]) torch.Size([nq, c])
        # B, N, C = query.shape
        mem = self.encoder_layer(proto).transpose(0, 1)
        tgt, _ = self.sa(query, query, query)

        score, tgt = self.transformer_decoder((tgt + query).transpose(0, 1), mem)
        score = score.squeeze(0)

        # del mem
        return score, tgt


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
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

    def forward(self, x):
        # x = self.fu(x,x)
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class DeepEMD(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args
        self.depth = args.N
        kwargs = {}
        # self.vm = SS2D(d_model=640, dropout=True, d_state=16, **kwargs)

        self.encoder = ResNet(args=args)
        dim = 640
        self.r = RelationNetwork(64, 8)
        # self.rh= nn.Linear(1280,640)
        # self.conv = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1, stride=1, padding=0)
        # self.atten1 = SeqAttention(640, 1, 1, sqa_type="linear", residual_mode=True)
        self.fu = iAFF()
        # self.conv2 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=1)
        self.fc1 = nn.Linear(640, 64)
        self.fc2 = nn.Linear(64, 1)
        if self.mode == 'pre_train':
            if args.dataset == 'cub':
                self.fc = nn.Linear(dim, self.args.num_class)
            else:
                self.fc = nn.Linear(dim, self.args.num_class)


        self.agent = AgentAttention(dim=640)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.mode == 'meta':
            support, query = input  # torch.Size([1, ns, c, h, w]) torch.Size([nq, c, h, w])
            support_orig,query_orig=support.squeeze(0),query
            sample_features_ext = support_orig.unsqueeze(0).repeat(75, 1, 1, 1, 1)
            batch_features_ext = query_orig.unsqueeze(0).repeat(5, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 640 * 2, 5, 5)
            # relation_pairs = self.conv(relation_pairs)
            # score = self.atten1(relation_pairs).view(-1, 5)
            score = self.r(relation_pairs).view(-1,5)
            # return relations
            # score = 0
            B_s, C, H, W = support_orig.shape
            B_q, _, _, _ = query_orig.shape
            support_orig = support_orig.reshape(B_s, C, -1).permute(2, 0, 1)
            query_orig = query_orig.reshape(B_q, C, -1).permute(2, 0, 1)  # 25,75,640

            # support_orig, _ = self.patchFormer(support_orig, support_orig, support_orig)
            support_orig = self.agent(support_orig.squeeze(0).permute(1, 0, 2)).permute(1, 0, 2)
            support_orig = self.args.lamda1 * support.squeeze(0) + (1 - self.args.lamda1) * support_orig.permute(1, 2,
                                                                                                                 0).reshape(
                B_s, C, H, W)
            # query_orig, _ = self.patchFormer(query_orig, query_orig, query_orig)
            query_orig = self.agent(query_orig.permute(1, 0, 2)).permute(1, 0, 2)
            query_orig = self.args.lamda1 * query + (1 - self.args.lamda1) * query_orig.permute(1, 2, 0).reshape(B_q, C,
                                                                                                                 H, W)

            # return self.emd_forward_1shot(support_orig, query_orig, score)
            # return  score,score
            return self.emd_forward_1shot(support_orig, query_orig, score), torch.div(score, self.args.tau)
            # return self.emd_forward_1shot(support_orig, query_orig, score), score

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

    # def emd_forward_sfc(self, proto, query, score):
    #     proto = proto.squeeze(0)
    #
    #     weight_1 = self.get_weight_vector(query, proto)
    #     weight_2 = self.get_weight_vector(proto, query)
    #
    #     proto = self.normalize_feature(proto)
    #     query = self.normalize_feature(query)
    #
    #     similarity_map = self.get_similiarity_map(proto, query)  # 4,5,25,25
    #     logits = similarity_map.sum(-1).sum(-1)
    #     # return logits
    #     if self.args.solver == 'opencv' or (not self.training):
    #         logits = self.get_emd_distance(similarity_map, weight_1, weight_2, score, solver='opencv')
    #     else:
    #         logits = self.get_emd_distance(similarity_map, weight_1, weight_2, score, solver='qpth')
    #     del proto, weight_1, weight_2, query, similarity_map
    #     return logits  # 4,5

    def get_sfc(self, support):
        support = support.squeeze(0)

        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0)

        support = self.fu(SFC, SFC)

        # support = self.conv(support)
        return support
        # SFC = nn.Parameter(SFC.detach(), requires_grad=True)
        # return SFC
        # support = support.squeeze(0)
        # # init the proto
        # support = self.agent(support.view(25, 640, 25).permute(0, 2, 1)).permute(0, 2, 1).view(25, 640, 5, 5)
        # SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        # SFC = nn.Parameter(SFC.detach(), requires_grad=True)
        # return SFC
        # optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)
        #
        # # crate label for finetune
        # label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        # label_shot = label_shot.type(torch.cuda.LongTensor)
        #
        # with torch.enable_grad():
        #     for k in range(0, self.args.sfc_update_step):
        #         rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
        #         for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
        #             selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
        #             batch_shot = support[selected_id, :]
        #             batch_label = label_shot[selected_id]
        #             optimizer.zero_grad()
        #             logits = self.emd_forward_sfc(SFC, batch_shot.detach(), score=0)
        #             loss = F.cross_entropy(logits, batch_label)
        #             loss.backward()
        #             optimizer.step()
        # del support, label_shot, k, j, rand_id, batch_shot, batch_label
        # return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, score, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight_1.shape[-1]
        # a=0.5
        # return score
        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

            temperature = (self.args.temperature / num_node)
            logitis = (1 - self.args.lamda) * similarity_map.sum(-1).sum(
                -1) * temperature + self.args.lamda * score * self.args.temperature2

            del num_query, num_node, num_proto, i, j, flow, temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2, form=self.args.form,
                                          l2_strength=self.args.l2_strength)

            logitis = (flows * similarity_map).view(num_query, num_proto, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
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

        if x.shape.__len__() == 5:  # batch of image patches
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
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out


if __name__ == '__main__':
    from thop import profile

    model = DeepEMD()
    input = torch.FloatTensor(1, 3, 84, 84)
    macs, paras = profile(model, inputs=(input,))
    print('FLOPs=', macs, (macs / 1e9))
    # bs=80: FLOPs= 281840271360.0  281.84027136
    # bs=1:  FLOPs= 3523003392.0    3.523003392
    print('Parameters=', paras, (paras / 1e6))  # Parameters= 12424320.0 12.42432
    print(aa)