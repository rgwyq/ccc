import torch as th
import torch.nn as nn
import os
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer

model_name = 'roberta-base'  # 替换为您使用的预训练模型的名称
model = AutoModel.from_pretrained(model_name)

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred   

folder_path = os.path.dirname("TGCN_2layers")
file_name = "ep_44_train_0.9262_val_0.7403_test_0.7316.pth"
file_path = os.path.join(folder_path, file_name)
checkpoint = th.load(file_path)

# 仅加载匹配的键
model_dict = model.state_dict()
checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

# 加载权重
model.load_state_dict(checkpoint, strict=False)

# 加载OHsumed数据集
current_dir = os.path.abspath(os.getcwd())
dataset = 'R52'
input_path = os.path.abspath(os.path.join(current_dir, '..', 'data', dataset))
output_path = os.path.abspath(os.path.join(current_dir, '..', 'data', dataset))

corpus = []
with open(input_path + '.clean.txt', 'r', encoding='gbk') as file:
    lines = file.readlines()
    for line in file:
        corpus.append(line.strip())
        
# 对每个文档/句子进行编码并获取BERT嵌入
embeddings = []
for doc in corpus:
    input_ids = torch.tensor([tokenizer.encode(doc, add_special_tokens=True)]).to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state
    embeddings.append(last_hidden_state.squeeze().cpu().numpy())

# Step 3: 计算单词-单词的边权重并生成_semantic.pkl文件

# 计算两个单词在语料库中具有语义关系的次数
word_semantic_count = dict()
word_total_count = dict()

for doc_emb in embeddings:
    word_idx = np.argmax(doc_emb, axis=0)  # 假设每个单词的嵌入为最大值的索引
    for i in range(len(word_idx)):
        for j in range(i + 1, len(word_idx)):
            word_i = tokenizer.convert_ids_to_tokens([word_idx[i]])[0]
            word_j = tokenizer.convert_ids_to_tokens([word_idx[j]])[0]
            if word_i not in word_semantic_count:
                word_semantic_count[word_i] = dict()
                word_total_count[word_i] = dict()
            if word_j not in word_semantic_count[word_i]:
                word_semantic_count[word_i][word_j] = 0
                word_total_count[word_i][word_j] = 0
            if word_i != word_j:
                word_semantic_count[word_i][word_j] += 1

# 计算边权重
word_semantic_weights = dict()
for word_i in word_semantic_count:
    word_semantic_weights[word_i] = dict()
    for word_j in word_semantic_count[word_i]:
        semantic_count = word_semantic_count[word_i][word_j]
        total_count = word_total_count[word_i][word_j]
        word_semantic_weights[word_i][word_j] = semantic_count / total_count

# 保存_semantic.pkl文件
output_file = open(output_path + '/{}_semantic.pkl'.format(dataset), 'wb')
pickle.dump(word_semantic_weights, output_file)
output_file.close()
print(f"Semantic weights are saved in {output_file}")
