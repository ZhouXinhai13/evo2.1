# coding=UTF-8
import json
import csv
import argparse
import os

root_dir = '../../../'
# vocab_dir = root_dir + 'datasets/data_preprocessed/icews14/vocab/'
# dir = root_dir + 'datasets/data_preprocessed/icews14/'
# vocab_dir = root_dir + 'datasets/data_preprocessed/icews05-15-7000/vocab/'
# dir = root_dir + 'datasets/data_preprocessed/icews05-15-7000/'
# vocab_dir = root_dir + 'datasets/data_preprocessed/icews05-15-7000/vocab/'
# dir = root_dir + 'datasets/data_preprocessed/icews05-15-7000/'
# vocab_dir = root_dir + 'datasets/data_preprocessed/icews18-7000/vocab/'
# dir = root_dir + 'datasets/data_preprocessed/icews18-7000/'
vocab_dir = root_dir + 'datasets/data_preprocessed/gdelt/vocab/'
dir = root_dir + 'datasets/data_preprocessed/gdelt/'
os.makedirs(vocab_dir)

entity_vocab = {}
relation_vocab = {}
time_vocab = {}

entity_vocab['PAD'] = len(entity_vocab)
entity_vocab['UNK'] = len(entity_vocab)
relation_vocab['PAD'] = len(relation_vocab)
relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
relation_vocab['NO_OP'] = len(relation_vocab)
relation_vocab['UNK'] = len(relation_vocab)

entity_counter = len(entity_vocab)
relation_counter = len(relation_vocab)
time_counter = len(time_vocab)

for f in ['graph.txt']:
    with open(dir + f) as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        for line in csv_file:
            e1, r, e2, t = line
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1
            if r + '_inverse' not in relation_vocab:
                relation_vocab[r + '_inverse'] = relation_counter
                relation_counter += 1
            if t not in time_vocab:
                time_vocab[t] = time_counter
                time_counter += 1

with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout, ensure_ascii=True, sort_keys=True)

with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout, ensure_ascii=True, sort_keys=True)

with open(vocab_dir + 'time_vocab.json', 'w') as fout:
    json.dump(time_vocab, fout, ensure_ascii=True, sort_keys=True)
# coding=UTF-8
# import json
# import csv
# import argparse
# import os

# root_dir = '../../../'
# vocab_dir = root_dir + 'datasets/data_preprocessed/gdelt/vocab/'
# dir = root_dir + 'datasets/data_preprocessed/gdelt/'

# # 创建词汇表目录（如果不存在）
# os.makedirs(vocab_dir, exist_ok=True)

# entity_vocab = {}
# relation_vocab = {}
# time_vocab = {}

# # 添加特殊标记
# entity_vocab['PAD'] = len(entity_vocab)
# entity_vocab['UNK'] = len(entity_vocab)
# relation_vocab['PAD'] = len(relation_vocab)
# relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
# relation_vocab['NO_OP'] = len(relation_vocab)
# relation_vocab['UNK'] = len(relation_vocab)

# entity_counter = len(entity_vocab)
# relation_counter = len(relation_vocab)
# time_counter = len(time_vocab)

# # 读取 graph.txt
# for f in ['graph.txt']:
#     filepath = dir + f
#     print(f"Processing {f}...")
    
#     with open(filepath) as raw_file:
#         csv_file = csv.reader(raw_file, delimiter='\t')
#         line_num = 0
#         for line in csv_file:
#             line_num += 1
            
#             # 跳过注释行（以 # 开头）
#             if len(line) > 0 and line[0].startswith('#'):
#                 continue
            
#             # 跳过空行
#             if len(line) == 0:
#                 continue
            
#             # 验证格式
#             if len(line) != 4:
#                 print(f"  Warning: Line {line_num} has {len(line)} columns, skipping")
#                 continue
            
#             e1, r, e2, t = line
            
#             # 添加实体
#             if e1 not in entity_vocab:
#                 entity_vocab[e1] = entity_counter
#                 entity_counter += 1
#             if e2 not in entity_vocab:
#                 entity_vocab[e2] = entity_counter
#                 entity_counter += 1
            
#             # 添加关系（包括反向关系）
#             if r not in relation_vocab:
#                 relation_vocab[r] = relation_counter
#                 relation_counter += 1
#             if r + '_inverse' not in relation_vocab:
#                 relation_vocab[r + '_inverse'] = relation_counter
#                 relation_counter += 1
            
#             # 添加时间戳
#             if t not in time_vocab:
#                 time_vocab[t] = time_counter
#                 time_counter += 1

# # 打印统计信息
# print(f"\nVocabulary Statistics:")
# print(f"  Entities: {len(entity_vocab)} (including PAD, UNK)")
# print(f"  Relations: {len(relation_vocab)} (including PAD, special tokens, and inverse)")
# print(f"  Timestamps: {len(time_vocab)}")

# # 保存词汇表
# with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
#     json.dump(entity_vocab, fout, ensure_ascii=True, sort_keys=True, indent=2)
# print(f"✓ Saved {vocab_dir}entity_vocab.json")

# with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
#     json.dump(relation_vocab, fout, ensure_ascii=True, sort_keys=True, indent=2)
# print(f"✓ Saved {vocab_dir}relation_vocab.json")

# with open(vocab_dir + 'time_vocab.json', 'w') as fout:
#     json.dump(time_vocab, fout, ensure_ascii=True, sort_keys=True, indent=2)
# print(f"✓ Saved {vocab_dir}time_vocab.json")

# print("\n✓ Vocabulary creation completed!")
