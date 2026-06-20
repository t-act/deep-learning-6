from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    while i < len(ids):
        # merge ruleに合致
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        # skip
        else:
            merged_ids.append(ids[i])
            i += 1
    
    return merged_ids

def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))

    # マージ回数を決定
    num_merges = vocab_size - 256  # 256は初期の語彙サイズ
    merge_rules = {}

    for step in range(num_merges):
        # 隣接ペアの統計を取得
        counts = count_pairs(ids)

        # ペアが存在しない場合の処理
        if not counts:
            break

        # 最頻出ペアを選択
        best_pair = max(counts, key=counts.get)

        # 新しいトークンIDを割り当て
        new_id = 256 + step   # 256は初期語彙で使用しているため256+1からスタート
        merge_rules[best_pair] = new_id

        # マージを実行
        ids = merge(ids, best_pair, new_id)
    
    return merge_rules

# 使用例
text = "Hello world! This is BPE training."
merge_rules = train_bpe(text, vocab_size=260)
print(merge_rules)  # {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}
    
# for k, v in zip(merge_rules.keys(), merge_rules.values()):
#     print(f"{v}: ", bytes([k[0], k[1]]).decode("utf-8"))

id_to_bytes = {i: bytes([i]) for i in range(256)}
for (id1, id2), new_id in merge_rules.items():
    id_to_bytes[new_id] = id_to_bytes[id1] + id_to_bytes[id2]
    print(f"{new_id}: {id_to_bytes[new_id]}")
