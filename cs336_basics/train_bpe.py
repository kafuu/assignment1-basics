import os
import regex as re
import multiprocessing as process
import json
import time
from tqdm import tqdm
from collections import Counter

from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.common import gpt2_bytes_to_unicode

def _unwrap_get_initial_counts(args):
    # 把元组拆解成独立的参数传给原函数
    return get_initial_counts(*args)

def train_bpe_from_file(input_path:str,
                vocab_size:int,
                special_tokens:list[str])-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #通过原始语料训练bpe,将其写如成员变量vocab和merges中

    #1.初始化vocab,merges
    merges:list[tuple[bytes, bytes]] = []
    vocab:dict[int, bytes]={}
    i = 0 #i等于词表当前大小
    for special_token in special_tokens:
        vocab[i] = special_token.encode("utf-8")
        i+=1
    for b in range(256):
        vocab[i]=bytes([b])
        i+=1
    
    time_init = time.time()
    print("[1]正在预分词")
    #2.预分词
    counts = paralelled_get_initial_counts(input_path,special_tokens)
    print(f"[1]预分词结束！用时{time.time()-time_init:.3f}s",)


    print("[2]正在统计组合频率")
    #3.组合频率统计
    couple_counts:dict[tuple[bytes,bytes],int] = {}
    for word in counts:
        couples = list(zip(word[:-1],word[1:]))
        for couple in couples:
            if couple in couple_counts:
                couple_counts[couple] += counts[word]
            else: couple_counts[couple] = counts[word]
    print(f"[2]统计组合频率！用时{time.time()-time_init:.3f}s")

    
    #4.bpe训练，合并最常见组合
    print(f"[3]正在合并，共{vocab_size-i}组")
    init_i = i-1
    while(i<vocab_size):
        if (i-init_i)%10==0:
            print(f"[3]正在合并，当前第{i-init_i}组")
        #4.1寻找最常见组合
        max_freq = 0
        most_common_couple = None
        for couple in couple_counts:
            if couple_counts[couple] > max_freq:
                most_common_couple = couple
                max_freq = couple_counts[couple]
            if couple_counts[couple] == max_freq:
                if couple > most_common_couple:
                    most_common_couple = couple
        if most_common_couple == None:
            print(f"[3]无可合并内容！")
            break

        merges.append(most_common_couple)
        vocab[i] = most_common_couple[0]+most_common_couple[1]

        counts_copy = counts.copy()
        #4.2合并
        for word in counts:
            new_word:list[bytes]=[]
            new_word.append(word[0])
            last_merged = False
            for pos in range(1,len(word)):
                if last_merged == True:
                    last_merged = False
                    new_word.append(word[pos])
                    continue
                if(word[pos-1],word[pos])==most_common_couple:
                    new_word.append(new_word.pop()+word[pos])
                    last_merged = True
                else:
                    new_word.append(word[pos])



            #如果有合并
            if tuple(new_word)!=word:
                new_couple_counts:dict[tuple[bytes,bytes],int] = {}
                new_couples = list(zip(new_word[:-1],new_word[1:]))
                for couple in new_couples:
                    if couple in new_couple_counts:
                        new_couple_counts[couple] += counts[word]
                    else: new_couple_counts[couple] = counts[word]

                
                old_couple_counts:dict[tuple[bytes,bytes],int] = {}
                old_couples = list(zip(word[:-1],word[1:]))
                for couple in old_couples:
                    if couple in old_couple_counts:
                        old_couple_counts[couple] += counts[word]
                    else: old_couple_counts[couple] = counts[word]
                
                for couple in old_couple_counts:
                    couple_counts[couple]-=old_couple_counts[couple]
                    if couple_counts[couple] <=0 :
                        couple_counts.pop(couple)
                
                for couple in new_couple_counts:
                    if couple in couple_counts:
                        couple_counts[couple]+=new_couple_counts[couple]
                    else:
                        couple_counts[couple]=new_couple_counts[couple]

                if tuple(new_word) in counts:
                    counts_copy[tuple(new_word)] += counts_copy.pop(word)
                else:
                    counts_copy[tuple(new_word)] = counts_copy.pop(word) 
        counts = counts_copy
        i+=1
    print(f"[3]合并结束！用时{time.time()-time_init:.3f}s")

    return(vocab,merges)
    

def paralelled_get_initial_counts(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    tasks = []
    first_special_tok = special_tokens[0]
    with open(input_path,'rb')as f:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, first_special_tok.encode("utf-8"))
    for boundry in list(zip(boundaries[:-1], boundaries[1:])):
        tasks.append((input_path,special_tokens,boundry[0],boundry[1]))
    
    global_counts = Counter()

    pool = process.Pool(os.cpu_count())  # 限制在物理核心数以内

    # 迭代器吐出一个结果，我们就拦截一个
    for result in tqdm(pool.imap_unordered(_unwrap_get_initial_counts, tasks), total=len(tasks), desc="统计词频"):
        # 1. 瞬间合并：把子进程的小字典里的频次，全部累加到大账本里
        global_counts.update(result)
        
        # 2. 挫骨扬灰：这步极其关键！立刻在内存里销毁这个 30MB 的子字典
        del result 

    pool.close()
    pool.join()

    # 如果后续你的 BPE 逻辑必须用普通的 dict，强转一下即可
    counts = dict(global_counts)
    return counts

def get_initial_counts(input_path: str, 
                       special_tokens: list[str],
                       start: int,
                       end: int) -> dict[tuple[bytes], int]:
    counts = {}

    with open(input_path,"rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # 1. 动态生成特殊 Token 的切割刀（带捕获组）
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        split_pattern = "(" + "|".join(escaped_tokens) + ")"
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    # 2. 官方给定的普通文本预分词正则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 3. 路由分发
    for chunk in chunks:
        if not chunk:  # 过滤掉切分产生的空字符串
            continue
        
        if special_tokens and chunk in special_tokens:
            # 特殊 Token 享有豁免权，绝对不参与 BPE 的普通词频统计！
            continue
        else:
            # 纯普通文本，送给 PAT 绞肉机
            for match in re.finditer(PAT, chunk):
                word = match.group()
                # 转换为 (b'h', b'e', b'l', b'l', b'o') 格式的元组
                word_bytes = word.encode("utf-8")
                byte_tuple = tuple(word_bytes[i : i+1] for i in range(len(word_bytes)))
                
                if byte_tuple in counts:
                    counts[byte_tuple] += 1
                else:
                    counts[byte_tuple] = 1

    return counts

def save_tokenizer(vocab,merges,vocab_path: str, merges_path: str):
    """
    将 vocab 和 merges 序列化为符合测试脚本 get_tokenizer_from_vocab_merges_path 预期的格式。
    """
    # 拿到【字节 -> 字符】的映射字典
    byte_encoder = gpt2_bytes_to_unicode()

    # ==========================================
    # 1. 序列化 Vocab (保存为 JSON)
    # 官方期望读取到的格式: { "映射后的字符串": token_id }
    # ==========================================
    json_vocab = {}
    for token_id, token_bytes in vocab.items():
        # 将底层字节逐个转换为官方指定的 Unicode 字符，并拼成完整的字符串
        token_str = "".join([byte_encoder[b] for b in token_bytes])
        json_vocab[token_str] = token_id

    with open(vocab_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证特殊 Unicode 字符不会变成 \uXXXX 乱码
        json.dump(json_vocab, f, ensure_ascii=False, indent=2)


    # ==========================================
    # 2. 序列化 Merges (保存为纯文本)
    # 官方期望读取到的格式: 每行 "映射后的左词 映射后的右词"
    # ==========================================
    with open(merges_path, 'w', encoding='utf-8') as f:
        for b1, b2 in merges:
            str1 = "".join([byte_encoder[b] for b in b1])
            str2 = "".join([byte_encoder[b] for b in b2])
            f.write(f"{str1} {str2}\n")

if __name__ == "__main__":
    res = train_bpe_from_file("/home/chino/prog/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",10000,["<|endoftext|>"])
    save_tokenizer(res[0],res[1],"/home/chino/prog/assignment1-basics/cs336_basics/vocab.txt","/home/chino/prog/assignment1-basics/cs336_basics/merges.txt")
