import os
import regex as re
import multiprocessing as process

from cs336_basics.pretokenization_example import find_chunk_boundaries

class Tokenizer:
    def __init__(self,
                 vocab:dict[int, bytes]=None,
                 merges:list[tuple[bytes,bytes]]=None,
                 special_tokens:list[str]=None):
        self.vocab:dict[int, bytes]
        self.revocab:dict[bytes, int]
        self.merges:list[tuple[bytes,bytes]]

        #成员变量vocab,merges
        self.vocab={}#token2bytes
        self.revocab={}#bytes2tokens
        self.vocab = {i:bytes([i])for i in range(256)}
        self.merges=[]
        
        #通过已有训练结果直接建立tokenizer
        if vocab != None and merges != None:
            self.restore_tokenizer(vocab=vocab,merges=merges)
        if special_tokens != None:
            self.add_special_tokens(special_tokens)
        
    def pre_tokenization_from_file(self,
                         input_path: str,
                         start:int,
                         end:int) -> dict[tuple[bytes],int]:
        with open(input_path,"r") as f:
            f.seek(start)
            chunk = f.read(end - start)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        finder = re.finditer(PAT,chunk)#得到预分词后的迭代器
        counts:dict[tuple[bytes],int] = {}
        for match in finder:
            byte_tuple = tuple(match.group().encode("utf-8")[i : i+1] for i in range(len(match.group().encode("utf-8"))) )
            if byte_tuple in counts:
                counts[byte_tuple]+=1
            else:
                counts[byte_tuple]=1
        return counts

    def train_bpe_from_text(self,
                  original_text:str,
                  vocab_size:int,
                  special_tokens:list[str]):
        #通过原始语料训练bpe,将其写如成员变量vocab和merges中

        #1.初始化vocab和merges,将其清空，vocab中填入初始字节
        self.merges = []
        self.vocab = {}
        self.vocab = {i:bytes([i])for i in range(256)}
        self.vocab[256] = b'<|endoftext|>'

        #2.语料预处理
        #预分词，将str转换为counts：{bytes:times}
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        finder = re.finditer(PAT,original_text)#得到预分词后的迭代器
        counts:dict[tuple[bytes],int] = {}
        for match in finder:
            byte_tuple = tuple(match.group().encode("utf-8")[i : i+1] for i in range(len(match.group().encode("utf-8"))) )
            if byte_tuple in counts:
                counts[byte_tuple]+=1
            else:
                counts[byte_tuple]=1
        
        #3.循环k次，k=vocab_size - special_tokens - 256(初始字节)
        i = 256
        for i in range(257,vocab_size - len(special_tokens)):
            #3.1统计各个组合频度
            couple_counts:dict[tuple[bytes,bytes],int] = {}
            for word in counts:
                times = counts[word]
                for pos in range(len(word)-1):
                    if (word[pos],word[pos+1]) in couple_counts:
                        couple_counts[(word[pos],word[pos+1])]+=times
                    else:
                        couple_counts[(word[pos],word[pos+1])]=times
            if couple_counts=={}:#所有的word都是单个token了
                i-=1
                break
            max_freq = 0
            for couple in couple_counts:
                if couple_counts[couple] > max_freq:
                    max_freq = couple_counts[couple]
                    most_common_couple = couple
                if couple_counts[couple] == max_freq:
                        if couple > most_common_couple:
                            most_common_couple = couple

            #3.2将频度最大的组合记录进vocab，为其分配一个新ID，ID=循环次数+256
            self.vocab[i] = most_common_couple[0]+most_common_couple[1]

            #3.3将组合方式计入merges
            self.merges.append(most_common_couple)

            #3.4将原始语料中的组合进行替换，替换counts的key
            counts_copy = counts.copy()
            for word in counts_copy:
                merged_word:list[bytes]=[]
                merged_word.append(word[0])
                last_merged = False
                for pos in range(1,len(word)):
                    if last_merged:
                        last_merged = False
                        merged_word.append(word[pos])
                        continue
                    if (word[pos-1],word[pos]) == most_common_couple:
                        merged_word.append(merged_word.pop()+word[pos])
                        last_merged = True
                    else:
                        merged_word.append(word[pos])
                #替换
                new_key = tuple(merged_word)
                old_count = counts.pop(word)
                if new_key in counts:
                    counts[new_key] += old_count # 如果新词已存在，累加频次
                else:
                    counts[new_key] = old_count  # 否则直接赋值

        #4.循环结束，将special_tokens合并到vocab内
        for token in special_tokens:
            if i < vocab_size:
                i+=1
                self.vocab[i] = token
        #5.训练结束，将结果保存

    def train_bpe_from_file(self,
                    input_path:str,
                    vocab_size:int,
                    special_tokens:list[str]):
            #通过原始语料训练bpe,将其写如成员变量vocab和merges中

            self.merges = []
            self.vocab = {}
            self.vocab = {i:bytes([i])for i in range(256)}
            self.vocab[256] = b'<|endoftext|>'


            counts = self.paralelled_pretokenization_from_file(input_path)

            #3.循环k次，k=vocab_size - special_tokens - 256(初始字节)
            i = 256
            for i in range(257,vocab_size - len(special_tokens)):
                #3.1统计各个组合频度
                couple_counts:dict[tuple[bytes,bytes],int] = {}
                for word in counts:
                    times = counts[word]
                    for pos in range(len(word)-1):
                        if (word[pos],word[pos+1]) in couple_counts:
                            couple_counts[(word[pos],word[pos+1])]+=times
                        else:
                            couple_counts[(word[pos],word[pos+1])]=times
                if couple_counts=={}:#所有的word都是单个token了
                    i-=1
                    break
                max_freq = 0
                for couple in couple_counts:
                    if couple_counts[couple] > max_freq:
                        max_freq = couple_counts[couple]
                        most_common_couple = couple
                    if couple_counts[couple] == max_freq:
                        if couple > most_common_couple:
                            most_common_couple = couple

                #3.2将频度最大的组合记录进vocab，为其分配一个新ID，ID=循环次数+256
                self.vocab[i] = most_common_couple[0]+most_common_couple[1]

                #3.3将组合方式计入merges
                self.merges.append(most_common_couple)

                #3.4将原始语料中的组合进行替换，替换counts的key
                counts_copy = counts.copy()
                for word in counts_copy:
                    merged_word:list[bytes]=[]
                    merged_word.append(word[0])
                    last_merged = False
                    for pos in range(1,len(word)):
                        if last_merged:
                            last_merged = False
                            merged_word.append(word[pos])
                            continue
                        if (word[pos-1],word[pos]) == most_common_couple:
                            merged_word.append(merged_word.pop()+word[pos])
                            last_merged = True
                        else:
                            merged_word.append(word[pos])
                    #替换
                    new_key = tuple(merged_word)
                    old_count = counts.pop(word)
                    if new_key in counts:
                        counts[new_key] += old_count # 如果新词已存在，累加频次
                    else:
                        counts[new_key] = old_count  # 否则直接赋值

            #4.循环结束，将special_tokens合并到vocab内
            for token in special_tokens:
                if i < vocab_size:
                    i+=1
                    self.vocab[i] = token
            #5.训练结束，将结果保存
            
    def optimized_train_bpe_from_file(self,
                    input_path:str,
                    vocab_size:int,
                    special_tokens:list[str]):
            """a,b,a,b,a,b,a,b,a,b
            couple a,b:5 b,a:4
            合并后：
            ab,ab,ab,ab,ab
            couple ab,ab:4
            每次合并，前组合-1，本组合-1,后组合-1,前改后组合+1,后改后组合+1
            
            a,b,a,b,a,b,a,b,a,b
            couple a,b:5 b,a:4
            
            ab,a,b,a,b,a,b,a,b
            前无组合，中间a,b-1，后b,a-1
            改后前无组合，后ab,a+1
            即a,b:4 b,a:3 ab,a:1
            
            ab,ab,a,b,a,b,a,b
            a,b:3 b,a:2 ab,a:1 ab,ab:1
            …………"""
            #通过原始语料训练bpe,将其写如成员变量vocab和merges中

            self.merges = []
            self.vocab = {}
            self.vocab = {i:bytes([i])for i in range(256)}
            self.vocab[256] = b'<|endoftext|>'

            counts = self.paralelled_pretokenization_from_file(input_path)
            
            #初始couple_counts
            couple_counts:dict[tuple[bytes,bytes],int] = {}
            for word in counts:
                    times = counts[word]
                    for pos in range(len(word)-1):
                        if (word[pos],word[pos+1]) in couple_counts:
                            couple_counts[(word[pos],word[pos+1])]+=times
                        else:
                            couple_counts[(word[pos],word[pos+1])]=times
            
            #3.循环k次，k=vocab_size - special_tokens - 256(初始字节)
            i = 256
            for i in range(257,vocab_size - len(special_tokens)):
                if couple_counts=={}:#所有的word都是单个token了
                    i-=1
                    break
                max_freq = 0
                for couple in couple_counts:
                    if couple_counts[couple] > max_freq:
                        max_freq = couple_counts[couple]
                        most_common_couple = couple
                    if couple_counts[couple] == max_freq:
                        if couple > most_common_couple:
                            most_common_couple = couple
                

                #3.2将频度最大的组合记录进vocab，为其分配一个新ID，ID=循环次数+256
                self.vocab[i] = most_common_couple[0]+most_common_couple[1]

                #3.3将组合方式计入merges
                self.merges.append(most_common_couple)

                #3.4将原始语料中的组合进行替换，替换counts的key
                counts_copy = counts.copy()
                for word in counts_copy:
                    times = counts[word]
                    lenth = len(word)
                    curr = 1
                    merged_word:list[bytes] = []
                    merged_word.append(word[0])
                    last_merged = False 
                    last_couple_half = b''
                    next_couple_half = b''

                    while(curr<lenth):
                        if(last_merged):
                            last_merged = False 
                            last_merged_bytes = merged_word.pop()
                            merged_word.append(last_merged_bytes)
                            merged_word.append(word[curr])

                            next_couple = (next_couple_half,word[curr])
                            new_next_couple = (last_merged_bytes,word[curr])
                            couple_counts[next_couple]-=times
                            if couple_counts[next_couple] <= 0:
                                couple_counts.pop(next_couple)
                            if new_next_couple in couple_counts:
                                couple_counts[new_next_couple] += times
                            else: couple_counts[new_next_couple] = times
                            curr+=1
                            continue
                        if (word[curr-1],word[curr]) == most_common_couple:
                            last_merged = True
                            merged_word.append((merged_word.pop())+word[curr])

                            couple_counts[(word[curr-1],word[curr])]-=times
                            if couple_counts[(word[curr-1],word[curr])]<=0:
                                couple_counts.pop((word[curr-1],word[curr]))

                            next_couple_half = word[curr]
                            if curr > 1:
                                last_couple_half = word[curr-2]
                                last_couple = (last_couple_half,word[curr-1])
                                new_last_couple = (last_couple_half,word[curr-1]+word[curr])
                                couple_counts[last_couple] -= times
                                if couple_counts[last_couple] <= 0:
                                    couple_counts.pop(last_couple)
                                if new_last_couple in couple_counts:
                                    couple_counts[new_last_couple]+=times
                                else: couple_counts[new_last_couple]=times
                        else:
                            merged_word.append(word[curr])
                        curr+=1    
                    
                    #替换
                    if tuple(merged_word) != word:
                        new_key = tuple(merged_word)
                        old_count = counts.pop(word)
                        if new_key in counts:
                            counts[new_key] += old_count # 如果新词已存在，累加频次
                        else:
                            counts[new_key] = old_count  # 否则直接赋值

            #4.循环结束，将special_tokens合并到vocab内
            for token in special_tokens:
                if i < vocab_size:
                    i+=1
                    self.vocab[i] = token
            #5.训练结束，将结果保存

    def paralelled_pretokenization_from_file(self,
                                       input_path: str) -> dict[tuple[bytes],int]:
        core_num = os.cpu_count()
        
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, core_num, b"<|endoftext|>")
            couple = list(zip(boundaries[:-1], boundaries[1:]))
            tasks_paras = [(input_path,start,end)for start,end in couple]
        
        process_pool = process.Pool(processes=core_num)
        counts_list = process_pool.starmap(self.pre_tokenization_from_file,tasks_paras)
        return self.merge_counts(counts_list)

    def merge_counts(self,
                     counts_list:list[dict[tuple[bytes],int]])->dict[tuple[bytes],int]:
        counts:dict[tuple[bytes],int] = {}
        for tmp_counts in counts_list:
            for count in tmp_counts:
                if count in counts:
                    counts[count]+=tmp_counts[count]
                else:
                    counts[count]=tmp_counts[count]
        return counts

    def restore_tokenizer(self,
                          vocab:dict[int, bytes],
                          merges:list[tuple[bytes,bytes]]):
        #通过vocab和merges建立tokenizers
        self.vocab=vocab
        self.merges=merges
        pass
    
    def add_special_tokens(self,
                   tokens:list[str]):
        #添加新的tokens
        lenth = len(self.vocab)
        for word in tokens:
            self.vocab[lenth]=word
            lenth+=1
        pass

    def save_tokenizer(self,
                       path:str):
        #将vocab和merges保存为文件
        pass

    def encode(self,
               text:str)->list[int]:
        #将字符串转换为ID序列
        #0.文本切分
        self.generate_revocab()
        spilted_text = re.split("(<\\|endoftext\\|>)",text)
        encoded_text=[]
        for _text in spilted_text:
            if _text == "<|endoftext|>":
                encoded_text.append(self.revocab[_text.encode("utf-8")])
                continue
            if _text == "":
                continue
            #1.预分词，并转换为字节序列 预分词->token的list
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            finder = re.finditer(PAT,_text)#得到预分词后的迭代器
            substitution:dict[str,list[int]] = {}
            for match in finder:
                if match.group() not in substitution:
                    substitution[match.group()]=[]
            #2.重复merges的反向过程，先merges substitution：o(len(merges)*len(substitution))
            #2.1重复使用merges合并操作每一个substitution的key

            for sub in substitution:
                tmp_list_bytes = list(sub.encode("utf-8")[i : i+1] for i in range(len(sub.encode("utf-8"))) )
                for merge in self.merges:
                    new_list_bytes=[]
                    last_merge=False
                    #字节组数量小于2则不合并
                    if len(tmp_list_bytes) < 2:continue
                    for pos in range(len(tmp_list_bytes)-1):#len(tmp_list_bytes)-1次合并尝试
                        if last_merge == True : 
                            last_merge = False
                            continue#上一次合并成功，无需合并
                        if (tmp_list_bytes[pos],tmp_list_bytes[pos+1]) == merge:#匹配成功
                            new_list_bytes.append(tmp_list_bytes[pos]+tmp_list_bytes[pos+1])
                            last_merge=True
                        else:
                            new_list_bytes.append(tmp_list_bytes[pos])
                            last_merge=False
                    if last_merge==False:
                        new_list_bytes.append(tmp_list_bytes[pos+1])
                    #一次合并结束，将newlist写回
                    tmp_list_bytes = new_list_bytes
                #对该sub合并结束，查找list_bytes对应的token
                for b in tmp_list_bytes:
                    substitution[sub].append(self.revocab[b])  
            #3.用substitution替换原文本，依然是使用迭代器
            finder2 = re.finditer(PAT,_text)#得到预分词后的迭代器
            for match in finder2:
                for token in substitution[match.group()]:
                    encoded_text.append(token)
        return encoded_text

    def generate_revocab(self):
        self.revocab={}
        for word in self.vocab:
            self.revocab[self.vocab[word]]=word
        
    def decode(self,
               ids:list[int])->str:
        #将ID序列转换为字符串
        #遍历ids,查vocab表，转换为字节序列
        tmp_bytes=b""
        for token in ids:
            tmp_bytes+=self.vocab[token]
        return tmp_bytes.decode("utf-8",errors="replace")

if __name__ == "__main__":
    test_text = "Hello word! how are you? Hello Hello Hello lo lo lo lo lo lo lo"
    print(f"\n[1] 原始文本: {test_text}")

    #文本转换为字节串：
    utf8_encoded = test_text.encode("utf-8")
    print(f"\n[2] 文本转换为字节串{utf8_encoded}")

    l = list(utf8_encoded)
    print(f"\n[3] 字节串list{l}")

    #解码，字节串转换为字符串
    print(f"\n[4] 字节串编码为字符串 {utf8_encoded.decode("utf-8")}")

    t = Tokenizer()

    t.train_bpe_from_text(test_text,299,["TEST_SPE_TOK","TEST_SPE_TOK2","TEST_SPE_TOK3"])
    print(f"\n[5] vocab：{t.vocab}")
    print(f"\n[5] merges：{t.merges}")

    #print(t.encode(test_text))
    #print(t.decode(t.encode(test_text)))

    t.train_bpe_from_file("/home/chino/prog/assignment1-basics/data/test_corpus.txt",400,[])
    
    #print(t.vocab)
    print(t.merges)

    t2 = Tokenizer()
    t2.optimized_train_bpe_from_file("/home/chino/prog/assignment1-basics/data/test_corpus.txt",400,[])
    print(t2.merges)

    test_txt = """Hello world! This is a robust test for the BPE encode and decode cycle.
Did you know that 42 is the answer? Or maybe 3.14159265...
Let's test some Punctuation: (brackets), [square], {braces}, and "quotes"!
What about multiple spaces? Here     they      are.
And empty lines?



Now, let's test repeated characters: Woooooow!!! That is sooooo coooool.
Don't forget contractions; they're notoriously tricky for regex pre-tokenizers.
How about some non-ASCII bytes to test the UTF-8 encoding? 
Let's grab a Café and a résumé. Maybe even a Python 🐍!
Finally, we must test if the special token is preserved intact and not shattered into pieces:
<|endoftext|>
If your text == decode(encode(text)) assertion survives this gauntlet, your Tokenizer is absolutely rock solid."""
    with open("/home/chino/prog/assignment1-basics/data/output.txt",'w') as f:
        f.write(t.decode(t.encode(test_txt)))

    for i in range(len(t.merges)):
        print(t.merges[i],' ',t2.merges[i],'\n')

    for i in range(len(t.vocab)):
        print(t.vocab[i],' ',t2.vocab[i],'\n')

    #t1 = Tokenizer(t.vocab,t.merges,["TEST_SPE_TOK","TEST_SPE_TOK2","TEST_SPE_TOK3"])
    #print(t1.vocab)