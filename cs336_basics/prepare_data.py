import numpy as np
from cs336_basics.tokenizer import Tokenizer

def prepare_data_streaming(txt_file_path, output_bin_path, tokenizer, chunk_size=1024*1024):
    """
    chunk_size: 内存里最多同时缓存多少个 Token。
    1024*1024 个 Token，按 uint16 算，物理内存只占区区 2MB！
    """
    print(f"开始流式处理: {txt_file_path}")
    
    # 1. 构造一个逐行读取的生成器 (这就是你的 Iterable[str])
    # 它每次只从硬盘读一行文本，绝不多吃内存
    def line_generator():
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
                
    # 2. 将生成器喂给你的流式编码函数，拿到 Iterator[int]
    token_iterator = tokenizer.encode_iterable(line_generator())
    
    # 3. 以纯二进制追加模式 ('wb') 打开目标文件
    with open(output_bin_path, 'wb') as f_out:
        chunk = []
        total_tokens = 0
        
        # 像流水线一样，一个个接收 Token
        for token_id in token_iterator:
            chunk.append(token_id)
            
            # 内存里的缓存达到了阈值，执行物理落盘
            if len(chunk) >= chunk_size:
                # 强行塑形成 uint16 的二进制砖块，砸进硬盘
                np.array(chunk, dtype=np.uint16).tofile(f_out)
                total_tokens += len(chunk)
                # 清空缓存，释放内存
                chunk.clear()
                print(f"已落盘 {total_tokens} 个 tokens...")
        
        # 4. 循环结束后，别忘了把最后没攒够 chunk_size 的“小尾巴”也写进去
        if chunk:
            np.array(chunk, dtype=np.uint16).tofile(f_out)
            total_tokens += len(chunk)
            
    print(f"流式处理彻底完成！生成文件: {output_bin_path}")
    print(f"总 Token 数: {total_tokens}")

if __name__ == "__main__":
    print("开始处理文本")
    tokenizer = Tokenizer.from_files(vocab_filepath="/home/chino/prog/assignment1-basics/cs336_basics/vocab_tiny.txt",
                                     merges_filepath="/home/chino/prog/assignment1-basics/cs336_basics/merges_tiny.txt")
    prepare_data_streaming("/home/chino/prog/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt","/home/chino/prog/assignment1-basics/data/train.bin",tokenizer)
    prepare_data_streaming("/home/chino/prog/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt","/home/chino/prog/assignment1-basics/data/valid.bin",tokenizer)
    print("文本处理结束")