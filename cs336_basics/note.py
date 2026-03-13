import regex as re

class Test:
    def __init__(self,value:int):
        self.value = value
        pass

    @classmethod
    def from_text(cls,value:str):
        return cls(int(value))

if __name__ == "__main__":
    #字符串转换为字节串
    test_str = "Hello word!"
    converted_bytes = test_str.encode("utf-8")
    print (converted_bytes)

    #字典类型：key：value。，或者叫做哈希表
    dict_test = {"a":114}
    print(dict_test["a"])#查找字典
    print(dict_test.get("a"))#软查找，如果查找失败不会报错
    print(dict_test.get("b"))#软查找，如果查找失败不会报错，返回none

    #迭代器，在局部访问等效于使用显示数据结构，只是无法纵览全局
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    print(re.findall(PAT,test_str))
    for i in re.findall(PAT,test_str):
        print(i)
    finder = re.finditer(PAT,test_str)#等效
    for i in finder:
        print(i.group())

   #拉链：
    zp = list(zip(range(0,4),range(1,5)))
    print(zp)
    #列表推导式
    lst = [("tst_str",start,end)for start,end in zp]#遍历元组，解包
    print (lst)

    byte_tuple = tuple(test_str.encode("utf-8")[i : i+1] for i in range(len(test_str.encode("utf-8"))) )
    zp = list(zip(byte_tuple[:-1],byte_tuple[1:]))
    print(zp)
    
    test_str = "Hello <|endoftext|> world"
    print(re.split("(<\\|endoftext\\|>)",test_str))

    tp = (b'ab',b'cd',b'ef')
    
    print(tp)

    t = Test(1)
    t2 = Test.from_text('1')

    print(type(t2.value))



"""
x1=1 2 3 4：属性a,b,c,d

WQ*WKT=A=
  a b c d
a 1 4 2 6
b 7 3 2 1
c 5 4 3 2
d 3 2 1 4

x2=1 3 4 5：属性a,b,c,d

x1Ax2T=

         1 4 2 6   1
         7 3 2 1   3
         5 4 3 2   4
         3 2 1 4   5
1 2 3 4  

1*1*1+2*7*1+3*5*1+4*3*1+
1*4*3+2*3*3+3*4*3+4*2*3+
1*2*4+2*2*4+3*3*4+4*1*4+
1*6*5+2*1*5+3*2*5+4*4*5

for i,j in range(len(x1)),range(len(x2)):
    从x1中取出第i个数字a，
    从x2中取出第j个数字b，
    查WQ*WKT表第i行第j列的数字c
    sigma(a*b*c)

"""

k = [i for i in range(1,int(4/2)+1)]
print(k)

ls = [0.0335, 0.0925, 0.0000, 0.2159, 0.1585, 0.1647, 0.0000, 0.0860, 0.0141,0.0000, 0.0000, 0.0000, 0.0991, 0.0455, 0.0504, 0.0398]
print(sum(ls))