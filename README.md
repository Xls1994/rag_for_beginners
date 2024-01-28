# rag_for_beginners

## 简化版 

参考   ``` rag_with_openai_embedding.py ```，只使用了OpenAI的embedding接口，没有调用向量模型，实现简单。</br>
可以先跑通这个流程。

## 安装依赖

```bash
pip install transformers torch torchvision
pip install numpy chromadb tqdm openai
```
## 向量数据库
参考 149行 ```ChromaDB``` 这个类的实现。

## 向量模型调用

如果你当前环境联网，下载模型权重很慢，可以使用本地下载的方式；
将下面的大模型权重下载下来，解压缩，放在一个目录里，比如D:\\AAA，然后对下面这个代码的地址修改为D:\\AAA。
```python
BGE_MODEL_PATH = "D:\\codes\\bge-large-zh"  #替换成你自己的实际地址
```

## 大模型权重下载地址

链接：https://pan.baidu.com/s/1XMVmn0C57D7A4mfOjQnl1g </br>
提取码：34r8
