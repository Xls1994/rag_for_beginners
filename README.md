# rag_for_beginners

## 安装依赖

```bash
pip install transformers torch torchvision
pip install numpy chromadb tqdm openai
```
## 向量数据库
参考 149行 ```ChromaDB``` 这个类的实现。

## 向量模型调用

如果你当前环境联网，下载模型权重很慢，可以使用本地下载的方式；
将下面的大模型权重下载下来放在一个目录，然后在代码中进行修改并指定成你自己的目录。
```python
BGE_MODEL_PATH = "D:\\codes\\bge-large-zh"  #替换成你自己的实际地址
```

## 大模型权重下载地址

链接：https://pan.baidu.com/s/1XMVmn0C57D7A4mfOjQnl1g </br>
提取码：34r8
