# @author yangyunlong
import chromadb
import openai
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import cast, List, Union
from transformers import AutoModel, AutoTokenizer

class FlagModel:
   def __init__(
           self,
           model_name_or_path: str = None,  # 预训练模型的名称或路径
           pooling_method: str = 'cls',  # 池化方法，'cls' 表示使用 [CLS] 标记，'mean' 表示使用均值池化
           normalize_embeddings: bool = True,  # 是否对嵌入向量进行归一化
           query_instruction_for_retrieval: str = None,  # 检索任务中查询指令的前缀
           use_fp16: bool = True  # 是否使用半精度浮点数（16位）进行计算，以节省内存和加速
   ) -> None:
       # 初始化tokenizer和model
       self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
       self.model = AutoModel.from_pretrained(model_name_or_path)
       self.query_instruction_for_retrieval = query_instruction_for_retrieval
       self.normalize_embeddings = normalize_embeddings
       self.pooling_method = pooling_method

       # 根据是否可用CUDA或MPS设置设备
       if torch.cuda.is_available():
           self.device = torch.device("cuda")
       elif torch.backends.mps.is_available():
           self.device = torch.device("mps")
       else:
           self.device = torch.device("cpu")
           use_fp16 = False
       if use_fp16: self.model.half()
       self.model = self.model.to(self.device)

       # 检查GPU数量，如果多于1个，则使用DataParallel进行并行计算
       self.num_gpus = torch.cuda.device_count()
       if self.num_gpus > 1:
           print(f"----------using {self.num_gpus}*GPUs----------")
           self.model = torch.nn.DataParallel(self.model)

   def encode_queries(self, queries: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512) -> np.ndarray:
       '''
       用于检索任务的函数，将查询文本编码为嵌入向量
       如果存在查询指令，将其添加到查询文本前
       '''
       if self.query_instruction_for_retrieval is not None:
           if isinstance(queries, str):
               input_texts = self.query_instruction_for_retrieval + queries
           else:
               input_texts = [
                   '{}{}'.format(self.query_instruction_for_retrieval, q) for q
                   in queries]
       else:
           input_texts = queries
       return self.encode(input_texts, batch_size=batch_size,
                          max_length=max_length)

   def encode_corpus(self,
                     corpus: Union[List[str], str],
                     batch_size: int = 256,
                     max_length: int = 512) -> np.ndarray:
       '''
       用于检索任务的函数，将语料库文本编码为嵌入向量
       '''
       return self.encode(corpus, batch_size=batch_size, max_length=max_length)

   @torch.no_grad()  # 装饰器，表示这个函数不会计算梯度，用于推理
   def encode(self, sentences: Union[List[str], str], batch_size: int = 256,
              max_length: int = 512) -> np.ndarray:
       '''
       用于编码句子的函数，将文本转换为嵌入向量
       '''
       if self.num_gpus > 0:
           batch_size = batch_size * self.num_gpus  # 如果有多个GPU，增加批次大小

       # 初始化一些变量
       self.model.eval()  # 设置模型为评估模式
       input_was_string = False  # 标记输入是否为单个字符串
       if isinstance(sentences, str):
           sentences = [sentences]
           input_was_string = True

       # 编码句子并收集所有嵌入向量
       all_embeddings = []
       for start_index in tqdm(range(0, len(sentences), batch_size),
                               desc="Inference Embeddings",  # tqdm用于显示进度条
                               disable=len(sentences) < 256):  # 如果句子数量少于256，不显示进度条
           sentences_batch = sentences[start_index:start_index + batch_size]
           inputs = self.tokenizer(
               sentences_batch,
               padding=True,
               truncation=True,
               return_tensors='pt',  # 返回PyTorch张量
               max_length=max_length,
           ).to(self.device)  # 将输入转换到指定设备
           last_hidden_state = self.model(**inputs,
                                          return_dict=True).last_hidden_state  # 获取最后一层的隐藏状态
           embeddings = self.pooling(last_hidden_state,
                                     inputs['attention_mask'])  # 应用池化方法
           if self.normalize_embeddings:
               embeddings = torch.nn.functional.normalize(embeddings, dim=-1)  # 归一化嵌入向量
           embeddings = cast(torch.Tensor, embeddings)  # 转换为PyTorch张量
           all_embeddings.append(embeddings.cpu().numpy())  # 将张量转换为NumPy数组

       # 将所有批次的嵌入向量拼接起来
       all_embeddings = np.concatenate(all_embeddings, axis=0)
       if input_was_string:
           return all_embeddings[0]  # 如果输入是单个字符串，返回第一个嵌入向量
       return all_embeddings

   def pooling(self,
               last_hidden_state: torch.Tensor,
               attention_mask: torch.Tensor = None) -> torch.Tensor:
       '''
       根据指定的池化方法对最后一层的隐藏状态进行池化
       '''
       if self.pooling_method == 'cls':
           return last_hidden_state[:, 0]  # 返回 [CLS] 标记的嵌入向量
       elif self.pooling_method == 'mean':
           s = torch.sum(
               last_hidden_state * attention_mask.unsqueeze(-1).float(),  # 计算加权和
               dim=1)
           d = attention_mask.sum(dim=1, keepdim=True).float()  # 计算权重
           return s / d  # 计算加权平均值


class BaaiEmbedding():

    def __init__(self, model_path, max_length=512, batch_size=256):
        self.model = FlagModel(model_path,
                               query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode_corpus(texts, self.batch_size,
                                        self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode_queries(text, self.batch_size,
                                         self.max_length).tolist()


class ChromaDB():

    def __init__(self, path):
        self.path = path
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("search")

    def load(self, file_path):
        self.path = file_path
        self.client = chromadb.PersistentClient(path=file_path)
        self.collection = self.client.get_collection("search")

    def search(self, vector, n_results):
        results = self.collection.query(query_embeddings=[vector],
                                        n_results=n_results)
        return results['documents'][0]

    def from_texts(self, vectors, documents):
        ids = []
        for i, doc in enumerate(documents):
            ids.append(str(i) + "_" + doc)
        self.collection.add(embeddings=vectors, documents=documents, ids=ids)

    def add_texts(self, vectors, documents, ids):
        self.collection.upsert(embeddings=vectors, documents=documents, ids=ids)


class TextLoader():

    def __init__(self, file_path, encoding):
        self.file_path = file_path
        self.encoding = encoding

    def load(self):
        """Load from file path."""
        text = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e
        return [text]


def extract_file_dirs(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):

        for file in files:
            fp = os.path.join(root, file)
            file_paths.append(fp)
    return file_paths


def split_chunks(text, chunk_size):
    docs = []
    doc_size = 0
    tmp = []
    for line in text.split():
        line = line.strip()
        if len(line) + doc_size < chunk_size:
            tmp.append(line)
            doc_size += len(line)
        else:
            docs.append("\n".join(tmp))
            tmp = []
            doc_size = 0
    return docs


def generate_prompt(query, docs):
    PROMPT_TEMPLATE = """
    已知信息：
    {context} 

    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
    问题是：{question}"""
    context = []
    for index, doc in enumerate(docs):
        doc = doc.strip()
        f_prompt = "<{a}>: {b}".format(a=index + 1, b=doc)
        context.append(f_prompt)
    context = "\n".join(context)
    prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}",
                                                                  context)
    return prompt

def get_openai_client():
    return openai.OpenAI(
        api_key="sk-6V2exWFBSa2lmuZ7C0D773D1BaEd4fB7A1B6A0A265D550C6",
        base_url="https://key.wenwen-ai.com/v1"
    )

def run():
    BGE_MODEL_PATH = "D:\\codes\\bge-large-zh"
    # BGE_MODEL_PATH = "BAAI/bge-large-zh-v1.5"
    FILE_PATH = "D:\\codes\\zsxq"
    embedding_model = BaaiEmbedding(BGE_MODEL_PATH)
    files = extract_file_dirs(FILE_PATH)
    loaders = [TextLoader(f, encoding='utf-8') for f in files]

    docs = []
    for l in loaders:
        docs.extend(l.load())
    chunks = []
    for doc in docs:
        chunk = split_chunks(doc, 200)
        chunks.extend(chunk)

    path = "./zsxq"
    vectordb = ChromaDB(path)
    load_data = False
    if load_data:
        emb = embedding_model.embed_documents(chunks)
        vectordb.from_texts(emb, chunks)

    query = "什么是知识星球?"

    result = vectordb.search(embedding_model.embed_query(query), 4)
    print(len(result))
    for r in result:
        print(r)
        print("******")

    prompt = generate_prompt(query, result)
    print(prompt)

    generate_for_llm = True
    # 调用 OpenAI 的 API 生成回答
    if generate_for_llm:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}])

        print(response.choices[0].message.content)

if __name__ == "__main__":
    print("hello world")
    run()

