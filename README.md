
# Embeddings

Large language models (LLMs) exhibit two primary capabilities: text generation and text representation. In a generative capacity, LLMs complete a given text prompt. For text representation, they produce numerical embeddings that capture the semantic meaning of the text. The core distinction between these functions lies in the underlying architecture. Decoder-based models excel at auto-completing text, while encoder-based models are designed to generate these dense vector representations, known as embeddings.

Processing raw sentences, paragraphs and passages for the applications where we try to access and makes use of their context with semantic meaning is very difficult, some keyword matching algorithms may be only and limited option. In contrast, the embeddings turns their raw and unstructured format into quantitative and structured system, which enables them to be efficiently used in mathematical operations and algorithmic thinking. Right now, by embeddings we can calculate the distance or similarity between them; we can quantitatively measure how related different pieces of text are. This underlies semantic search, search engines and recommender systems. 

- Recommender Systems
- Search Engines
- Semantic Search
- Social Media Content Moderation
- Email Spam Filtering 

In supervised learning paradigms, sentiment analysis and text classification are the most common fields in which we use dense vector embeddings. Additionally, these embeddings help to group and cluster many different passages with respect to the topic or context as a typical example of unsupervised learning.

## Embedding Quality Across Tasks and Training Approaches

The training strategy that we used for embedding models actually determines for which tasks the generated embeddings are suitable. If we take qwen3 embeddings into account, it actually follows 3 stages training paradigm:

1. Weakly Supervised Contrastive Learning by InfoNCE Loss on 150M LLM Generated Query-Document Pairs
2. Supervised Finetuning on 12M High-Quality Filtered Query-Document Pairs + Keeping Contrastive Learning
3. Merging Multiple Checkpoints from Finetuning

Gemini embeddings adapt completely different approach and pay more attention to the finetuning:

1. Pre-training: Initializing the embedding model from Gemini model
2. Pre-finetuning by NCE Loss on Billions Web-Corpus Title-Passage Pairs
3. Finetuning on Mixture of Task-Specific Datasets built on Triplets
4. Model Soup Averaging

In Qwen3, hard-negative contrastive loss is preserved in both stage 1 and 2, which triggers sharper query–document geometry. Besides, instruction prompt added at the beginning of each query lets its encoder specialise to "search query” versus “search document” differentiation. Meanwhile, Gemini does not opt for hard negatives in pre-finetuning stage. That is why, we observe that Qwen3-8B achieves 70.88 retrieval score on the MTEB Multilingual benchmark, outperforming Gemini Embedding, which scores 67.71 on the same task.

## Cohere Embeddings

Cohere also started to leverage and prioritize task-specific optimization of its embedding models. Starting from ***Embed V3.0***, each embedding model accepts a new parameter called `input_type`. Generated embeddings can be used for multiple purposes, but this does not mean that it is optimal for every task. That is why, embedding models are optimized for the type of input and task in the finetuning stage. At this point, the training strategy that they follow tends to alter feature space. See the [documentation](https://cohere.com/blog/introducing-embed-v3) and [v2-API](https://docs.cohere.com/docs/embeddings#the-input_type-parameter)

- *"search_document":* Used for the documents that you want to store in your vector database
- *"search_query":* Used for the queries that you want to search so as to find the most relevant documents in your vector database
- *"classification":* Use this when you want to do text classification by these embeddings
- *"clustering":* Use this when you want to do text clustering by these embeddings


```python
import cohere

co = cohere.ClientV2()

response = co.embed(
    inputs=text_inputs,
    model="embed-v4.0",
    input_type="classification",
    embedding_types=["float"],
)
```


