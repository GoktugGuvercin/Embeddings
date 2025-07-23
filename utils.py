import cohere
from google import genai

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def group_by(df: pd.DataFrame, total_size: int):
    """
    Groups the dataset by classes

    * It groups the data entries in dataframe object by classes
    and create its distribution table.

    Args:
        df: A dataframe object where all data entries are stored.
        total_size: the size of all dataset
    """

    df_count = df.groupby('intent').size().reset_index(name="count")
    df_count['percentage'] = (df_count['count'] / total_size) * 100
    return df_count


def create_subset(
        df: pd.DataFrame,
        frac: float,
        class_names: list,
        drop_index: bool = False
):

    """
    Create a random subset of dataframe and filter by class names

    * A small subset of input dataframe is sampled randomly. The
    data entries in this subset will have some class names. Only
    the ones with given class names will be preserved, the remaining
    ones will be filtered.

    * Data entried in created subset keeps the original index values
    coming from original dataframe object. By reset_index(), we can
    create a new set of indices. If we want to drop old, original
    indices, we set "drop_index" to True.

    Args:
        df: A dataframe object where all data entries are stored.
        frac: the percentage of data to be sampled.
        class names: the names of classes for filtering
        drop_index: whether we drop old indices or not

    Returns:
        A list of float embeddings.
    """
    df_subset = df.sample(frac=frac, random_state=17)
    exist_ids = df_subset.intent.isin(class_names)
    df_subset = df_subset[exist_ids]
    df_subset.reset_index(drop=drop_index, inplace=True)
    return df_subset


def cohere_embeddings(
        co: cohere.ClientV2,
        texts: list[str],
        model: str = "embed-v4.0",
        input_type: str = "search_document",
):
    """
    Generates embeddings for a list of texts using the Cohere API.

    Args:
        co: Cohere Client object
        texts: A list of strings to embed.
        model: The name of Cohere embedding model to use.
        input_type: The type of input text

    Returns:
        A list of float embeddings.
    """

    output = co.embed(
        texts=texts,
        model=model,
        input_type=input_type,
        embedding_types=["float"]
    )
    return output.embeddings.float



def gemini_embeddings(
        go: genai.Client,
        texts: list[str],
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        embed_dim: int = 1536,
):
    """
    Generates embeddings for a list of texts using the Gemini API.

    Args:
        go: Gemini Client object
        texts: A list of strings to embed.
        model: The name of Gemini embedding model to use.
        task_type: For which type of task, the embedding will be used [1]
        embed_dim: The dimension of embeddings, it can be 768, 1536 or 3072 [1]

    [1]: https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types

    Returns:
        A list of float embeddings.
    """

    if embed_dim not in [768, 1536, 3072]:
        raise ValueError("Embedding dimension can be 768, 1536, or 3072")

    config = genai.types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=embed_dim
    )

    output = go.models.embed_content(
        model=model,
        contents=texts,
        config=config,
    )

    output_list = [content.values for content in output.embeddings]
    return output_list


def compress_by_pca(embeds: np.ndarray, size: int):
    """
    Compress and downscale the embeddings

    Args:
        embeds: text or vision embeddings
        size: the size to which embeddings will be compressed

    Returns:
        a numpy array of compressed embeds
    """
    pca = PCA(n_components=size)
    embeds_transform = pca.fit_transform(embeds)
    return embeds_transform


def create_embed_heatmap(
    df_subset: pd.DataFrame, 
    pca_dim: int, 
    embed_type: str, 
    sample_range: tuple
):
    
    """ Downscale embeddings and generate their heatmaps

    * Airline Travel Information System Records are organized as a dataset
    and it is passed to this function as input dataframe. These records
    are combined with cohere and google embeddings, which are stored in
    "co_embeds" and "go_embeds" columns.

    * One of these embeddings are chosen to be projected into lower dimension
    by PCA, and the entries in a specific range are chosen for the heatmap 
    visualization. 

    Args:
        df_subset: A dataframe object composed of customer feedbacks, 
                   intent classes, cohere embeds, and gemini embeds
        pca_dim: PCA dimension to which the embeddings of customer entries
                 will be projected
        embed_type: Cohere or gemini embeddings will be used for projection
                    and heatmap visualization ("co_embeds" or "go_embeds")
        sample_range: the samples in the given range will be used for 
                      heatmap visualization
    
    Returns: PCA embeds in numpy array
    """

    start = sample_range[0]
    end = sample_range[1]
    
    embeds = np.array(df_subset[embed_type].tolist())
    pca_embeds = compress_by_pca(embeds, pca_dim)

    selected_class_names = df_subset.iloc[start:end]["intent"].tolist()
    selected_queries = df_subset[start:end]["query"].tolist()
    selected_embeds = pca_embeds[start:end]

    queries_classes = [f"{q} ({c})" for q, c in
                    zip(selected_queries, selected_class_names)]
    num_queries = len(selected_queries)

    # creating heatmap
    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(selected_embeds, cmap="Reds", aspect="auto")

    # configuring x axis
    ax.set_xticks(np.arange(pca_dim))
    ax.set_xticklabels([f"{i}" for i in range(pca_dim)])
    ax.set_xlabel("Embedding")

    # configuring y axis
    ax.set_yticks(np.arange(num_queries))
    ax.set_yticklabels(queries_classes)
    ax.set_ylabel("Sentence")

    # configuring the plot
    embed_name = "Cohere" if embed_type == "co_embeds" else "Gemini"
    title = embed_name + " Embeddings and Queries"
  
    ax.set_title(title, pad=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value")

    plt.tight_layout()
    plt.show()

    return selected_embeds


class EmbedSimilarity:
    # A class to measure similarity between embeddings

    def __init__(self, embeds: np.ndarray):
        """
        Computes embedding similarity and sort their indices

        * Cosine similarity is calculated between embeddings. This results in
        a similarity matrix where each entry represents the similarity between
        an embedding pair.

        * Sample indices of similarity scores are sorted in descending order. 
        This gives rise to a matrix where the most and least similar samples to
        sample i are located in row i from left to right.

        Args:
            - embeds: a 2D numpy array of accumulated 1D embedding vectors
        """

        self.embeds = embeds
        self.sim_matrix = cosine_similarity(embeds)
        self.sorted_indices = np.argsort(self.sim_matrix, axis=1)[:, ::-1]

    def get_proximity(
            self,
            target_index: int,
            query_indices: list[int],
            search_frame: int
    ):

        """
        Checks if which embeddings are similar to target embedding

        * We aim to find which embeddings are similar to target embedding. To
        understand this, we check the most similar N entries to target embedding.
        How many entries will be checked given as an input parameter called 
        search frame.

        * Which embeddings can be located in these first N entries ? We need to
        provide the indices of our guesses. This function will interrogate these 
        indices are among the most similar N samples of target embedding.

        * Target embedding and query embeddings are specified by their indices,
        not themselves. 

        Args:
        -----
            target_index: the index of target embedding
            query_indices: the indices of the embeddings that we check in similarity
            search_frame: The size of search space

        """

        top_n_indices = self.sorted_indices[target_index, 1 : (search_frame + 1)]

        hit_queries = []
        for index in query_indices:
            if index in top_n_indices:
                hit_queries.append(index)
                print(f"CORRECT, {index} is highly similar to {target_index}")
            else:
                print(f"WRONG, {index} is distant from {target_index}")

        return hit_queries, top_n_indices
