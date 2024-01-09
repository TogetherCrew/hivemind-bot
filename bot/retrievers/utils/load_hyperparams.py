import os

from dotenv import load_dotenv


def load_hyperparams() -> tuple[int, int, int]:
    """
    load the k1, k2, and d hyperparams that are used for retrievers

    Returns
    ---------
    k1 : int
        the value for the first summary search
        to get the `k1` count similar nodes
    k2 : int
        the value for the secondary raw search
        to get the `k2` count simliar nodes
    d : int
        the before and after day interval
    """
    load_dotenv()

    k1 = os.getenv("K1_RETRIEVER_SEARCH")
    k2 = os.getenv("K2_RETRIEVER_SEARCH")
    d = os.getenv("D_RETRIEVER_SEARCH")

    if k1 is None:
        raise ValueError("No `K1_RETRIEVER_SEARCH` available in .env file!")
    if k2 is None:
        raise ValueError("No `K2_RETRIEVER_SEARCH` available in .env file!")
    if d is None:
        raise ValueError("No `D_RETRIEVER_SEARCH` available in .env file!")

    return int(k1), int(k2), int(d)
