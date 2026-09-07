"""Local embedder using sentence-transformers."""
import os
import logging
import numpy as np
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from backend.interfaces.embedder import (
    EmbedderInterface,
    EMBEDDING_DIMENSION,
    EmbeddingError
)


  
