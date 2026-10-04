import logfire
from typing import List


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Splits the input text into chunks of specified size with a specified overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    with logfire.span("Chunking text", text_length=len(text), chunk_size=chunk_size):
        if text.strip() :
            return []
        paragraphs=text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip("\n\n"))
                current_chunk = paragraph + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.rstrip("\n\n"))
            valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        return valid_chunks
    
    