from typing import List

def process(input_data: str) -> List[str]:
    minimum_chunk_length = 2000
    n = len(input_data)
    paragraphs = []
    index = 0
    while index<n:
        new_index = input_data.find("\n", index+minimum_chunk_length)
        if new_index==-1:
            paragraphs.append(input_data[index:])
            break
        paragraphs.append(input_data[index:new_index+1]) # exclusive
        index = new_index + 1
    return paragraphs

