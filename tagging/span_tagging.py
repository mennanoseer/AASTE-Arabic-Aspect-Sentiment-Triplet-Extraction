
def create_tag_table(data, version='3D'):
    """
    Creates a 2D table of string tags representing the relationships between tokens.

    Args:
        data (dict): A dictionary containing the token list and triplet annotations.
        version (str, optional): The tagging schema version ('1D', '2D', or '3D'). Defaults to '3D'.

    Returns:
        list of lists: A 2D list representing the tagged table.
    """
    tokens = data['token']
    triplets = data['triplets']
    tag_table = [['' for _ in range(len(tokens))] for _ in range(len(tokens))]

    # Extract unique aspect and opinion spans from the triplets
    aspect_spans = set(tuple(triplet[0]) for triplet in triplets)
    opinion_spans = set(tuple(triplet[1]) for triplet in triplets)
    
    # Map sentiment to the span covering both the aspect and opinion
    sentiment_map = {(min(t[0][0], t[1][0]), max(t[0][1], t[1][1])): t[2] for t in triplets}
    
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            span = (i, j)
            
            if version == '3D':
                # Format: Aspect-Opinion-Sentiment (e.g., "A-O-POS")
                aspect_tag = 'A' if span in aspect_spans else 'N'
                opinion_tag = 'O' if span in opinion_spans else 'N'
                sentiment_tag = sentiment_map.get(span, 'N')
                tag_table[i][j] = f"{aspect_tag}-{opinion_tag}-{sentiment_tag}"
            
            elif version == '2D':
                # Format: Entity-Sentiment (e.g., "A-POS", "O-NEG")
                if span in aspect_spans:
                    entity_tag = 'A'
                elif span in opinion_spans:
                    entity_tag = 'O'
                else:
                    entity_tag = 'N'
                sentiment_tag = sentiment_map.get(span, 'N')
                tag_table[i][j] = f"{entity_tag}-{sentiment_tag}"

            elif version == '1D':
                # Single label for each role
                if span in aspect_spans:
                    tag_table[i][j] = 'A'
                elif span in opinion_spans:
                    tag_table[i][j] = 'O'
                else:
                    tag_table[i][j] = sentiment_map.get(span, 'N')
                    
    return tag_table

def create_label_maps(version='3D'):
    """
    Creates mappings between label strings and integer IDs based on the schema version.

    Args:
        version (str, optional): The tagging schema version. Defaults to '3D'.

    Returns:
        tuple: A tuple containing (label_to_id_map, id_to_label_map).
    """
    labels = []
    sentiments = ['N', 'NEG', 'NEU', 'POS']
    
    if version == '3D':
        for aspect_tag in ['N', 'A']:
            for opinion_tag in ['N', 'O']:
                for sentiment_tag in sentiments:
                    labels.append(f"{aspect_tag}-{opinion_tag}-{sentiment_tag}")
    elif version == '2D':
        for entity_tag in ['N', 'O', 'A']:
            for sentiment_tag in sentiments:
                labels.append(f"{entity_tag}-{sentiment_tag}")
    elif version == '1D':
        labels = ['N', 'NEG', 'NEU', 'POS', 'O', 'A']

    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    return label_to_id, id_to_label

def create_sentiment_maps():
    """
    Creates mappings between sentiment labels and integer IDs.

    Returns:
        tuple: A tuple containing (sentiment_to_id_map, id_to_sentiment_map).
    """
    sentiments = ['N', 'NEG', 'NEU', 'POS']
    sentiment_to_id = {label: i for i, label in enumerate(sentiments)}
    id_to_sentiment = {i: label for i, label in enumerate(sentiments)}
    return sentiment_to_id, id_to_sentiment

def convert_tags_to_ids(tag_table, label_to_id):
    """Converts a table of string tags to a table of integer IDs."""
    return [[label_to_id.get(tag, 0) for tag in row] for row in tag_table]

def convert_ids_to_tags(id_table, id_to_label):
    """Converts a table of integer IDs back to a table of string tags."""
    return [[id_to_label[tag_id] for tag_id in row] for row in id_table]
