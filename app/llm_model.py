from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from collections import Counter

# Use DialoGPT-small for conversational AI - better for Q&A than T5
MODEL_NAME = "microsoft/DialoGPT-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

def generate_answer(context, query, sources=None):
    """Generate a proper answer by analyzing the context and addressing the specific question"""
    
    # Handle the case where no information is available
    if not context or context.strip() == "I don't have information about that in my knowledge base.":
        return "I don't have information about that in my knowledge base."
    
    # Clean and prepare the context
    context_clean = clean_context(context)
    
    # If we have source information, try to use content from the same document
    if sources and len(sources) > 1:
        context_clean = filter_by_primary_source(context, sources)
    
    # Create a focused answer based on the question type
    answer = create_smart_answer(context_clean, query)
    
    return answer

def clean_context(context):
    """Clean and prepare the context for better processing"""
    # Remove extra whitespace and normalize
    context = re.sub(r'\s+', ' ', context.strip())
    # Remove common artifacts
    context = re.sub(r'\[.*?\]', '', context)  # Remove citations like [1], [2]
    return context

def filter_by_primary_source(context, sources):
    """Filter context to use only content from the most relevant source"""
    try:
        # Find the most common source (most relevant document)
        source_counts = Counter(sources)
        primary_source = source_counts.most_common(1)[0][0]
        
        # Split context back into chunks and filter by primary source
        chunks = context.split('\n')
        filtered_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i < len(sources) and sources[i] == primary_source:
                filtered_chunks.append(chunk)
        
        if filtered_chunks:
            return ' '.join(filtered_chunks)
    except:
        pass
    
    return context

def create_smart_answer(context, query):
    """Create a smart, coherent answer that directly addresses the question"""
    
    # Extract the main topic from the query
    topic = extract_main_topic(query)
    
    # Split context into complete sentences
    sentences = extract_complete_sentences(context)
    
    # Find the most relevant sentences for this topic
    relevant_sentences = find_topic_sentences(sentences, topic, query)
    
    if not relevant_sentences:
        return "I don't have specific information about that in my knowledge base."
    
    # Create a well-structured answer
    answer = construct_coherent_answer(relevant_sentences, query, topic)
    
    return answer

def extract_main_topic(query):
    """Extract the main topic from the query"""
    query_lower = query.lower().strip()
    
    # Remove question words
    query_clean = re.sub(r'\b(what|is|are|how|do|does|can|will|would|should)\b', '', query_lower)
    query_clean = query_clean.strip()
    
    # Extract key terms
    words = query_clean.split()
    if words:
        # Return the main concept (usually the first meaningful word)
        return words[0] if words[0] not in ['the', 'a', 'an'] else (words[1] if len(words) > 1 else words[0])
    
    return query_lower

def extract_complete_sentences(text):
    """Extract complete, meaningful sentences from text"""
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    
    complete_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Only keep sentences that are complete and meaningful
        if (len(sentence) > 20 and 
            not sentence.isupper() and 
            ' ' in sentence and
            not sentence.endswith(('and', 'or', 'but', 'the', 'a', 'an'))):
            complete_sentences.append(sentence)
    
    return complete_sentences

def find_topic_sentences(sentences, topic, query):
    """Find sentences that are most relevant to the topic"""
    query_words = set(query.lower().split())
    topic_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        # Calculate relevance score
        relevance_score = 0
        
        # Check if main topic is mentioned
        if topic in sentence_lower:
            relevance_score += 3
        
        # Check for query word matches
        word_matches = len(query_words.intersection(sentence_words))
        relevance_score += word_matches
        
        # Bonus for definition-style sentences
        if any(phrase in sentence_lower for phrase in [' is ', ' are ', ' refers to', ' defined as']):
            relevance_score += 2
        
        if relevance_score > 0:
            topic_sentences.append((sentence, relevance_score))
    
    # Sort by relevance and return top sentences
    topic_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in topic_sentences[:3]]

def construct_coherent_answer(sentences, query, topic):
    """Construct a coherent answer from relevant sentences"""
    if not sentences:
        return "I don't have specific information about that in my knowledge base."
    
    # For "what is" questions, prioritize definition sentences
    if query.lower().startswith('what is'):
        # Find the best definition sentence
        definition_sentence = None
        for sentence in sentences:
            if topic in sentence.lower() and any(word in sentence.lower() for word in [' is ', ' are ']):
                definition_sentence = sentence
                break
        
        if definition_sentence:
            # Add additional relevant information
            additional_info = []
            for sentence in sentences:
                if (sentence != definition_sentence and 
                    len(additional_info) < 2 and
                    not is_too_similar(sentence, definition_sentence)):
                    additional_info.append(sentence)
            
            result = definition_sentence
            if additional_info:
                result += ". " + ". ".join(additional_info) + "."
            return result
    
    # For other questions, combine the most relevant sentences
    unique_sentences = []
    for sentence in sentences:
        if not any(is_too_similar(sentence, existing) for existing in unique_sentences):
            unique_sentences.append(sentence)
        if len(unique_sentences) >= 3:
            break
    
    return ". ".join(unique_sentences) + "."

def is_too_similar(sentence1, sentence2):
    """Check if two sentences are too similar"""
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return False
    
    overlap = len(words1.intersection(words2))
    similarity = overlap / min(len(words1), len(words2))
    
    return similarity > 0.7  # 70% similarity threshold





# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# MODEL_NAME = "microsoft/phi-2"  # or "distilgpt2"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# def generate_answer(context, query):
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)