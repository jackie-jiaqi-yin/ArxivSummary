import re
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import spacy
from spacy.tokens import Token
from spacy.symbols import NOUN


class TopicPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Add domain-specific stopwords
        self.stopwords = self.nlp.Defaults.stop_words
        domain_stopwords = {
            'using', 'based', 'via', 'towards', 'method', 'approach',
            'methods', 'approaches', 'llm', 'llms', 'nlp'
        }
        self.stopwords.update(domain_stopwords)

        # Special terms to preserve
        self.special_terms = {
            'chain-of-thought', 'rag', 'data filtering', 'tuning',
            'pre-training', 'fine-tuning', 'few-shot', 'zero-shot',
            'in-context', 'cross-attention', 'self-attention',
            'multimodal learning', 'fine-tuning', 'pretraining'
        }

        # Add special case rules to the tokenizer
        for term in self.special_terms:
            # Add the term as a single token
            case = [{spacy.attrs.ORTH: term}]
            self.nlp.tokenizer.add_special_case(term, case)
            # Add lowercase version if different
            if term.lower() != term:
                case_lower = [{spacy.attrs.ORTH: term.lower()}]
                self.nlp.tokenizer.add_special_case(term.lower(), case_lower)

    def is_relevant_token(self, token) -> bool:
        """Check if token is relevant"""
        # Consider special terms as relevant
        if token.text.lower() in self.special_terms:
            return True
        return (
                not token.is_stop and
                not token.is_punct and
                token.lemma_.lower() not in self.stopwords and
                token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN'}
        )

    def clean_text(self, text: str) -> str:
        """Clean text while preserving important characters but removing parenthetical content"""
        # Remove text within parentheses including the parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove special characters but keep hyphens
        text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
        # Standardize spacing but preserve hyphenated terms
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy"""
        doc = self.nlp(text)
        lemmatized = []
        for token in doc:
            # Preserve special terms exactly as they are
            if token.text.lower() in self.special_terms:
                lemmatized.append(token.text.lower())
            elif self.is_relevant_token(token):
                lemmatized.append(token.lemma_)
        return ' '.join(lemmatized)

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text using noun chunks"""
        doc = self.nlp(text)
        key_phrases = []
        for chunk in doc.noun_chunks:
            # Get relevant tokens from the chunk
            relevant_tokens = []
            i = 0
            while i < len(chunk):
                token = chunk[i]
                # Check if this token starts a special term
                found_special_term = False
                for term in self.special_terms:
                    if chunk[i:].text.lower().startswith(term.lower()):
                        relevant_tokens.append(term)
                        i += len(self.nlp(term))
                        found_special_term = True
                        break
                if not found_special_term:
                    if token.text.lower() not in self.stopwords:
                        relevant_tokens.append(token.text)
                    i += 1

            if relevant_tokens:
                # Join tokens, maintaining hyphens and parentheses
                phrase = ''.join([' ' + token if not (token.startswith('-') or token.startswith('('))
                                  else token for token in relevant_tokens]).strip()
                if phrase:
                    key_phrases.append(phrase)
        return key_phrases

    def lemmatize_phrases(self, phrases: List[str]) -> List[str]:
        """Lemmatize a list of phrases"""
        lemmatized_phrases = []
        for phrase in phrases:
            lemmatized = self.lemmatize_text(phrase)
            if lemmatized:
                # Clean up any extra spaces around hyphens and parentheses
                lemmatized = re.sub(r'\s*-\s*', '-', lemmatized)
                lemmatized = re.sub(r'\s*\(\s*', '(', lemmatized)
                lemmatized = re.sub(r'\s*\)\s*', ')', lemmatized)
                lemmatized_phrases.append(lemmatized)
        return lemmatized_phrases

    def process_text(self, date_topics: Dict[str, List[str]]) -> pd.DataFrame:
        """Process text data and create DataFrame"""
        processed_data = []
        for date, topics in date_topics.items():
            for topic in topics:
                cleaned_text = self.clean_text(topic)
                key_phrases = self.extract_key_phrases(cleaned_text)
                lemmatized_phrases = self.lemmatize_phrases(key_phrases)

                processed_data.append({
                    'date': date,
                    'original_topic': topic,
                    'cleaned_topic': cleaned_text,
                    'original_phrases': key_phrases,
                    'lemmatized_phrases': lemmatized_phrases
                })

        return pd.DataFrame(processed_data)

