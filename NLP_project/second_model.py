import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')


# Optimized imports
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


class AdvancedHealthcareChatbot:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.healthcare_intents = self._load_healthcare_intents()
        self.symptom_checker = SymptomChecker()
        self.lifestyle_advisor = LifestyleAdvisor()
        self.medical_ner = MedicalNER()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.emergency_detector = EmergencyDetector()
        self.conversation_memory = {}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
        self._train_intent_classifier()

    def _load_healthcare_intents(self) -> Dict:
        return {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "responses": [
                    "Hello! I'm your healthcare assistant. How can I help you today?",
                    "Hi there! I'm here to assist with your health questions."
                ]
            },
            "symptoms": {
                "patterns": ["symptom", "pain", "hurt", "fever", "headache", "nausea"],
                "responses": [],
                "action": "analyze_symptoms"
            },
            "diet_advice": {
                "patterns": ["diet", "food", "eat", "nutrition", "meal"],
                "responses": [],
                "action": "provide_diet_advice"
            },
            "lifestyle": {
                "patterns": ["exercise", "workout", "sleep", "stress", "lifestyle"],
                "responses": [],
                "action": "provide_lifestyle_tips"
            },
            "medication": {
                "patterns": ["medicine", "pill", "prescription", "drug", "medication"],
                "responses": [],
                "action": "handle_medication_query"
            },
            "emergency": {
                "patterns": ["emergency", "help", "sos", "urgent", "critical"],
                "responses": ["This sounds serious. Please seek immediate medical attention!"],
                "priority": "high"
            }
        }

    def _train_intent_classifier(self):
        patterns = []
        intents = []
        for intent_name, intent_data in self.healthcare_intents.items():
            for pattern in intent_data["patterns"]:
                patterns.append(pattern)
                intents.append(intent_name)
        additional_patterns = [
            "I have a headache", "My stomach hurts", "What should I eat?",
            "Exercise routine", "Sleep problems", "Medicine reminder"
        ]
        additional_intents = ["symptoms", "symptoms", "diet_advice", "lifestyle", "lifestyle", "medication"]
        patterns.extend(additional_patterns)
        intents.extend(additional_intents)
        self.vectorizer.fit(patterns)
        self.intent_labels = list(set(intents))

    def process_message(self, message: str, user_id: str) -> Dict[str, Any]:
        cleaned_message = self._preprocess_text(message)
        intent = self._recognize_intent(cleaned_message)
        entities = self.medical_ner.extract_entities(message)
        sentiment = self._analyze_sentiment(message)
        emotion = self._detect_emotion(message)
        is_emergency = self.emergency_detector.detect_emergency(message)
        response = self._generate_response(intent, message, entities, user_id)
        self._update_conversation_memory(user_id, message, intent, entities)
        return {
            "response": response,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "emotion": emotion,
            "is_emergency": is_emergency,
            "timestamp": datetime.now().isoformat()
        }

    def _preprocess_text(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        doc = self.nlp(" ".join(tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        stemmed_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        return " ".join(stemmed_tokens)

    def _recognize_intent(self, text: str) -> str:
        tfidf_vector = self.vectorizer.transform([text])
        best_similarity = 0
        best_intent = "unknown"
        for intent_name, intent_data in self.healthcare_intents.items():
            for pattern in intent_data["patterns"]:
                pattern_vector = self.vectorizer.transform([pattern])
                similarity = cosine_similarity(tfidf_vector, pattern_vector)[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_intent = intent_name
        if best_similarity < 0.3:
            best_intent = self._rule_based_intent_detection(text)
        return best_intent

    def _rule_based_intent_detection(self, text: str) -> str:
        text_lower = text.lower()
        emergency_keywords = ['emergency', 'help', 'sos', 'urgent', 'dying', 'heart attack']
        if any(keyword in text_lower for keyword in emergency_keywords):
            return "emergency"
        symptom_patterns = [r'hurt', r'pain', r'fever', r'headache', r'nausea', r'vomit']
        if any(re.search(pattern, text_lower) for pattern in symptom_patterns):
            return "symptoms"
        diet_patterns = [r'eat', r'food', r'diet', r'nutrition', r'meal']
        lifestyle_patterns = [r'exercise', r'sleep', r'stress', r'workout']
        if any(re.search(pattern, text_lower) for pattern in diet_patterns):
            return "diet_advice"
        elif any(re.search(pattern, text_lower) for pattern in lifestyle_patterns):
            return "lifestyle"
        return "general_query"

    def _generate_response(self, intent: str, message: str, entities: List, user_id: str) -> str:
        if intent == "emergency":
            return "🚨 This sounds like an emergency! Please call emergency services immediately or go to the nearest hospital."
        elif intent == "symptoms":
            return self.symptom_checker.analyze_symptoms(message, entities)
        elif intent == "diet_advice":
            return self.lifestyle_advisor.provide_diet_advice(entities, user_id)
        elif intent == "lifestyle":
            return self.lifestyle_advisor.provide_lifestyle_tips(entities, user_id)
        elif intent == "medication":
            return "I can help with medication information. Please consult your doctor for specific advice."
        else:
            return "I'm here to help with health-related questions. You can ask about symptoms, diet, exercise, or general health advice."

    def _analyze_sentiment(self, text: str) -> Dict:
        try:
            result = self.sentiment_analyzer(text)[0]
            return {"label": result['label'], "score": float(result['score'])}
        except Exception:
            return {"label": "NEUTRAL", "score": 0.5}

    def _detect_emotion(self, text: str) -> str:
        emotion_keywords = {
            "happy": ["good", "great", "excellent", "happy", "well"],
            "sad": ["sad", "bad", "terrible", "awful", "hurt"],
            "anxious": ["worried", "anxious", "nervous", "scared", "afraid"],
            "pain": ["pain", "hurt", "sore", "aching", "uncomfortable"]
        }
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        return "neutral"

    def _update_conversation_memory(self, user_id: str, message: str, intent: str, entities: List):
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        self.conversation_memory[user_id].append({
            "message": message,
            "intent": intent,
            "entities": entities,
            "timestamp": datetime.now()
        })
        if len(self.conversation_memory[user_id]) > 10:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-10:]


class MedicalNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.medical_terms = self._load_medical_terms()

    def extract_entities(self, text: str) -> List[Dict]:
        entities = []
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        entities.extend(self._extract_medical_terms(text))
        entities.extend(self._extract_symptoms(text))
        return entities

    def _load_medical_terms(self) -> List[str]:
        return [
            "headache", "fever", "nausea", "cough", "pain", "fatigue",
            "hypertension", "diabetes", "asthma", "arthritis", "migraine"
        ]

    def _extract_medical_terms(self, text: str) -> List[Dict]:
        entities = []
        text_lower = text.lower()
        for term in self.medical_terms:
            if term in text_lower:
                start = text_lower.find(term)
                entities.append({
                    "text": term,
                    "label": "MEDICAL_TERM",
                    "start": start,
                    "end": start + len(term)
                })
        return entities

    def _extract_symptoms(self, text: str) -> List[Dict]:
        symptoms = []
        symptom_patterns = {
            r'\b(headache|migraine)\b': 'HEADACHE',
            r'\b(fever|temperature)\b': 'FEVER',
            r'\b(nausea|vomit)\b': 'NAUSEA',
            r'\b(cough|coughing)\b': 'COUGH',
            r'\b(pain|hurt|sore)\b': 'PAIN'
        }
        for pattern, label in symptom_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                symptoms.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })
        return symptoms


class SymptomChecker:
    def __init__(self):
        self.symptom_database = self._load_symptom_database()
        self.condition_mapping = self._load_condition_mapping()

    def analyze_symptoms(self, text: str, entities: List) -> str:
        symptoms = [entity for entity in entities if entity['label'] in ['SYMPTOM', 'MEDICAL_TERM', 'PAIN']]
        if not symptoms:
            return "Could you please describe your symptoms in more detail?"
        symptom_texts = [symptom['text'] for symptom in symptoms]
        conditions = self._match_conditions(symptom_texts)
        if conditions:
            advice = self._generate_symptom_advice(conditions, symptom_texts)
            return f"Based on your symptoms ({', '.join(symptom_texts)}), {advice}"
        else:
            return f"I understand you're experiencing {', '.join(symptom_texts)}. It's important to consult a healthcare professional for proper diagnosis."

    def _load_symptom_database(self) -> Dict:
        return {
            "headache": ["migraine", "tension headache", "sinusitis"],
            "fever": ["flu", "infection", "covid-19"],
            "cough": ["cold", "flu", "asthma", "covid-19"],
            "nausea": ["food poisoning", "migraine", "pregnancy"]
        }

    def _load_condition_mapping(self) -> Dict:
        return {
            "migraine": "try resting in a dark room and consider over-the-counter pain relief",
            "flu": "get plenty of rest, stay hydrated, and consider antiviral medication",
            "cold": "rest, fluids, and over-the-counter cold remedies may help",
            "covid-19": "please get tested and self-isolate until you receive results"
        }

    def _match_conditions(self, symptoms: List[str]) -> List[str]:
        conditions = []
        for symptom in symptoms:
            if symptom in self.symptom_database:
                conditions.extend(self.symptom_database[symptom])
        return list(set(conditions))

    def _generate_symptom_advice(self, conditions: List[str], symptoms: List[str]) -> str:
        if not conditions:
            return "please consult a healthcare professional."
        advice_parts = []
        for condition in conditions[:2]:
            if condition in self.condition_mapping:
                advice_parts.append(self.condition_mapping[condition])
        if advice_parts:
            return "it might be " + " or ".join(conditions) + ". " + " ".join(advice_parts)
        else:
            return "it could be " + " or ".join(conditions) + ". Please consult a doctor."


class LifestyleAdvisor:
    def __init__(self):
        self.diet_recommendations = self._load_diet_recommendations()
        self.lifestyle_tips = self._load_lifestyle_tips()

    def provide_diet_advice(self, entities: List, user_id: str) -> str:
        health_conditions = [entity['text'] for entity in entities if entity['label'] == 'MEDICAL_TERM']
        if health_conditions:
            condition = health_conditions[0]
            if condition in self.diet_recommendations:
                return f"For {condition}, {self.diet_recommendations[condition]}"
        return "A balanced diet with plenty of fruits, vegetables, and whole grains is generally recommended. Consult a nutritionist for personalized advice."

    def provide_lifestyle_tips(self, entities: List, user_id: str) -> str:
        topics = [entity['text'] for entity in entities if entity['label'] in ['MEDICAL_TERM', 'SYMPTOM']]
        if topics:
            topic = topics[0]
            if topic in self.lifestyle_tips:
                return self.lifestyle_tips[topic]
        return "Regular exercise, adequate sleep, and stress management are key to good health. Aim for 30 minutes of activity daily and 7-8 hours of sleep."

    def _load_diet_recommendations(self) -> Dict:
        return {
            "diabetes": "focus on low-glycemic foods like whole grains, vegetables, and lean proteins",
            "hypertension": "reduce sodium intake and increase potassium-rich foods like bananas and leafy greens",
            "obesity": "portion control and balanced meals with emphasis on vegetables and protein"
        }

    def _load_lifestyle_tips(self) -> Dict:
        return {
            "stress": "Practice mindfulness meditation and deep breathing exercises daily",
            "insomnia": "Maintain a consistent sleep schedule and avoid screens before bedtime",
            "fatigue": "Ensure adequate hydration and consider gentle exercise like walking"
        }


class EmergencyDetector:
    def __init__(self):
        self.emergency_patterns = [
            r'\b(heart attack|chest pain|stroke|difficulty breathing)\b',
            r'\b(severe pain|unconscious|bleeding heavily)\b',
            r'\b(suicide|self harm|kill myself)\b',
            r'\b(emergency|911|ambulance)\b'
        ]
        self.emergency_keywords = ['help', 'emergency', 'urgent', 'dying', 'now']

    def detect_emergency(self, text: str) -> bool:
        text_lower = text.lower()
        for pattern in self.emergency_patterns:
            if re.search(pattern, text_lower):
                return True
        emergency_words = [word for word in self.emergency_keywords if word in text_lower]
        if len(emergency_words) >= 2:
            return True
        return False


class NLPAnalyzer:
    @staticmethod
    def advanced_tokenization(text: str) -> Dict:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "tokens": words
        }

    @staticmethod
    def pos_tagging_analysis(text: str) -> List[Tuple]:
        tokens = word_tokenize(text)
        return pos_tag(tokens)

    @staticmethod
    def named_entity_recognition(text: str) -> List[Tuple]:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
        return entities

    @staticmethod
    def ngram_analysis(text: str, n: int = 2) -> List[Tuple]:
        tokens = word_tokenize(text)
        return list(ngrams(tokens, n))

    @staticmethod
    def chunking_analysis(text: str) -> List[Dict]:
        grammar = r"""
            NP: {<DT>?<JJ>*<NN.*>+}
            VP: {<VB.*><NP|PP>*}
        """
        chunk_parser = nltk.RegexpParser(grammar)
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tree = chunk_parser.parse(pos_tags)
        chunks = []
        for subtree in tree.subtrees():
            if subtree.label() in ['NP', 'VP']:
                chunks.append({
                    "type": subtree.label(),
                    "text": ' '.join(word for word, tag in subtree.leaves())
                })
        return chunks


class ChatbotService:
    def __init__(self):
        self.chatbot = AdvancedHealthcareChatbot()
        self.nlp_analyzer = NLPAnalyzer()

    def process_chat_message(self, message: str, user_id: str) -> Dict:
        if not message or len(message.strip()) == 0:
            return {"error": "Empty message"}
        result = self.chatbot.process_message(message, user_id)
        nlp_analysis = {
            "tokenization": self.nlp_analyzer.advanced_tokenization(message),
            "pos_tags": self.nlp_analyzer.pos_tagging_analysis(message),
            "ner": self.nlp_analyzer.named_entity_recognition(message),
            "ngrams": self.nlp_analyzer.ngram_analysis(message),
            "chunks": self.nlp_analyzer.chunking_analysis(message)
        }
        result["nlp_analysis"] = nlp_analysis
        return result


# if __name__ == "__main__":
#     chatbot_service = ChatbotService()


#     test_messages = input("Enter the message")

#     response = chatbot_service.process_chat_message(test_messages, user_id="test_user")
#     print(response)
    
