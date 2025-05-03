import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import re
import string
import os
import random
from datetime import datetime, timedelta

# Download necessary NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def generate_dates(start_date="2020-01-01", end_date="2021-03-31", n=4511):
    """Generate random dates between start and end date"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate date range in days
    date_range = (end - start).days
    
    # Generate random days to add
    random_days = np.random.randint(0, date_range, size=n)
    
    # Generate dates
    dates = [start + timedelta(days=int(day)) for day in random_days]
    
    # Convert to string format
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]
    
    return date_strings

def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms"""
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Don't replace more words than exist in the text
    n = min(n, len(words))
    
    # Choose random indices to replace
    indices = random.sample(range(len(words)), n)
    
    for idx in indices:
        word = words[idx]
        # Skip short words, stopwords or special terms
        if len(word) <= 3:
            continue
            
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        if synonyms:
            synonym = random.choice(synonyms)
            words[idx] = synonym.replace('_', ' ')
    
    return ' '.join(words)

def word_swap(text, swap_rate=0.15):
    """Swap words between classes to create more challenging examples"""
    hate_terms = ["China", "virus", "Chinese", "Wuhan", "Asian", "Asians"]
    neutral_terms = ["COVID-19", "coronavirus", "pandemic", "social distancing", "masks"]
    
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < swap_rate:
            # Sometimes swap in words from other categories
            if any(term.lower() in word.lower() for term in hate_terms):
                words[i] = random.choice(neutral_terms)
            elif any(term.lower() in word.lower() for term in neutral_terms):
                if random.random() < 0.5:
                    words[i] = random.choice(hate_terms)
    
    return ' '.join(words)

def add_noise(text, noise_level=0.1):
    """Add random noise to text"""
    words = text.split()
    if random.random() < noise_level:
        # Add typos
        if len(words) > 0:
            idx = random.randint(0, len(words)-1)
            word = words[idx]
            if len(word) > 3:
                pos = random.randint(0, len(word)-1)
                chars = list(word)
                chars[pos] = random.choice(string.ascii_lowercase)
                words[idx] = ''.join(chars)
    
    # Sometimes remove a word
    if random.random() < noise_level and len(words) > 3:
        idx = random.randint(0, len(words)-1)
        words.pop(idx)
    
    # Sometimes add a common word
    if random.random() < noise_level:
        common_words = ["the", "a", "and", "is", "in", "to", "for", "of", "with", "on"]
        words.insert(random.randint(0, len(words)), random.choice(common_words))
    
    return ' '.join(words)

def create_variations(base_texts, n_variations=10, ambiguity_level=0.2):
    """Create variations of base texts with more ambiguity"""
    variations = []
    
    for text in base_texts:
        variations.append(text)  # Keep the original
        
        # Create n_variations of the text
        for _ in range(n_variations):
            # Replace 1-3 words with synonyms
            new_text = synonym_replacement(text, n=random.randint(1, 3))
            
            # Sometimes swap words from different categories
            if random.random() < ambiguity_level:
                new_text = word_swap(new_text)
            
            # Add noise to text
            new_text = add_noise(new_text)
            
            # Add random hashtags occasionally
            if random.random() < 0.3:
                hashtags = ['#covid19', '#coronavirus', '#china', '#wuhan', '#asians',
                           '#stayathome', '#pandemic', '#virus']
                new_text += ' ' + random.choice(hashtags)
            
            # Add random punctuation variations
            if random.random() < 0.5:
                puncts = ['!', '!!', '...', '?', '.']
                new_text += random.choice(puncts)
            
            variations.append(new_text)
    
    return variations

def create_ambiguous_examples(hate_texts, neutral_texts, counter_hate_texts, n=300):
    """Create ambiguous examples that mix elements from different categories"""
    ambiguous_examples = []
    labels = []
    
    for _ in range(n):
        # Pick base text category
        category = random.choice(['Hate', 'Neutral', 'Counter hate'])
        if category == 'Hate':
            base = random.choice(hate_texts)
            # Add some neutral or counter-hate elements
            other = random.choice(neutral_texts + counter_hate_texts)
        elif category == 'Neutral':
            base = random.choice(neutral_texts)
            # Add some hate or counter-hate elements
            other = random.choice(hate_texts + counter_hate_texts)
        else:
            base = random.choice(counter_hate_texts)
            # Add some neutral or hate elements
            other = random.choice(neutral_texts + hate_texts)
        
        # Extract phrases from both
        base_words = base.split()
        other_words = other.split()
        
        # Create a mixed example
        if len(other_words) > 3:
            insertion_point = random.randint(0, len(base_words))
            phrase_length = random.randint(1, min(3, len(other_words)))
            phrase_start = random.randint(0, len(other_words) - phrase_length)
            
            # Insert phrase from other category
            mixed_words = base_words[:insertion_point] + \
                        other_words[phrase_start:phrase_start+phrase_length] + \
                        base_words[insertion_point:]
            
            mixed_text = ' '.join(mixed_words)
            
            # Apply some additional transformations
            mixed_text = add_noise(mixed_text, noise_level=0.2)
            
            ambiguous_examples.append(mixed_text)
            labels.append(category)
    
    return ambiguous_examples, labels

def create_synthetic_dataset(output_file='data/synthetic_asian_hate_tweets.csv', n_samples=4511):
    """Create a synthetic dataset of tweets about COVID-19 and Asian hate speech"""
    
    # Base texts for each category
    hate_texts = [
        "China virus has killed thousands of people worldwide",
        "Chinese people are responsible for the Wuhan virus outbreak",
        "These Chinese people brought COVID to our country",
        "Stay away from Asian restaurants, they spread the virus",
        "The Chinese government created this virus as a bioweapon",
        "Kung-flu is destroying our economy thanks to China",
        "Chinese virus is ruining everything",
        "All these Asians coming here spreading disease",
        "China needs to pay for what they've done to the world",
        "Asian people should go back to their country with their virus",
        "Chinese people eat bats and that's how the virus started",
        "I'm staying away from all Asians until this virus thing is over",
        "They should ban all flights from China and other Asian countries",
        "Wuhan virus is the biggest attack on America",
        "Those Asians don't even wear masks in their own stores",
        "We should make China pay reparations for this Chinese flu",
        "Asians are the reason we're all in lockdown now",
        "Chinese coronavirus has ruined my business",
        "Those people brought their nasty habits and viruses here",
        "Asians are walking around spreading their germs everywhere"
    ]
    
    counter_hate_texts = [
        "Stop blaming Asian people for COVID-19, racism is not the answer",
        "We need to stand with our Asian community during these difficult times",
        "COVID-19 is not the 'Chinese virus', stop using racist terms",
        "Racism against Asians has increased due to COVID, we must speak out",
        "Asian Americans are not responsible for the coronavirus",
        "Supporting our local Asian businesses during this pandemic",
        "Stand against anti-Asian racism in the time of COVID",
        "Being Asian doesn't make someone more likely to have COVID",
        "Blaming Asians for coronavirus is ignorant and harmful",
        "COVID-19 doesn't discriminate and neither should we",
        "The rise in hate crimes against Asians must stop now",
        "Calling it the 'Chinese virus' incites racism against all Asians",
        "Asian healthcare workers are fighting COVID alongside everyone else",
        "Stop the hate against our AAPI community #StopAsianHate",
        "My Asian friends are afraid to go outside because of racist attacks",
        "This isn't about a country or ethnicity, it's a global health crisis",
        "Solidarity with Asian Americans facing discrimination right now",
        "Using terms like 'Chinese virus' is harmful to the Asian community",
        "Racism is not an appropriate response to a pandemic",
        "The virus doesn't have a nationality, stop the xenophobia"
    ]
    
    neutral_texts = [
        "COVID-19 cases continue to rise across the country",
        "Scientists are working on developing vaccines for the coronavirus",
        "Remember to wash your hands and practice social distancing",
        "Working from home during the pandemic has its challenges",
        "Mask mandates are being implemented in several states",
        "COVID-19 testing is available at local health centers",
        "The pandemic has changed how we work and socialize",
        "Health officials recommend staying 6 feet apart from others",
        "Many businesses have closed temporarily due to COVID-19",
        "Online learning has become the norm during the pandemic",
        "The coronavirus pandemic has affected the global economy",
        "Public gatherings are limited to prevent virus spread",
        "COVID symptoms include fever, cough, and loss of taste",
        "Hospitals are overwhelmed with coronavirus patients",
        "Social distancing helps reduce the spread of COVID-19",
        "Many countries have implemented travel restrictions",
        "Essential workers continue to serve during the pandemic",
        "Schools are implementing hybrid learning models",
        "WHO provides guidelines for COVID-19 prevention",
        "Vaccine distribution has begun in several countries"
    ]
    
    # Define exact numbers for each category
    n_hate = 1244
    n_neutral = 2862
    n_counter_hate = 405
    
    # Create variations to expand the dataset
    # Calculate required variations per base text
    hate_variations_per_text = max(1, (n_hate // len(hate_texts)) - 1)
    neutral_variations_per_text = max(1, (n_neutral // len(neutral_texts)) - 1)
    counter_hate_variations_per_text = max(1, (n_counter_hate // len(counter_hate_texts)) - 1)
    
    # Create variations with increased ambiguity
    hate_variations = create_variations(hate_texts, n_variations=hate_variations_per_text, ambiguity_level=0.3)
    counter_hate_variations = create_variations(counter_hate_texts, n_variations=counter_hate_variations_per_text, ambiguity_level=0.3)
    neutral_variations = create_variations(neutral_texts, n_variations=neutral_variations_per_text, ambiguity_level=0.3)
    
    # Create some ambiguous examples with mixed characteristics
    ambiguous_hate, ambiguous_hate_labels = create_ambiguous_examples(hate_texts, neutral_texts, counter_hate_texts, n=200)
    ambiguous_neutral, ambiguous_neutral_labels = create_ambiguous_examples(neutral_texts, hate_texts, counter_hate_texts, n=300)
    ambiguous_counter, ambiguous_counter_labels = create_ambiguous_examples(counter_hate_texts, neutral_texts, hate_texts, n=100)
    
    # Add ambiguous examples to the variations
    hate_variations.extend([text for text, label in zip(ambiguous_hate, ambiguous_hate_labels) if label == 'Hate'])
    neutral_variations.extend([text for text, label in zip(ambiguous_neutral, ambiguous_neutral_labels) if label == 'Neutral'])
    counter_hate_variations.extend([text for text, label in zip(ambiguous_counter, ambiguous_counter_labels) if label == 'Counter hate'])
    
    # Sample exact numbers from variations
    sampled_hate = random.sample(hate_variations, min(n_hate, len(hate_variations)))
    while len(sampled_hate) < n_hate:
        # Create more variations if needed
        more_variations = create_variations(hate_texts, n_variations=10, ambiguity_level=0.4)
        remaining = n_hate - len(sampled_hate)
        sampled_hate.extend(random.sample(more_variations, min(remaining, len(more_variations))))
    
    sampled_neutral = random.sample(neutral_variations, min(n_neutral, len(neutral_variations)))
    while len(sampled_neutral) < n_neutral:
        # Create more variations if needed
        more_variations = create_variations(neutral_texts, n_variations=10, ambiguity_level=0.4)
        remaining = n_neutral - len(sampled_neutral)
        sampled_neutral.extend(random.sample(more_variations, min(remaining, len(more_variations))))
    
    sampled_counter_hate = random.sample(counter_hate_variations, min(n_counter_hate, len(counter_hate_variations)))
    while len(sampled_counter_hate) < n_counter_hate:
        # Create more variations if needed
        more_variations = create_variations(counter_hate_texts, n_variations=10, ambiguity_level=0.4)
        remaining = n_counter_hate - len(sampled_counter_hate)
        sampled_counter_hate.extend(random.sample(more_variations, min(remaining, len(more_variations))))
    
    # Combine all samples
    tweets = sampled_hate + sampled_neutral + sampled_counter_hate
    labels = ['Hate'] * len(sampled_hate) + ['Neutral'] * len(sampled_neutral) + ['Counter hate'] * len(sampled_counter_hate)
    
    # Generate random dates for the tweets
    dates = generate_dates(n=len(tweets))
    
    # Create user IDs
    user_ids = [f"user_{i:06d}" for i in range(1, len(tweets) + 1)]
    
    # Create dataframe
    df = pd.DataFrame({
        'tweet_id': range(1, len(tweets) + 1),
        'user_id': user_ids,
        'date': dates,
        'text': tweets,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Created synthetic dataset with {len(df)} tweets")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print(f"Dataset saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Create the synthetic dataset
    df = create_synthetic_dataset()
    
    # Display sample tweets
    print("\nSample tweets from each category:")
    
    print("\nHate tweets:")
    for tweet in df[df['label'] == 'Hate']['text'].head(3).values:
        print(f"- {tweet}")
    
    print("\nNeutral tweets:")
    for tweet in df[df['label'] == 'Neutral']['text'].head(3).values:
        print(f"- {tweet}")
    
    print("\nCounter hate tweets:")
    for tweet in df[df['label'] == 'Counter hate']['text'].head(3).values:
        print(f"- {tweet}")