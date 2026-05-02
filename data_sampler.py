import json
import random
import os
import sys
from typing import List, Dict, Any
from openai import OpenAI

# Add agent directory to sys.path to import memory
sys.path.append(os.path.join(os.getcwd(), 'agent'))
try:
    from memory import StrictMemoryManager
except ImportError:
    # If running from within agent or other structure issues
    sys.path.append(os.getcwd())
    from memory import StrictMemoryManager

class DataSampler:
    def __init__(self, raw_json_path: str = "CPsyCounR.json"):
        self.raw_json_path = raw_json_path
        self.config = StrictMemoryManager().get_config()
        self.client = OpenAI(
            base_url=self.config["api_config"]["openai"]["base_url"],
            api_key=self.config["api_config"]["openai"]["api_key"],
        )

    def load_data(self) -> List[Dict]:
        print(f"Loading data from {self.raw_json_path}...")
        with open(self.raw_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def sample_by_categories(self, data: List[Dict], n_categories: int = 10, n_per_category: int = 1) -> List[Dict]:
        category_map = {}
        for item in data:
            categories = item.get("案例类别", [])
            if not categories:
                continue
            # Use the first category as the primary one for grouping
            cat = categories[0]
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(item)
        
        # Sample categories
        available_categories = list(category_map.keys())
        sampled_categories = random.sample(available_categories, min(n_categories, len(available_categories)))
        print(f"Sampled categories: {sampled_categories}")
        
        sampled_items = []
        for cat in sampled_categories:
            items_in_cat = category_map[cat]
            sampled_items.extend(random.sample(items_in_cat, min(n_per_category, len(items_in_cat))))
        
        return sampled_items

    def translate_case(self, case_data: Dict) -> Dict:
        # Prepare content for translation
        # Note: fields in source are lists of strings
        chinese_content = {
            "title": case_data.get("案例标题", ""),
            "categories": case_data.get("案例类别", []),
            "summary": "".join(case_data.get("案例简述", [])),
            "process": "".join(case_data.get("咨询经过", []))
        }
        
        prompt = f"""
##Task:
Translate the following Chinese psychological counseling case information into English.
Ensure the translation is professional, clinical, and accurate.

[Chinese Content]:
- Title: {chinese_content['title']}
- Categories: {', '.join(chinese_content['categories'])}
- Summary: {chinese_content['summary']}
- Process: {chinese_content['process']}

##Requirement:
Your output must be a JSON object with the following keys:
- "case title": Translated title.
- "case category": Translated categories as a list of strings.
- "case summary": Professional English summary of the patient's condition.
- "case description": Detailed patient backstory and description (expand reasonably if needed).
- "consultation process": Detailed clinical consultation process.

Strictly output a JSON object. Do not provide any explanation.
"""
        response = self.client.chat.completions.create(
            model=self.config["api_config"]["openai"]["model"],
            messages=[
                {"role": "system", "content": "You are a professional medical translator specializing in psychological counseling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        if response.choices:
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {content}")
                return {}
        return {}

    def run(self, n_categories=10, n_per_category=1, output_path="raw_data_translate.json"):
        data = self.load_data()
        print(f"Sampling {n_categories} categories...")
        samples = self.sample_by_categories(data, n_categories, n_per_category)
        
        translated_results = []
        for i, sample in enumerate(samples):
            patient_id = f"patient_{i+1:03d}"
            title = sample.get('案例标题', 'Unknown Title')
            if isinstance(title, list):
                title = title[0] if title else "Unknown Title"
            
            print(f"Processing and translating {patient_id}: {title}...")
            translated = self.translate_case(sample)
            if translated:
                translated_results.append({patient_id: translated})
            else:
                print(f"Failed to translate {patient_id}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_results, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(translated_results)} cases to {output_path}")

if __name__ == "__main__":
    # Example usage: sample 10 categories, 1 per category
    sampler = DataSampler()
    sampler.run(n_categories=10, n_per_category=1)
