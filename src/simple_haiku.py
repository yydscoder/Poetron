"""
Simple rule-based haiku generator  
More reliable than pre-trained models that weren't trained on poetry
"""
import random
import re


class SimpleHaikuGenerator:
    """Generate coherent haikus using templates and word banks"""
    
    def __init__(self):
        # 5-syllable line templates and words
        self.line_5 = [
            ("{nature} {motion} {time}", ["cherry blossoms", "autumn leaves", "winter snow", "spring rain"], 
             ["drift down", "fall soft", "dance light"], ["at dusk", "at dawn"]),
            ("{adjective} {noun} {verb}", ["silent", "gentle", "peaceful", "quiet"],
             ["moonlight", "starlight", "sunlight", "twilight"], ["glows bright", "shines through", "breaks forth"]),
            ("{noun} on the {place}", ["Mist", "Dew", "Frost", "Snow"], 
             ["hillside", "river", "meadow", "branches"]),
        ]
        
        # 7-syllable line templates and words  
        self.line_7 = [
            ("{adjective} {noun} {motion} {place}", ["gentle", "soft", "cool", "warm"],
             ["breeze", "wind", "rain", "mist"], ["whispers through", "flows across", "dances on"], 
             ["the forest", "the valley", "the mountain"]),
            ("{noun} {verb} in {adjective} {time}", ["birds", "flowers", "clouds", "leaves"],
             ["sing", "bloom", "drift", "turn"], ["the", "this"], ["autumn day", "spring morning", "winter cold"]),
            ("{verb} the {adjective} {noun} {motion}", ["watching", "seeing", "hearing"],
             ["pale", "bright", "dark", "soft"], ["moon", "sun", "stars", "sky"], ["rise slowly", "fade away"]),
        ]
        
        # Word banks themed by topic
        self.themes = {
            'spring': {
                'nouns': ['blossoms', 'flowers', 'petals', 'buds', 'blooms'],
                'adjectives': ['fresh', 'new', 'bright', 'soft', 'tender'],
                'verbs': ['unfold', 'awaken', 'emerge', 'bloom', 'grow']
            },
            'summer': {
                'nouns': ['sunshine', 'heat', 'cicadas', 'breeze', 'gardens'],
                'adjectives': ['warm', 'bright', 'golden', 'lazy', 'long'],
                'verbs': ['blaze', 'shimmer', 'hum', 'flow', 'ripen']
            },
            'autumn': {
                'nouns': ['leaves', 'harvest', 'wind', 'geese', 'maples'],
                'adjectives': ['crisp', 'golden', 'red', 'falling', 'cool'],
                'verbs': ['drift', 'turn', 'fly', 'gather', 'fade']
            },
            'winter': {
                'nouns': ['snow', 'frost', 'ice', 'cold', 'silence'],
                'adjectives': ['white', 'frozen', 'still', 'quiet', 'bare'],
                'verbs': ['falls', 'covers', 'blankets', 'freezes', 'sleeps']
            },
            'night': {
                'nouns': ['moonlight', 'stars', 'darkness', 'silence', 'shadows'],
                'adjectives': ['silver', 'quiet', 'deep', 'pale', 'dark'],
                'verbs': ['glows', 'shines', 'whispers', 'falls', 'watches']
            },
            'water': {
                'nouns': ['river', 'pond', 'rain', 'stream', 'waves'],
                'adjectives': ['flowing', 'clear', 'gentle', 'deep', 'still'],
                'verbs': ['flows', 'ripples', 'falls', 'mirrors', 'reflects']
            }
        }
        
        # Simple haiku patterns (5-7-5 syllables)
        self.patterns = [
            # Pattern 1: Nature observation
            [
                "Silent {noun1} waits",
                "{Adjective} {noun2} {verb} {adverb}",
                "{Noun3}'s gentle {noun4}"
            ],
            # Pattern 2: Seasonal imagery  
            [
                "{Noun1} {verb1} {prep} {noun2}",
                "{Adjective} {time} brings new {noun3}",
                "{Noun4} at {time2}"
            ],
            # Pattern 3: Moment in time
            [
                "In the {time} light",
                "{Adjective} {noun1} {verb} and {verb2}",
                "{Noun2}  waits alone"
            ],
        ]
    
    def generate(self, prompt: str = "nature", num_haikus: int = 1) -> list:
        """Generate haikus based on prompt"""
        # Detect theme from prompt
        theme = self._detect_theme(prompt)
        
        haikus = []
        for _ in range(num_haikus):
            haiku = self._generate_one(theme, prompt)
            haikus.append(haiku)
        
        return haikus
    
    def _detect_theme(self, prompt: str) -> str:
        """Detect theme from prompt text"""
        prompt_lower = prompt.lower()
        
        # Check for theme keywords
        for theme in self.themes.keys():
            if theme in prompt_lower:
                return theme
        
        # Check for related words
        if any(word in prompt_lower for word in ['flower', 'blossom', 'cherry', 'bloom']):
            return 'spring'
        if any(word in prompt_lower for word in ['sun', 'hot', 'summer']):
            return 'summer'
        if any(word in prompt_lower for word in ['leaf', 'leaves', 'fall', 'maple']):
            return 'autumn'
        if any(word in prompt_lower for word in ['snow', 'ice', 'cold', 'frost']):
            return 'winter'
        if any(word in prompt_lower for word in ['moon', 'star', 'night', 'dark']):
            return 'night'
        if any(word in prompt_lower for word in ['rain', 'river', 'stream', 'water', 'ocean', 'sea']):
            return 'water'
        
        # Default to autumn (classic haiku theme)
        return 'autumn'
    
    def _generate_one(self, theme: str, prompt: str) -> str:
        """Generate a single haiku using the actual prompt"""
        words = self.themes.get(theme, self.themes['autumn'])
        
        # Extract key noun from prompt (use first meaningful word)
        prompt_words = prompt.lower().split()
        prompt_noun = prompt_words[0] if prompt_words else 'nature'
        
        # Fix common typos
        typo_fixes = {
            'mountians': 'mountains',
            'occean': 'ocean',
            'forrest': 'forest'
        }
        prompt_noun = typo_fixes.get(prompt_noun, prompt_noun)
        
        # Determine if plural (simple heuristic)
        is_plural = prompt_noun.endswith('s') and not prompt_noun.endswith('ss')
        
        # Conjugate verbs based on singular/plural
        def conjugate(base_verb):
            if is_plural:
                return base_verb  # "flowers wait", "seas flow"
            else:
                # Add 's' for singular third person
                if base_verb.endswith(('s', 'sh', 'ch', 'x', 'z')):
                    return base_verb + 'es'
                elif base_verb.endswith('y'):
                    return base_verb[:-1] + 'ies'
                else:
                    return base_verb + 's'
        
        # Build three lines that reference the prompt
        lines = []
        
        # Line 1 (5 syllables) - introduce the subject
        adj = random.choice(words['adjectives']).capitalize()
        verb1 = conjugate(random.choice(['rest', 'wait', 'bloom', 'stand']))
        
        templates_5 = [
            f"Silent {prompt_noun} {conjugate('wait')}",
            f"{adj} {prompt_noun} {verb1}",
            f"Deep {prompt_noun} at rest"
        ]
        lines.append(random.choice(templates_5))
        
        # Line 2 (7 syllables) - develop the image
        element = random.choice(['wind', 'rain', 'light', 'mist'])
        feeling = random.choice(['peace', 'calm', 'grace', 'hope'])
        pronoun_subj = 'they' if is_plural else 'it'  # subject form
        pronoun_obj = 'them' if is_plural else 'it'   # object form
        verb2 = 'rest' if is_plural else 'rests'
        
        templates_7 = [
            f"{random.choice(words['adjectives'])} {element} brings {feeling} and {random.choice(['grace', 'hope', 'joy'])}",
            f"beneath the {random.choice(['pale', 'bright', 'soft'])} {random.choice(['moon', 'sun', 'stars'])} {pronoun_subj} {verb2}",
            f"watching {pronoun_obj} {random.choice(['dance', 'sway', 'drift'])} in {element}"
        ]
        lines.append(random.choice(templates_7))
        
        # Line 3 (5 syllables) - concluding image
        templates_5_end = [
            f"{random.choice(['seasons', 'moments'])} {random.choice(['flow', 'pass', 'drift'])} onward",
            f"{random.choice(['peace', 'joy', 'grace'])} fills the {random.choice(['air', 'sky', 'world'])}",
            f"nature finds its way"
        ]
        lines.append(random.choice(templates_5_end))
        
        return '\n'.join(lines)


def generate_simple_haiku(prompt: str='nature', num_haikus: int = 1) -> list:
    """Convenience function"""
    generator = SimpleHaikuGenerator()
    return generator.generate(prompt, num_haikus)


if __name__ == "__main__":
    generator = SimpleHaikuGenerator()
    
    test_prompts = ['spring', 'moonlight', 'river', 'winter']
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Theme: {prompt}")
        print('='*50)
        haikus = generator.generate(prompt, 1)
        print(haikus[0])
