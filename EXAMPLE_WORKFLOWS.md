# Example Workflows for Prompt Generation

## Workflow 1: Instant Amazing Prompts

### Setup:
1. Add "Quick Prompt Generator" node
2. Connect output to your text input

### Settings:
- **Prompt Style**: "Amazing Portrait"
- **Subject Type**: "Person"
- **Gender**: "Female"
- **Quality Level**: "Amazing"
- **Seed**: -1 (for random)

### Expected Output:
"breathtaking portrait of Elena, Italian person, auburn layered hair, radiant expression, blouse, jeans, necklace, standing with hands on hips, award-winning, stunning, photorealistic, cinematic lighting, digital painting"

---

## Workflow 2: Fashion Photography

### Setup:
1. Add "Prompt Template Manager" node
2. Connect to "Prompt Enhancer" node
3. Connect final output to your text input

### Template Manager Settings:
- **Template**: "Casual Fashion"
- **Gender**: "Female"
- **Variation Level**: "High"
- **Seed**: 12345

### Enhancer Settings:
- **Enhancement Type**: "Lighting Effects"
- **Intensity**: "Strong"
- **Seed**: 12345

### Expected Output:
"casual fashion portrait of Maya, Swedish person, platinum blonde shoulder-length curls, cheerful expression, t-shirt, jeans, earrings, relaxed pose, contemporary, digital painting, cinematic lighting, god rays, volumetric lighting, ethereal glow"

---

## Workflow 3: Fantasy Character Creation

### Setup:
1. Add "Smart Prompt Builder" node
2. Connect to "Prompt Enhancer" node
3. Connect final output to your text input

### Smart Builder Settings:
- **Prompt Type**: "Fantasy"
- **Gender**: "Female"
- **Complexity**: "Complex"
- **Style Preference**: "magical, ethereal"
- **Seed**: 54321

### Enhancer Settings:
- **Enhancement Type**: "Artistic Style"
- **Intensity**: "Strong"
- **Custom Enhancements**: "mystical, otherworldly"
- **Seed**: 54321

### Expected Output:
"fantasy character Luna, Chinese person, dragon scale emerald wavy hair, mysterious expression, traditional hanfu, magical forest, ethereal pose, magical, ethereal, fantasy art, enchanted moonlight, breathtaking masterpiece, award-winning, legendary artwork, mystical, otherworldly"

---

## Workflow 4: Explore and Combine

### Setup:
1. Add multiple "Random JSON Selector" nodes
2. Combine outputs manually or with text processing
3. Use "Prompt Enhancer" for final polish

### Selector 1 - Hair:
- **JSON File**: "hair_female.json"
- **Category**: "" (random)
- **Count**: 1
- **Seed**: 111

### Selector 2 - Clothing:
- **JSON File**: "attire_female_topwear.json"
- **Category**: "" (random)
- **Count**: 1
- **Seed**: 222

### Selector 3 - Pose:
- **JSON File**: "poses.json"
- **Category**: "" (random)
- **Count**: 1
- **Seed**: 333

### Selector 4 - Style:
- **JSON File**: "styles.json"
- **Category**: "" (random)
- **Count**: 1
- **Seed**: 444

### Manual Combination:
"beautiful woman, [hair result], [clothing result], [pose result], [style result]"

### Final Enhancement:
- **Enhancement Type**: "Technical Quality"
- **Intensity**: "Strong"

---

## Workflow 5: Professional Portrait

### Setup:
1. Add "Quick Prompt Generator" node
2. Connect to "Prompt Enhancer" node (optional)
3. Connect to your text input

### Quick Generator Settings:
- **Prompt Style**: "Professional Photo"
- **Subject Type**: "Professional"
- **Gender**: "Male"
- **Quality Level**: "Great"
- **Prefix**: "corporate headshot"
- **Suffix**: "business suit, confident smile, office background"
- **Seed**: 777

### Optional Enhancement:
- **Enhancement Type**: "Professional"
- **Intensity**: "Moderate"

### Expected Output:
"corporate headshot, beautiful professional portrait of professional James, German person, dark brown crew cut, confident expression, studio quality, commercial, professional lighting, clean composition, professional, polished, studio quality, business suit, confident smile, office background"

---

## Tips for Best Results:

1. **Use consistent seeds** across connected nodes for coherent results
2. **Experiment with different quality levels** to find your sweet spot
3. **Try different prompt styles** to discover new creative directions
4. **Use custom additions** to add your specific requirements
5. **Layer enhancements** for increasingly polished results

## Seed Management:

- **Use -1** for completely random results
- **Use the same seed** across multiple nodes for consistency
- **Change seeds gradually** (777, 778, 779) for variations on a theme
- **Use memorable seeds** (birthdays, etc.) to recreate favorite results

## Common Combinations:

- **Portrait + Lighting Enhancement** = Professional headshots
- **Fashion + Artistic Style** = Editorial photography
- **Fantasy + Mood Enhancement** = Cinematic characters
- **Casual + Detail Boost** = Lifestyle photography
- **Professional + Technical Quality** = Corporate portraits
