# Prompt Generation Nodes - User Guide

## Overview

The new prompt generation nodes make it incredibly easy to create amazing, unique prompts from your extensive JSON data collection. No more struggling with which files to use or how to combine elements - these nodes do the heavy lifting for you!

## Quick Start - For Instant Amazing Results

### 🚀 QuickPromptGenerator (One-Click Amazing Prompts)

**Best for:** When you want instant, professional-quality prompts with minimal effort.

**How to use:**
1. Add the "Quick Prompt Generator" node to your workflow
2. Choose a prompt style (e.g., "Amazing Portrait", "Stunning Fashion", "Epic Fantasy")
3. Set gender preference (Any/Female/Male)
4. Select quality level (Good/Great/Amazing/Legendary)
5. Optionally add prefix elements (added at the beginning)
6. Optionally add suffix elements (added at the end)
7. Hit generate!

**Example outputs:**
- "Amazing Portrait": *"breathtaking portrait of Sarah, German person, platinum blonde layered hair, joyful expression, dress shirt, cardigan, earrings, standing with hands on hips, award-winning, stunning, photorealistic, cinematic lighting, digital painting"*
- "Epic Fantasy": *"legendary masterpiece fantasy character of Marcus, Japanese person, silver wavy hair, mysterious expression, hakama, magical atmosphere, fantasy art, ethereal"*

### 📋 PromptTemplateManager (Pre-Built Templates)

**Best for:** When you want consistent, themed prompts with professional structure.

**Templates available:**
- **Elegant Portrait**: Sophisticated, refined portraits
- **Casual Fashion**: Relaxed, contemporary looks
- **Fantasy Character**: Magical, ethereal beings
- **Action Pose**: Dynamic, energetic scenes
- **Romantic Scene**: Tender, emotional moments
- **Professional Look**: Confident, polished appearance
- **Vintage Style**: Nostalgic, classic aesthetics

**How to use:**
1. Select a template that matches your vision
2. Choose gender and variation level
3. Add custom elements if desired
4. Generate your templated prompt

## Advanced Tools

### 🎲 RandomJSONSelector (Explore Your Data)

**Best for:** When you want to explore specific categories or randomly discover new elements.

**Features:**
- Access all your JSON files directly
- Select from specific categories or random ones
- Control how many items to select
- Customize separators

**Example:** Select 3 random items from "attire_female_topwear.json" → "blouse, cardigan, tank top"

### 🧠 SmartPromptBuilder (Intelligent Combination)

**Best for:** When you want AI-powered intelligent prompt construction.

**Prompt Types:**
- **Portrait**: Character-focused with facial details
- **Fashion**: Clothing and style emphasis
- **Artistic**: Creative and artistic elements
- **Scenic**: Location and environment focus
- **Character**: Full character development
- **Fantasy**: Magical and ethereal elements

**Complexity Levels:**
- **Simple**: Basic elements only
- **Medium**: Balanced detail level
- **Complex**: Maximum detail with creative elements

### ✨ PromptEnhancer (Upgrade Any Prompt)

**Best for:** When you have a basic prompt and want to make it incredible.

**Enhancement Types:**
- **Artistic Style**: Adds artistic flair and style terms
- **Lighting Effects**: Enhances with lighting descriptions
- **Mood Enhancement**: Adds emotional depth
- **Detail Boost**: Increases detail level
- **Composition**: Improves visual composition
- **Technical Quality**: Adds quality and technical terms

**Intensity Levels:**
- **Subtle**: Light enhancement
- **Moderate**: Balanced improvement
- **Strong**: Maximum enhancement

## Workflow Examples

### Example 1: Quick Amazing Portrait
```
[QuickPromptGenerator]
├── Style: "Amazing Portrait"
├── Subject: "Person"
├── Gender: "Female"
├── Quality: "Amazing"
└── Output: Complete amazing prompt ready to use
```

### Example 2: Custom Fashion Look
```
[SmartPromptBuilder]
├── Type: "Fashion"
├── Gender: "Female"
├── Complexity: "Complex"
├── Style: "modern, elegant"
└── Output: Intelligent fashion prompt

[PromptEnhancer]
├── Input: Previous output
├── Enhancement: "Artistic Style"
├── Intensity: "Strong"
└── Output: Enhanced artistic fashion prompt
```

### Example 3: Explore and Combine
```
[RandomJSONSelector] → Hair Color
[RandomJSONSelector] → Female Clothing
[RandomJSONSelector] → Poses
[RandomJSONSelector] → Artistic Style
    ↓
[Combine results into final prompt]
```

## Pro Tips

### 🎯 For Best Results:

1. **Start with QuickPromptGenerator** - It creates complete, professional prompts instantly
2. **Use PromptTemplateManager** for consistent themes and professional structure
3. **Experiment with seeds** - Same settings + different seeds = infinite variations
4. **Layer enhancements** - Use PromptEnhancer on already good prompts for amazing results
5. **Mix and match** - Combine outputs from different nodes for unique results

### 🔧 Customization Tips:

1. **Use custom additions** - Add your own elements to any node
2. **Exclude categories** - Remove unwanted elements (e.g., exclude "nudity" categories)
3. **Style preferences** - Specify your preferred artistic style
4. **Variation levels** - Control how much randomness you want

### 🎨 Creative Workflows:

1. **The Layered Approach:**
   - Start with PromptTemplateManager
   - Enhance with PromptEnhancer
   - Add random elements with RandomJSONSelector

2. **The Exploration Method:**
   - Use RandomJSONSelector to discover new combinations
   - Build complete prompts with SmartPromptBuilder
   - Polish with PromptEnhancer

3. **The One-Click Magic:**
   - Just use QuickPromptGenerator with different seeds
   - Instant professional results every time

## File Organization

Your JSON files are automatically organized by category:

**Character & Appearance:**
- Names (male/female)
- Hair colors and styles
- Facial expressions
- Nationality/ethnicity

**Clothing & Accessories:**
- Gender-specific clothing
- Jewelry and accessories
- Styles and patterns
- Traditional clothing

**Poses & Posture:**
- Basic poses
- Arm positions
- Movement and action
- Multiple character poses

**Environment & Style:**
- Locations and scenes
- Art styles
- Materials and textures
- Creative concepts

## Troubleshooting

**If a node returns empty results:**
- Check if the JSON file exists
- Verify category names are correct
- Try using "Any" gender selection
- Use a different seed value

**For more variety:**
- Change the seed value
- Try different complexity/variation levels
- Use custom additions
- Combine multiple nodes

## Next Steps

1. Start with **QuickPromptGenerator** for instant results
2. Experiment with different **seeds** to get variations
3. Try **PromptTemplateManager** for themed prompts
4. Use **PromptEnhancer** to upgrade existing prompts
5. Explore **RandomJSONSelector** to discover new elements

The system is designed to make prompt generation effortless while giving you access to all your rich data. Have fun creating amazing prompts!
