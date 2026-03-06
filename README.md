# Pokémon AI Controller with CLIP

An AI-powered controller that plays Pokémon Leaf Green using vision-language models (CLIP) to decide actions based on what it sees on screen.

## What It Does

This project uses OpenAI's CLIP model to:
1. Capture screenshots from a GBA emulator running in the browser
2. Understand the visual context of the game state
3. Select appropriate actions (movement, buttons) based on natural language descriptions
4. Execute those actions via Selenium browser automation

The system maps natural language action descriptions (e.g., "go right", "talk using A", "open the menu") to game controls and uses CLIP's image-text matching to decide which action makes sense given the current screen.

## Components

### 1. Control Collection (`controls_library.py`)
- Loads CLIP ViT-B/32 model for vision-language understanding
- Maps 100+ natural language sentences to game controls via `control_mappings.json`
- Samples random control candidates and ranks them using CLIP similarity
- **Includes trainable linear projection layer with RL** (currently experimental)
- Tracks action priors to debias frequently selected actions

### 2. Emulator Controller (`emulator_controller.py`)
- Selenium-based browser automation for the GBA emulator
- Interactive corner selection to define screenshot area
- Maps abstract controls (A, B, ↑, ↓, ←, →, select) to keyboard inputs
- Ad-blocking configuration to prevent pop-ups
- Screenshot capture and action execution

### 3. Main Loop (`main.py`)
- Integrates CLIP model with emulator
- Runs perception-action loop: capture → decide → execute
- Saves periodic screenshots and statistics
- Displays action frequencies and diversity metrics

### 4. Control Mappings (`control_mappings.json`)
Natural language templates for each control, such as:
```json
{
  "A": ["press A", "use the A button", "talk using A", ...],
  "^": ["go up", "move upward", "head north", ...],
  ...
}
```

## Current RL Algorithm

**The current RL algorithm is ass.** 

It optimizes a linear projection on the image encoder to maximize visual diversity of screenshots. The problem? It just learns to spam the Start/Select button to open and close menus forever because that creates maximum screen variation. Completely useless for actually playing the game.

### Why It's Ass:
- **Objective is too shallow**: Diversity of pixels ≠ game progress
- **No understanding of game state**: Opening menus 1000 times isn't exploration
- **Greedy short-term reward**: Maximizes immediate visual change, ignores long-term goals
- **Wrong policy model**: A single linear layer can't capture complex action strategies

## TODO: Make the RL Not Ass

### Better Objectives
- [ ] **Progress-based rewards**: Track in-game progress (badges, Pokédex entries, story flags)
- [ ] **Exploration rewards**: Reward visiting new map tiles/areas (requires map tracking)
- [ ] **Sparse achievement rewards**: Major milestones (catching Pokémon, winning battles, getting items)
- [ ] **Curiosity-driven intrinsic motivation**: Use prediction error instead of raw diversity
- [ ] **Multi-objective optimization**: Balance exploration + progress + efficiency

### Better Policy Models
- [ ] **Replace linear projection with MLP**: Multi-layer network for image→action mapping
- [ ] **Add recurrent/memory component**: LSTM/GRU to track temporal game state
- [ ] **Full RL policy network**: Actor-critic with proper value estimation
- [ ] **Transformer-based policy**: Use temporal attention over recent frames
- [ ] **Behavioral cloning pretraining**: Learn from human gameplay first
- [ ] **Hierarchical RL**: High-level goals (get to next city) → low-level actions (navigate maze)

### Training Improvements
- [ ] **Experience replay buffer**: Learn from past trajectories
- [ ] **PPO/A3C instead of policy gradient**: More stable RL algorithms
- [ ] **Curriculum learning**: Start with simple tasks, gradually increase difficulty
- [ ] **Reward shaping**: Dense intermediate rewards to guide learning
- [ ] **Multi-agent setup**: Different policies for battle vs exploration vs menus

## Installation
```bash
# Install dependencies
pip install torch open-clip-torch pillow numpy selenium webdriver-manager

# Install Chrome WebDriver
# (or use webdriver-manager to auto-download)

# Prepare control mappings
# Ensure control_mappings.json is in the project directory
```

## Usage
```bash
# Run the main loop
python main.py
```

**Setup steps:**
1. Browser opens to the emulator URL
2. Click **top-left corner** of game screen when prompted
3. Click **bottom-right corner** of game screen when prompted
4. AI starts playing automatically

**Controls:**
- Press `Ctrl+C` to stop
- Screenshots saved to `screenshots/` every 50 steps
- Statistics printed every 100 steps

## Configuration

Edit `main.py` to adjust:
- `num_steps`: Number of actions to take (default: 1000)
- `action_duration`: How long to hold buttons (default: 0.15s)
- `step_delay`: Delay between actions (default: 0.3s)
- `temperature`: CLIP softmax temperature (default: 1.0)
- `train_rl`: Enable/disable RL training (default: True)

## Files
```
.
├── controls_library.py          # CLIP model + RL projection
├── emulator_controller.py       # Selenium automation
├── main.py                      # Main game loop
├── control_mappings.json        # Natural language control templates
├── screenshots/                 # Captured game frames
└── README.md                    # This file
```

## Current Limitations

- RL is terrible (see above)
- No game state parsing (HP, inventory, position, etc.)
- No long-term planning or strategy
- Can't distinguish battle from exploration contexts
- Action selection is memoryless (no temporal reasoning)
- Prone to getting stuck in loops

## Future Work

Make the RL not ass (see TODO section above).

## License

MIT (or whatever, it's just a fun experiment)