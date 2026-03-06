import time
from controls_library import ControlCollection
from emulator import EmulatorController
from PIL import Image
import numpy as np

def main():
    """
    Main game loop that integrates AI control with the emulator.
    """
    print("\n" + "="*60)
    print("POKÉMON AI CONTROLLER")
    print("="*60 + "\n")
    
    # Initialize the control collection (CLIP model)
    print("Loading CLIP model...")
    control_collection = ControlCollection(device='cuda')
    print(f"Loaded {len(control_collection.idx2sentence)} control sentences\n")
    
    # Initialize the emulator controller
    print("Opening emulator...")
    emulator = EmulatorController()
    print("Emulator ready!\n")
    
    # Wait for game to fully load
    print("Waiting for game to load...")
    time.sleep(5)
    
    # Game loop parameters
    # Number of actions to take
    action_duration = 0.15  # How long to hold each button
    step_delay = 0.3  # Delay between actions
    
    print("\n" + "="*60)
    print("STARTING GAME LOOP")
    print("="*60 + "\n")
    step = -1
    try:
        while 1:
            step += 1
            print(f"\n--- Step {step + 1} ---")
            
            # 1. Capture current game state
            screenshot = emulator.get_screenshot()
            
            # Optional: Save screenshots periodically
            if step % 50 == 0:
                screenshot.save(f'screenshots/step_{step:04d}.png')
                print(f"Screenshot saved: step_{step:04d}.png")
            
            # 2. Get next action from AI
            # In the game loop
            sentence, button, probs, diversity_reward = control_collection.get_next_action(
                screenshot, 
                temperature=1.0,
                train_rl=True  # Enable RL training
            )

            print(f"Diversity Reward: {diversity_reward:.4f}")

            # Save periodically
            if step % 500 == 0:
                control_collection.save_projection(f'projection_step_latest.pt')
            
            # 3. Display action info
            print(f"Action: {sentence}")
            print(f"Button: {button}")
            print(f"Confidence: {np.max(probs):.3f}")
            
            # 4. Execute action on emulator
            emulator.send_action(button, duration=action_duration)
            
            # 5. Wait before next action
            time.sleep(step_delay)
            
            # Optional: Display prior distribution every 100 steps
            if (step + 1) % 100 == 0:
                print("\n--- Prior Distribution Update ---")
                top_5_indices = np.argsort(control_collection.prior_counts)[-5:][::-1]
                print("Most frequently selected actions:")
                for idx in top_5_indices:
                    sentence = control_collection.idx2sentence[idx]
                    count = control_collection.prior_counts[idx]
                    print(f"  {sentence}: {count:.0f} times")
                print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    finally:
        # Cleanup
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)
        
        # Save final statistics
        print("\nFinal Action Statistics:")
        print(f"Total steps taken: {step + 1}")
        print(f"\nTop 10 most frequent actions:")
        top_10_indices = np.argsort(control_collection.prior_counts)[-10:][::-1]
        for i, idx in enumerate(top_10_indices, 1):
            sentence = control_collection.idx2sentence[idx]
            count = control_collection.prior_counts[idx]
            button = control_collection.sentence2control[sentence]
            print(f"{i:2d}. [{button:6s}] {sentence}: {count:.0f} times")
        
        emulator.close()
        print("\nEmulator closed. Goodbye!")


if __name__ == "__main__":
    # Create screenshots directory if it doesn't exist
    import os
    os.makedirs('screenshots', exist_ok=True)
    
    main()