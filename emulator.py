from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import time

class EmulatorController:
    def __init__(self, emulator_url='https://gbapokemon.com/play/pokemon-leaf-green/'):
        """
        Initialize the emulator controller with Selenium.
        
        Parameters
        ----------
        emulator_url : str
            URL of the GBA emulator
        """
        # Control mapping
        self.control_mapping = {
            "A": "z",
            "B": "x",
            "select": Keys.RETURN,
            "^": Keys.ARROW_UP,
            "v": Keys.ARROW_DOWN,
            ">": Keys.ARROW_RIGHT,
            "<": Keys.ARROW_LEFT
        }
        
        # Initialize browser
        options = webdriver.ChromeOptions()
        # Block pop-ups and new tabs
        options.add_argument('--disable-popup-blocking')
        options.add_experimental_option('prefs', {
            'profile.default_content_setting_values.popups': 2,
            'profile.default_content_setting_values.notifications': 2,
            'profile.block_third_party_cookies': True
        })

        # Additional ad/pop-up blocking
        options.add_argument('--disable-extensions')
        options.add_argument('--no-default-browser-check')
        # options.add_argument('--start-maximized')
        self.driver = webdriver.Chrome(options=options)
        
        # Open emulator
        self.driver.get(emulator_url)
        
        # Wait for page to load
        time.sleep(3)
        
        # Get screenshot area from user clicks
        self.screenshot_area = self._get_screenshot_area()
        
        print(f"Screenshot area configured: {self.screenshot_area}")

    def _get_screenshot_area(self):
        """
        Prompt user to click 2 corners to define screenshot area.
        Returns
        -------
        dict
            Dictionary with 'x', 'y', 'width', 'height' keys
        """
        print("\n" + "="*60)
        print("SCREENSHOT AREA SETUP")
        print("="*60)
        print("Please click on the 2 corners of the game screen:")
        print("="*60 + "\n")
        
        points = []
        
        # Get top-left corner
        input("Press Enter when ready to click TOP-LEFT corner...")
        action = ActionChains(self.driver)
        action.click().perform()
        time.sleep(0.5)
        
        # Get cursor position via JavaScript
        top_left = self.driver.execute_script("""
            return new Promise((resolve) => {
                document.addEventListener('click', function handler(e) {
                    document.removeEventListener('click', handler);
                    resolve({x: e.clientX, y: e.clientY});
                });
            });
        """)
        print(f"Top-left corner recorded: ({top_left['x']}, {top_left['y']})")
        points.append(top_left)
        
        # Get bottom-right corner
        input("\nPress Enter when ready to click BOTTOM-RIGHT corner...")
        
        bottom_right = self.driver.execute_script("""
            return new Promise((resolve) => {
                document.addEventListener('click', function handler(e) {
                    document.removeEventListener('click', handler);
                    resolve({x: e.clientX, y: e.clientY});
                });
            });
        """)
        print(f"Bottom-right corner recorded: ({bottom_right['x']}, {bottom_right['y']})")
        points.append(bottom_right)
        
        # Calculate bounding rectangle
        x = int(points[0]['x'])
        y = int(points[0]['y'])
        width = int(points[1]['x'] - points[0]['x'])
        height = int(points[1]['y'] - points[0]['y'])
        
        print(f"\nScreenshot area: x={x}, y={y}, width={width}, height={height}\n")
        input('Press enter to start!!!')
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }

    def get_screenshot(self):
        """
        Capture screenshot of the defined game area.
        
        Returns
        -------
        PIL.Image
            Screenshot of the game area
        """
        # Take full page screenshot
        screenshot_bytes = self.driver.get_screenshot_as_png()
        full_image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Crop to screenshot area
        cropped = full_image.crop((
            self.screenshot_area['x'],
            self.screenshot_area['y'],
            self.screenshot_area['x'] + self.screenshot_area['width'],
            self.screenshot_area['y'] + self.screenshot_area['height']
        ))
        
        return cropped
    
    def send_action(self, control, duration=0.1):
        """
        Send a control action to the emulator.
        
        Parameters
        ----------
        control : str
            Control button name (e.g., 'A', 'B', '^', 'v', etc.)
        duration : float
            How long to hold the key (seconds)
        """
        if control not in self.control_mapping:
            print(f"Warning: Unknown control '{control}'")
            return
        
        key = self.control_mapping[control]
        
        # Send key press to the page body
        body = self.driver.find_element(By.TAG_NAME, 'body')
        
        # Press and hold
        actions = ActionChains(self.driver)
        actions.key_down(key).perform()
        time.sleep(duration)
        actions.key_up(key).perform()
    
    def send_sequence(self, controls, duration_per_action=0.1, delay_between=0.05):
        """
        Send a sequence of control actions.
        
        Parameters
        ----------
        controls : list of str
            List of control button names
        duration_per_action : float
            How long to hold each key
        delay_between : float
            Delay between actions
        """
        for control in controls:
            self.send_action(control, duration=duration_per_action)
            time.sleep(delay_between)
    
    def close(self):
        """
        Close the browser and cleanup.
        """
        self.driver.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage:
if __name__ == "__main__":
    # Initialize controller (will prompt for corner clicks)
    controller = EmulatorController()
    
    # Wait a bit for game to load
    time.sleep(2)
    
    # Get a screenshot
    screenshot = controller.get_screenshot()
    screenshot.save('game_screenshot.png')
    print("Screenshot saved!")
    
    # Send some actions
    controller.send_action('A')  # Press A button
    time.sleep(0.5)
    controller.send_action('^')  # Press Up
    
    # Send a sequence
    controller.send_sequence(['^', '^', 'A', 'B'])
    
    # Close when done
    controller.close()