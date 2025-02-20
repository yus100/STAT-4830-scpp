import os
import importlib.util

class Loader:
    def __init__(self, game_id: str):
        """
        Initialize the loader with a game ID (e.g. 'game1')
        """
        self.game_id = game_id
        self.game_module = None
        self._load_game_data()

    def _load_game_data(self):
        """
        Load game data from the corresponding details.py file
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        game_file = os.path.join(base_path, self.game_id, 'details.py')
        
        try:
            spec = importlib.util.spec_from_file_location("details", game_file)
            self.game_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.game_module)
        except FileNotFoundError:
            raise FileNotFoundError(f"Game details file not found for {self.game_id}")
        except Exception as e:
            raise ValueError(f"Error loading game file for {self.game_id}: {str(e)}")

    def getRules(self):
        """
        Get the rules for the game
        Returns:
            str: Rules for the game
        """
        if not self.game_module:
            raise ValueError("Game data not loaded")
        return getattr(self.game_module, 'GAME_RULES', "")

    def getQuestions(self):
        """
        Get the questions for the game
        Returns:
            list: List of questions for the game
        """
        if not self.game_module:
            raise ValueError("Game data not loaded")
        return getattr(self.game_module, 'TEST_QUESTIONS', [])