import json
from ollama import AsyncClient
import asyncio

class LLM:
    async def query(self, prompt: str) -> str:
        raise NotImplementedError("Query method not implemented")
    async def query_json(self, prompt: str) -> str:
        response = await self.query(prompt)
        print(response)
        return json.loads(response)

class LLama(LLM):
    """
    A class to interact with a locally running Ollama model asynchronously.
    """
    def __init__(self, model_name="llama3.2"):
        """
        Initializes the LLM class with a specified Ollama model.

        Args:
            model_name (str): The name of the Ollama model to use.
                               Defaults to "llama3.2".
        """
        self.model_name = model_name
        self.client = AsyncClient()

    async def query(self, prompt: str, stream: bool = False) -> str:
        """
        Queries the Ollama model with a given prompt and returns the response asynchronously.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: The text response from the Ollama model.
                 Returns an error message if there's an issue querying the model.
        """

        #TODO: Clear Context

        try:
            response = await self.client.chat(model=self.model_name, messages=[
                { 'role': 'user', 'content': prompt }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying Ollama model: {e}"

    def set_model(self, model_name: str):
        """
        Changes the Ollama model to be used for subsequent queries.

        Args:
            model_name (str): The new model name to set.
        """
        self.model_name = model_name

    async def list_models(self):
        """
        Lists the locally available Ollama models asynchronously.

        Returns:
            list: A list of available Ollama model names.
                  Returns an error message if models cannot be listed.
        """
        try:
            models = await self.client.list()
            return [model['model'] for model in models['models']]
        except Exception as e:
            return f"Error listing Ollama models: {e}"

if __name__ == '__main__':
    async def main():
        # Example usage
        llm_instance = LLM(model_name="llama3.2") # You can change the model here

        # List available models
        available_models = await llm_instance.list_models()
        if isinstance(available_models, list):
            print("Available Ollama models:")
            for model in available_models:
                print(f"- {model}")
        else:
            print(available_models) # Print error message if listing failed

        prompt_text = "What is the capital of France?"
        response_text = await llm_instance.query(prompt_text)

        if isinstance(response_text, str) and response_text.startswith("Error"):
            print(response_text) # Print error message if query failed
        else:
            print(f"Prompt: {prompt_text}")
            print(f"Response: {response_text}")

    asyncio.run(main())