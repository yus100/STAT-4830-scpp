import json
import time
from typing import List, Dict, Any, Tuple, Optional
import requests # Using synchronous requests for simplicity, can be swapped with httpx for async

class GnicLLMWrapper:
    """
    A class to interact with the OpenAI API for knowledge graph operations.
    """
    def __init__(self, api_key: str, api_url: str = "https://api.openai.com/v1/chat/completions", model: str = "gpt-4o"):
        if not api_key or api_key.startswith("sk-YOUR"):
            raise ValueError("OpenAI API Key not configured properly.")
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _call_openai(self, messages: List[Dict[str, str]], temperature: float = 0.5, response_format_json: bool = False) -> Dict[str, Any]:
        """Helper function to make calls to OpenAI API."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        retry_attempts = 3
        wait_time = 5  # seconds

        for attempt in range(retry_attempts):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120) # Increased timeout
                response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                data = response.json()
                if not data.get("choices") or not data["choices"][0].get("message") or data["choices"][0]["message"].get("content") is None:
                    raise ValueError("Invalid response structure from OpenAI API.")
                return data
            except requests.exceptions.RequestException as e:
                print(f"OpenAI API call failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(wait_time * (attempt + 1)) # Exponential backoff
                else:
                    raise ConnectionError(f"Failed to connect to OpenAI API after {retry_attempts} attempts: {e}") from e
            except ValueError as e:
                print(f"OpenAI API response error: {e}")
                raise

    def call_simple(self, query: str, temperature: float = 0.5) -> str:
        messages = [
                {"role": "user", "content": query}
        ]

        response = self._call_openai(messages)
        return response["choices"][0]["message"]["content"]

    def extract_asks(self, text_str) -> List[List[str]]:

        system_prompt = """
        You are an expert question asker. Extract knowledgeable questions to ask from the query under the key 'questions'. Keep your questions simple and basic, but you can include as many as neccessary. 
        If there is no simple way to ask a question, just say 'information about [item]'
        Ensure the output is a valid JSON object containing only this key, and its value is a list of strings.
        OUTPUT NOTHING ELSE, not even triple qoutes around the JSON, as your response will be parsed by a downstream agent and this will break this system.

        For example:

        Question: What color is Joe's vehicle
        Response: {
            "questions": [
                "Who is Joe",
                "What vehicle does Joe have",
                "What color is Joe's Vehicle"
            ]
        }

        Question: There is a blue toliet in the restroom next to the kitchen. Where is the painting? New Jersey is good state. The door is blue.
        Response: {
            "questions": [
                "Information about blue toliet",
                "Information about Restroom",
                "Information about Kitchen",
                "Information about Painting",
                "Where is Painting",
                "Information about New Jersey",
                "Information about Door",
            ]
        }


        """

        user_prompt = f"{text_str}"

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            response_data = self._call_openai(messages, temperature=0.3, response_format_json=True) # Lower temp for structured output
            content_str = response_data["choices"][0]["message"]["content"]
            parsed_content = json.loads(content_str)
            questions = parsed_content.get("questions")
            return questions if questions else []
        except:
            return []


    def extract_triplets(self, text: str) -> List[List[str]]:
        """
        Extracts factual triplets from text using OpenAI.
        Expected format: [subject, relation, object]
        """
        system_prompt = "You are an expert knowledge graph extractor. Extract factual triplets in the form [subject, relation, object] under the key 'triplets'. Ensure the output is a valid JSON object containing only this key, and its value is a list of lists (triplets)."
        user_prompt = f"Extract triplets from:\n\n{text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response_data = self._call_openai(messages, temperature=0.3, response_format_json=True) # Lower temp for structured output
            content_str = response_data["choices"][0]["message"]["content"]
            parsed_content = json.loads(content_str)

            triplets = parsed_content.get("triplets")
            if triplets is None: # Fallback if 'triplets' key is missing directly
                # Try to find a key that contains a list of lists
                for key, value in parsed_content.items():
                    if isinstance(value, list) and all(isinstance(item, list) for item in value):
                        triplets = value
                        break
            
            if not isinstance(triplets, list) or not all(isinstance(t, list) and len(t) == 3 and all(isinstance(s, str) for s in t) for t in triplets):
                print(f"Warning: LLM extraction did not return a clean list of triplets. Received: {triplets}. Attempting to salvage.")
                # Attempt to salvage if the structure is slightly off but contains lists of 3 strings
                salvaged_triplets = []
                if isinstance(triplets, list):
                    for item in triplets:
                        if isinstance(item, list) and len(item) == 3 and all(isinstance(s, str) for s in item):
                            salvaged_triplets.append(item)
                if not salvaged_triplets and triplets is not None: # If salvage failed but we got *something*
                     print(f"Could not parse triplets from: {content_str}")
                     return [] # Return empty if parsing fails to meet criteria
                triplets = salvaged_triplets

            return triplets if triplets else []

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in extract_triplets: {e}. Response: {response_data['choices'][0]['message']['content']}")
            return []
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing triplets response: {e}. Response: {response_data['choices'][0]['message']['content']}")
            return []


    def reconcile_triplets(self, new_triplets: List[List[str]], existing_triplets: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Merges new triplets with existing ones, removing duplicates and ensuring consistency using OpenAI.
        """
        if not new_triplets:
            return existing_triplets

        system_prompt = (
            """You are a knowledge graph reconciliation engine. Your task is to merge two lists of triplets (subject, relation, object). 
            The goal is to create a single, consistent list of unique triplets. Prioritize information from 'New Triplets' if there are direct contradictions, 
            but generally aim for a comprehensive union.
            
            Try to match the format of the existing nodes AS MUCH AS POSSIBLE. If neccessary, match new information to the existing infromation.
            Only output new tripets (id as 'new'), or changed triplets (use same id), do not re-output unchanged triplets
            
             Ensure the final output is a valid JSON object with a single key 'result' 
            containing the list of changed triplets. Each triplet must be a list of three strings.
            Output your result as follows (DO NOT INCLUDE THE TRIPLE JSON BACKTICKS)
            
            {
                "result": [
                {
                    id: 'new' or id
                    triplet: list[3]
                }, 
                {
                    id: 'new' or id
                    triplet: list[3]
                }, 
                ...
                ]
            }


            """
        )
        user_prompt = f"Existing Triplets:\n{json.dumps(existing_triplets)}\n\nNew Triplets:\n{json.dumps(new_triplets)}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response_data = self._call_openai(messages, temperature=0.2, response_format_json=True) # Low temp for factual merging
            content_str = response_data["choices"][0]["message"]["content"]
            parsed_content = json.loads(content_str)
            
            result = parsed_content.get("result")
            if result is None: # Fallback
                for key, value in parsed_content.items():
                     if isinstance(value, list) and all(isinstance(item, list) for item in value):
                        result = value
                        break

            # print(result)
            return result

            # if not isinstance(result, list) or not all(isinstance(t, list) and len(t) == 3 and all(isinstance(s, str) for s in t) for t in result):
                # print(f"Warning: LLM reconciliation did not return a clean list of triplets. Received: {result}. Using simple union instead.")
                # Fallback to simple union if LLM fails to produce valid output
                # combined_triplets = existing_triplets + new_triplets
                # unique_triplets_set = set()
                # unique_triplets_list = []
                # for t in combined_triplets:
                #     triplet_tuple = tuple(t)
                #     if triplet_tuple not in unique_triplets_set:
                #         unique_triplets_set.add(triplet_tuple)
                #         unique_triplets_list.append(list(triplet_tuple))
                # return unique_triplets_list

            # return result if result else []

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in reconcile_triplets: {e}. Response: {response_data['choices'][0]['message']['content']}")
            return existing_triplets # Fallback
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing reconciliation response: {e}. Response: {response_data['choices'][0]['message']['content']}")
            return existing_triplets # Fallback

    def answer_question(self, question: str, current_triplets: List[List[str]]) -> str:
        """
        Answers a question based strictly on the provided triplets using OpenAI.
        """
        system_prompt = (
            "You are an assistant that answers questions strictly based on the provided knowledge triplets. "
            "If the information is not in the triplets, state that the information is not available in the current knowledge base. "
            "Do not infer or use external knowledge. Your answer should be concise."
        )
        user_prompt = f"World Knowledge (Triplets):\n{json.dumps(current_triplets, indent=2)}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response_data = self._call_openai(messages, temperature=0.7, response_format_json=False)
            return response_data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing question answering response: {e}")
            return "I encountered an error trying to answer the question."



if __name__ == "__main__":
    model = GnicLLMWrapper(api_key="")

    messages = [
            {"role": "user", "content": "WASSUP"}
    ]

    response = model._call_openai(messages)
    print(response["choices"][0]["message"]["content"])
