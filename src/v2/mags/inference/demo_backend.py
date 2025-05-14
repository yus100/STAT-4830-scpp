class KnowledgeGraph:
    """
    Manages the knowledge graph, including nodes, edges, and interactions.
    """
    DEFAULT_HYPERPARAMS = {
        "decay_rate": 0.99,
        "strengthen_rate": 1.03,
        "max_nodes": 100,
        "max_edges": 250,
        "max_outdegree": 10,
        "forget_threshold": 0.20,
        "permanence_threshold": 3.00
    }

    def __init__(self, openai_caller: GnicLLMWrapper, initial_hyperparams: Optional[Dict[str, float]] = None):
        self.openai_caller = openai_caller
        self.nodes: List[Dict[str, Any]] = [] # List of node dicts: {'id': str, 'label': str, 'weight': float, 'size': float}
        self.edges: List[Dict[str, Any]] = [] # List of edge dicts: {'id': str, 'from': str, 'to': str, 'label': str, 'weight': float, 'width': float}
        self.world_triplets: List[List[str]] = []
        self._edge_id_counter: int = 0
        
        self.hyperparams = self.DEFAULT_HYPERPARAMS.copy()
        if initial_hyperparams:
            self.hyperparams.update(initial_hyperparams)

        self._initialize_graph()

    def update_hyperparams(self, new_hyperparams: Dict[str, float]):
        """Updates the graph's hyperparameters."""
        self.hyperparams.update(new_hyperparams)
        print(f"Hyperparameters updated: {self.hyperparams}")

    def _get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None

    def _update_node(self, node_id: str, updates: Dict[str, Any]):
        for node in self.nodes:
            if node['id'] == node_id:
                node.update(updates)
                return
        # If node not found, this implies an issue or it was removed
        # print(f"Warning: Attempted to update non-existent node {node_id}")


    def _remove_node(self, node_id: str):
        self.nodes = [n for n in self.nodes if n['id'] != node_id]
        # Also remove connected edges
        self.edges = [e for e in self.edges if e['from'] != node_id and e['to'] != node_id]


    def _get_edges_connected_to_node(self, node_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.edges if e['from'] == node_id or e['to'] == node_id]

    def _get_outgoing_edges_from_node(self, node_id: str) -> List[Dict[str, Any]]:
         return [e for e in self.edges if e['from'] == node_id]

    def _initialize_graph(self):
        """Clears graph and adds a default 'player' node."""
        self.nodes.clear()
        self.edges.clear()
        self._edge_id_counter = 0
        # Add player node
        player_node = {
            'id': 'player',
            'label': 'Player',
            'weight': 1.0, # Initial weight
            'size': 16.0   # Default size
        }
        self.nodes.append(player_node)
        print("Graph initialized with player node.")
        self._print_graph_info()

    def _print_graph_info(self):
        """Prints current node and edge counts."""
        print(f"Graph Info: Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")

    def update_graph_visualization_from_triplets(self, triplets: List[List[str]]):
        """
        Builds/rebuilds the graph's nodes and edges from triplets.
        Weights are reset for new entities/relations.
        """
        if not triplets and not self.world_triplets: # if no triplets and graph is empty (except player)
            self._initialize_graph() # ensure player node exists if triplets are empty initially
            self._print_graph_info()
            return

        new_nodes_set = set()
        if any(n['id'] == 'player' for n in self.nodes): # Preserve player node if it exists
            new_nodes_set.add('player')

        new_edges_list = []

        for s, r, o in triplets:
            new_nodes_set.add(s)
            new_nodes_set.add(o)
            new_edges_list.append({'from': s, 'to': o, 'label': r, 'weight': 1.0, 'width': 2.0})

        # Clear existing nodes (except player if present) and edges
        player_node_data = self._get_node('player')
        self.nodes.clear()
        self.edges.clear()
        self._edge_id_counter = 0
        
        if player_node_data and 'player' in new_nodes_set:
             self.nodes.append(player_node_data) # re-add player if it was part of the graph concept
        elif player_node_data and not new_nodes_set : #if triplets were empty, player should persist.
             self.nodes.append(player_node_data)
        elif 'player' not in new_nodes_set and player_node_data: # If player was not in triplets but existed
            # This scenario implies the 'player' entity might not be relevant to the new triplets
            # For now, let's keep it if it existed, it can be pruned later by decay if not used
            self.nodes.append(player_node_data)

        for node_id in new_nodes_set:
            if not self._get_node(node_id): # Add only if not already added (e.g. player)
                self.nodes.append({'id': node_id, 'label': node_id, 'weight': 1.0, 'size': 16.0})
        
        for edge_data in new_edges_list:
            edge_data['id'] = f'e{self._edge_id_counter}'
            self.edges.append(edge_data)
            self._edge_id_counter += 1
        
        print("Graph visualization updated from triplets.")
        self._print_graph_info()


    def apply_decay_and_strengthen(self, relevant_text_for_strengthening: str):
        """
        Applies decay to all nodes/edges, strengthens relevant ones, and prunes the graph.
        """
        decay_rate = self.hyperparams['decay_rate']
        strengthen_rate = self.hyperparams['strengthen_rate']
        max_nodes_limit = self.hyperparams['max_nodes']
        max_edges_limit = self.hyperparams['max_edges']
        max_outdegree = self.hyperparams['max_outdegree']
        forget_threshold = self.hyperparams['forget_threshold']
        permanence_threshold = self.hyperparams['permanence_threshold']

        # 1. Decay
        for node in self.nodes:
            if node['id'] != 'player': # Player node might have special decay rules or no decay
                 node['weight'] *= decay_rate
        for edge in self.edges:
            edge['weight'] *= decay_rate
        
        print("Applied decay.")

        # 2. Strengthen used nodes and their edges
        used_node_ids = [node['id'] for node in self.nodes if node['id'].lower() in relevant_text_for_strengthening.lower()]
        
        for node_id in used_node_ids:
            node = self._get_node(node_id)
            if node:
                node['weight'] *= strengthen_rate
        
        for edge in self.edges:
            if edge['from'] in used_node_ids or edge['to'] in used_node_ids:
                edge['weight'] *= strengthen_rate
        
        print(f"Strengthened nodes: {used_node_ids} and their edges.")

        # 3. Remove nodes below forget_threshold (if not permanent)
        # Collect IDs to remove to avoid modification issues during iteration
        nodes_to_remove = []
        for node in self.nodes:
            if node['id'] == 'player': continue # Don't remove player this way
            if node['weight'] < forget_threshold and node['weight'] < permanence_threshold: # Original logic: weight <= perm
                nodes_to_remove.append(node['id'])
        
        for node_id in nodes_to_remove:
            self._remove_node(node_id) # This also removes connected edges
        if nodes_to_remove:
            print(f"Removed {len(nodes_to_remove)} nodes below forget threshold: {nodes_to_remove}")


        # 4. Prune to max_nodes
        # Sort eligible nodes (not permanent, by weight ascending) for removal
        if len(self.nodes) > max_nodes_limit:
            eligible_nodes_for_pruning = [n for n in self.nodes if n['weight'] < permanence_threshold and n['id'] != 'player']
            eligible_nodes_for_pruning.sort(key=lambda x: x['weight'])
            
            num_to_prune = len(self.nodes) - max_nodes_limit
            nodes_pruned_by_max = []
            for i in range(min(num_to_prune, len(eligible_nodes_for_pruning))):
                node_to_remove = eligible_nodes_for_pruning[i]
                nodes_pruned_by_max.append(node_to_remove['id'])
                self._remove_node(node_to_remove['id'])
            if nodes_pruned_by_max:
                print(f"Pruned {len(nodes_pruned_by_max)} nodes to meet max_nodes limit: {nodes_pruned_by_max}")


        # 5. Prune to max_edges
        # Sort eligible edges (not permanent, by weight ascending) for removal
        if len(self.edges) > max_edges_limit:
            eligible_edges_for_pruning = [e for e in self.edges if e['weight'] < permanence_threshold]
            eligible_edges_for_pruning.sort(key=lambda x: x['weight'])

            num_to_prune = len(self.edges) - max_edges_limit
            edges_pruned_by_max = []
            for i in range(min(num_to_prune, len(eligible_edges_for_pruning))):
                edge_to_remove = eligible_edges_for_pruning[i]
                edges_pruned_by_max.append(edge_to_remove['id'])
                self.edges = [e for e in self.edges if e['id'] != edge_to_remove['id']]
            if edges_pruned_by_max:
                print(f"Pruned {len(edges_pruned_by_max)} edges to meet max_edges limit.")


        # 6. Prune outdegree
        for node in list(self.nodes): # Iterate over a copy if nodes can be removed indirectly
            outgoing_edges = self._get_outgoing_edges_from_node(node['id'])
            if len(outgoing_edges) > max_outdegree:
                eligible_edges_for_outdegree_pruning = [e for e in outgoing_edges if e['weight'] < permanence_threshold]
                eligible_edges_for_outdegree_pruning.sort(key=lambda x: x['weight'])
                
                num_to_prune = len(outgoing_edges) - max_outdegree
                edges_pruned_by_outdegree = []
                for i in range(min(num_to_prune, len(eligible_edges_for_outdegree_pruning))):
                    edge_to_remove = eligible_edges_for_outdegree_pruning[i]
                    edges_pruned_by_outdegree.append(edge_to_remove['id'])
                    self.edges = [e for e in self.edges if e['id'] != edge_to_remove['id']]
                if edges_pruned_by_outdegree:
                    print(f"Pruned {len(edges_pruned_by_outdegree)} outgoing edges for node '{node['id']}' to meet max_outdegree.")


        # 7. Update visual attributes (size for nodes, width for edges based on weight)
        # These are stored in case the graph data is exported or used by another visualizer
        for node in self.nodes:
            node['size'] = max(10.0, 16.0 * node['weight'])
        for edge in self.edges:
            edge['width'] = max(1.0, 2.0 * edge['weight'])
        
        print("Applied decay, strengthening, and pruning.")
        self._print_graph_info()


    def load_world_from_text(self, game_text: str) -> str:
        """
        Processes initial game text to build the world model.
        """
        if not game_text.strip():
            return "Error: Game text cannot be empty."
        
        print("Generating world model from text...")
        try:
            initial_triplets = self.openai_caller.extract_triplets(game_text)
            if not initial_triplets:
                # If extraction yields nothing, but we need to reconcile with an empty existing set.
                # This typically means the LLM found nothing, or there was a parsing error handled in extract_triplets.
                print("LLM could not extract any initial triplets from the provided text.")
                # We still call reconcile to ensure the worldTriplets list is correctly initialized (empty in this case)
                # and graph visualization is updated (cleared or showing only player).
                self.world_triplets = self.openai_caller.reconcile_triplets([], [])
            else:
                print(f"Extracted {len(initial_triplets)} initial triplets.")
                # For initial load, existing_triplets is empty
                self.world_triplets = self.openai_caller.reconcile_triplets(initial_triplets, [])
                print(f"Reconciled into {len(self.world_triplets)} world triplets.")

            self.update_graph_visualization_from_triplets(self.world_triplets)
            return f"World model generated with {len(self.nodes)} entities and {len(self.edges)} relationships."
        except Exception as e:
            print(f"Error generating world model: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating world model: {str(e)}"

    def reset_world(self) -> str:
        """Resets the graph to its initial state."""
        self.world_triplets = []
        self._initialize_graph() # This clears nodes/edges and adds player
        return "World reset. Graph cleared and player node initialized."

    def handle_user_input(self, user_text: str) -> str:
        """
        Processes user questions or statements to query or update the graph.
        """
        if not user_text.strip():
            return "Error: Input cannot be empty."

        user_text = user_text.strip()
        is_question = user_text.endswith('?') or \
                      any(user_text.lower().startswith(kw) for kw in ["what", "where", "who", "is ", "are ", "does", "do ", "can ", "list "])
        
        try:
            if is_question:
                print(f"Answering question: {user_text}")
                answer = self.openai_caller.answer_question(user_text, self.world_triplets)
                print(f"LLM Answer: {answer}")
                self.apply_decay_and_strengthen(user_text) # Pass question to inform strengthening
                return answer
            else:
                print(f"Updating world model with statement: {user_text}")
                new_triplets = self.openai_caller.extract_triplets(user_text)
                if new_triplets:
                    print(f"Extracted {len(new_triplets)} new triplets from statement.")
                    self.world_triplets = self.openai_caller.reconcile_triplets(new_triplets, self.world_triplets)
                    print(f"Reconciled into {len(self.world_triplets)} total world triplets.")
                    self.update_graph_visualization_from_triplets(self.world_triplets)
                    # After updating graph from new statement, apply decay/strengthen based on the statement
                    self.apply_decay_and_strengthen(user_text)
                    return "World model updated based on your statement."
                else:
                    print("Couldn't extract new information from the statement.")
                    # Even if no new triplets, the interaction itself might cause decay/strengthening
                    self.apply_decay_and_strengthen(user_text)
                    return "Couldn't extract new information from that statement, but the world has aged."
        except Exception as e:
            print(f"Error processing user input: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing input: {str(e)}"

    def get_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns the current nodes and edges."""
        return {
            "nodes": [node.copy() for node in self.nodes], # Return copies to prevent external modification
            "edges": [edge.copy() for edge in self.edges]
        }

    def get_world_triplets(self) -> List[List[str]]:
        """Returns the current list of world triplets."""
        return [triplet[:] for triplet in self.world_triplets]


if __name__ == '__main__':
    # This is an example of how to use the classes.
    # IMPORTANT: Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key.
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" 

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("Please replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key to run the example.")
    else:
        try:
            print("Initializing GnicLLMWrapper and KnowledgeGraph...")
            ai_caller = GnicLLMWrapper(api_key=OPENAI_API_KEY)
            graph = KnowledgeGraph(openai_caller=ai_caller)
            print("Initialization complete.")

            # --- Test 1: Load initial world model ---
            print("\n--- Test 1: Load initial world model ---")
            initial_text = (
                "The Red Keep is a castle in King's Landing. "
                "King's Landing is the capital of the Seven Kingdoms. "
                "The Iron Throne is located in the Red Keep. "
                "Daenerys Targaryen has a dragon named Drogon. "
                "Jon Snow is a Targaryen."
            )
            load_status = graph.load_world_from_text(initial_text)
            print(f"Load Status: {load_status}")
            # print("Current Graph Data:", json.dumps(graph.get_graph_data(), indent=2))
            # print("Current World Triplets:", json.dumps(graph.get_world_triplets(), indent=2))

            # --- Test 2: Ask a question ---
            print("\n--- Test 2: Ask a question ---")
            question1 = "Where is the Iron Throne located?"
            answer1 = graph.handle_user_input(question1)
            print(f"Q: {question1}\nA: {answer1}")
            # print("Graph Data after question:", json.dumps(graph.get_graph_data(), indent=2))

            # --- Test 3: Make a statement to update the model ---
            print("\n--- Test 3: Make a statement ---")
            statement1 = "Jon Snow is the King in the North. Tyrion Lannister is Hand of the Queen to Daenerys."
            update_status1 = graph.handle_user_input(statement1)
            print(f"Update Status: {update_status1}")
            # print("Graph Data after statement:", json.dumps(graph.get_graph_data(), indent=2))
            # print("Updated World Triplets:", json.dumps(graph.get_world_triplets(), indent=2))

            # --- Test 4: Ask another question based on new and old info ---
            print("\n--- Test 4: Ask another question ---")
            question2 = "Who is the Hand of the Queen?"
            answer2 = graph.handle_user_input(question2)
            print(f"Q: {question2}\nA: {answer2}")

            question3 = "What is the relationship between Jon Snow and Daenerys Targaryen?" # May require inference not directly in triplets
            answer3 = graph.handle_user_input(question3)
            print(f"Q: {question3}\nA: {answer3}")

            # --- Test 5: Hyperparameter change and decay/strengthen demonstration ---
            print("\n--- Test 5: Hyperparameter and Decay/Strengthen Demo ---")
            graph.update_hyperparams({"decay_rate": 0.5, "forget_threshold": 0.1})
            # Simulate some interactions to see weights change
            graph.handle_user_input("Is King's Landing a city?") # Strengthens "King's Landing"
            graph.handle_user_input("Tell me about dragons.") # Strengthens "dragon" if it exists
            
            # Manually call apply_decay_and_strengthen with a neutral phrase if not part of handle_user_input for testing
            # This is usually called within handle_user_input.
            # graph.apply_decay_and_strengthen("some neutral text") 
            print("Graph Data after further interactions and decay:", json.dumps(graph.get_graph_data(), indent=2))

            # --- Test 6: Reset world ---
            print("\n--- Test 6: Reset world ---")
            reset_status = graph.reset_world()
            print(reset_status)
            print("Graph Data after reset:", json.dumps(graph.get_graph_data(), indent=2))
            print("World Triplets after reset:", json.dumps(graph.get_world_triplets(), indent=2))

            # --- Test 7: Empty initial load ---
            print("\n--- Test 7: Load with empty initial text ---")
            load_status_empty = graph.load_world_from_text(" ")
            print(f"Load Status (empty): {load_status_empty}")
            print("Graph Data after empty load:", json.dumps(graph.get_graph_data(), indent=2))
            print("World Triplets after empty load:", json.dumps(graph.get_world_triplets(), indent=2))


        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except ConnectionError as ce:
            print(f"API Connection Error: {ce}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()