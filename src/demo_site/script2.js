document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const gameTextInput = document.getElementById('game-text-input');
    const loadButton = document.getElementById('load-button');
    const resetButton = document.getElementById('reset-button');
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const graphContainer = document.getElementById('graph-container');
    const nodeCountSpan = document.getElementById('node-count');
    const edgeCountSpan = document.getElementById('edge-count');

    // --- Graph State ---
    let nodes = new vis.DataSet([]);
    let edges = new vis.DataSet([]);
    let network = null;
    let worldTriplets = []; // Store the canonical triplets representing the world

    // --- Initialize Graph ---
    function initializeGraph() {
        const data = { nodes: nodes, edges: edges };
        const options = {
            nodes: {
                shape: 'dot',
                size: 16,
                font: {
                    size: 12,
                    color: '#333'
                },
                borderWidth: 2,
                 color: {
                    border: '#2B7CE9',
                    background: '#D2E5FF',
                    highlight: {
                        border: '#2B7CE9',
                        background: '#FFD569'
                    },
                 }
            },
            edges: {
                width: 2,
                font: {
                    size: 10,
                    align: 'middle',
                     background: 'rgba(255,255,255,0.7)',
                     color: '#666'
                },
                arrows: {
                    to: { enabled: true, scaleFactor: 0.7 }
                },
                 color: {
                     color: '#848484',
                     highlight: '#0078d4'
                 }
            },
            physics: {
                forceAtlas2Based: {
                    gravitationalConstant: -26,
                    centralGravity: 0.005,
                    springLength: 230,
                    springConstant: 0.18,
                    avoidOverlap: 1.5 // Helps spread out nodes
                },
                maxVelocity: 146,
                solver: 'forceAtlas2Based', // Good for spreading out
                timestep: 0.35,
                stabilization: { iterations: 150 } // Stabilize faster
            },
             interaction: {
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
                hover: true
            }
        };
        network = new vis.Network(graphContainer, data, options);
        updateGraphInfo();

        nodes.add([{
            id: 'player',
            label: 'Player',
            title: 'Entity: Player',
        }])
    }

    // --- Add Message to Chat ---
    function addMessageToChat(message, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', type + '-message');
        messageElement.textContent = message;
        chatHistory.appendChild(messageElement);
        // Scroll to the bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

     // --- Update Graph Info ---
    function updateGraphInfo() {
        nodeCountSpan.textContent = nodes.length;
        edgeCountSpan.textContent = edges.length;
    }


    // --- Update Vis.js Graph from Triplets ---
    function updateGraphVisualization(triplets) {
        const uniqueNodes = new Set();
        const newEdges = [];

        triplets.forEach(([subject, relation, object]) => {
            uniqueNodes.add(subject);
            uniqueNodes.add(object);
            newEdges.push({ from: subject, to: object, label: relation });
        });

        const newNodes = Array.from(uniqueNodes).map(node => ({
            id: node,
            label: node,
            title: `Entity: ${node}` // Tooltip on hover
        }));

        // Clear existing data and add new data
        // nodes.clear();
        // edges.clear();
        // nodes.add(newNodes);
        // edges.add(newEdges);

        // // Optional: Stabilize the network after update if needed
        // // network.stabilize();
        // updateGraphInfo();
        //  // Let physics run for a bit to rearrange
        // setTimeout(() => network.stopSimulation(), 2000);

        nodes.clear();
        edges.clear();
        nodes.add(newNodes);
        edges.add(newEdges);
    
        // make the network re-fit & redraw itself into the (now correctly-sized) container
        network.fit();
        network.redraw();
    
        updateGraphInfo();
    }

    // --- Reset World ---
    function resetWorld() {
        worldTriplets = [];
        nodes.clear();
        edges.clear();
        updateGraphInfo();
        // Clear chat or add reset message
        chatHistory.innerHTML = '';
        addMessageToChat('World reset.', 'system-message');
        gameTextInput.value = '';
        userInput.value = '';
        console.log("World reset.");
    }


    // --- LLM & Graph Algorithm PLACEHOLDERS ---
    // You MUST replace these with actual logic / API calls

  // --- OpenAI Configuration ---
// ⚠️ DANGER ZONE: Replace with your key for LOCAL testing ONLY.
// ⚠️ DO NOT DEPLOY THIS KEY IN FRONTEND CODE. Use a backend proxy instead.
const OPENAI_API_KEY = ""; // <--- !!! REPLACE OR REMOVE !!!

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_MODEL = "gpt-4o"; // Use the desired model

// Helper function for making API calls
async function callOpenAI(messages) {
    if (!OPENAI_API_KEY || OPENAI_API_KEY === "sk-YOUR_OPENAI_API_KEY_HERE") {
        throw new Error("OpenAI API Key not configured. Please set it in script.js (for local testing ONLY).");
    }

    console.log("Calling OpenAI with messages:", messages);

    try {
        const response = await fetch(OPENAI_API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${OPENAI_API_KEY}`
            },
            body: JSON.stringify({
                model: OPENAI_MODEL,
                messages: messages,
                // Adjust parameters as needed
                // max_tokens: 150,
                temperature: 0.5, // Lower temperature for more deterministic triplet extraction
                response_format: { type: "json_object" } // Request JSON output where appropriate
            })
        });

        const responseData = await response.json();

        if (!response.ok) {
            console.error("OpenAI API Error Response:", responseData);
            const errorMsg = responseData.error?.message || `HTTP error! status: ${response.status}`;
            throw new Error(`OpenAI API Error: ${errorMsg}`);
        }

        console.log("OpenAI API Success Response:", responseData);

        if (!responseData.choices || responseData.choices.length === 0 || !responseData.choices[0].message?.content) {
             throw new Error("Invalid response structure from OpenAI API.");
        }

        return responseData.choices[0].message.content;

    } catch (error) {
        console.error("Error calling OpenAI API:", error);
        // Re-throw the error so the calling function can handle it and show a UI message
        throw error;
    }
}


// --- Updated LLM Functions ---

async function llm_extractTriplets(text) {
    console.log("Calling OpenAI to extract triplets for:", text.substring(0, 80) + "...");

    const systemPrompt = `You are an expert knowledge graph extractor. Analyze the provided text and extract factual relationships in the form of simple triplets: [subject, relation, object].
Guidelines:
- Output ONLY a valid JSON array containing the extracted triplets. Example: [["kitchen", "is west of", "living room"], ["sofa", "is in", "living room"], ["apple", "is", "red"]].
- Focus on concrete facts: locations, containment, properties, states.
- Subjects and objects should generally be nouns or noun phrases.
- Relations should describe the connection (e.g., 'is in', 'contains', 'is west of', 'has state', 'is color').
- Break down complex sentences into multiple simple triplets if necessary.
- Do NOT include triplets describing the player's current location (e.g., ["player", "is in", "kitchen"] or ["you", "are in", "..."]).
- Do NOT include triplets about the act of observation itself (e.g., ["you", "see", "table"]).
- If no meaningful triplets can be extracted, return an empty JSON array: [].
- The array should be under the key triplets.
- Ensure the output is strictly JSON. Do not add any introductory text or explanations.`;

    const userPrompt = `Extract triplets from the following text:\n\n${text}`;

    const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
    ];

    try {
        const responseContent = await callOpenAI(messages);
        console.log("Raw response for triplets:", responseContent);

        // Attempt to parse the JSON response
        let parsedJson = JSON.parse(responseContent);
        if (parsedJson["triplets"]) {
            parsedJson = parsedJson["triplets"];
        }

         // Validate the structure (basic check: is it an array?)
        if (!Array.isArray(parsedJson)) {
             console.error("LLM response for triplets was not a JSON array:", parsedJson);
             throw new Error("Triplet extraction failed: LLM did not return a valid JSON array.");
        }
         // Optional: Further validation to check if elements are arrays of 3 strings
        parsedJson.forEach((item, index) => {
            if (!Array.isArray(item) || item.length !== 3 || item.some(el => typeof el !== 'string')) {
                console.warn(`Triplet at index ${index} has incorrect format:`, item);
                // Decide whether to filter it out or throw an error
            }
        });


        console.log("Parsed extracted triplets:", parsedJson);
        return parsedJson; // Returns array of [subject, relation, object]

    } catch (error) {
        console.error("Error in llm_extractTriplets:", error);
        // Propagate the error to be handled by the UI logic
        throw new Error(`Triplet extraction failed: ${error.message}`);
    }
}

async function llm_reconcileTriplets(newTriplets, existingTriplets) {
    console.log("Calling OpenAI to reconcile triplets...");
    if (newTriplets.length === 0) {
        console.log("No new triplets to reconcile.");
        return existingTriplets; // Nothing to do
    }

     const systemPrompt = `You are a knowledge graph reconciliation engine. You will receive two lists of triplets: 'existing' and 'new'.
Your task is to merge these lists into a single, consistent knowledge graph state.
Rules:
1.  Identify Conflicts: A conflict occurs when a new triplet has the same subject and relation as an existing triplet, but a different object (e.g., existing: ["key", "is in", "box"], new: ["key", "is in", "inventory"]). In case of conflict, the information from the 'new' triplet is usually more current and should replace the existing conflicting triplet(s).
2.  Remove Redundancy: If a triplet from the 'new' list is an exact match to one in the 'existing' list, keep only one copy.
3.  Combine: Include all non-conflicting, non-redundant triplets from both lists in the final output.
4.  Output Format: Return ONLY a valid JSON array containing the final, reconciled list of triplets. Example: [["subj1", "rel1", "obj1"], ["subj2", "rel2", "obj2"]]. Do not add any explanatory text.
5.  If the input lists are empty, return an empty array [].
6.  The result should be under the key result
`;

    // Format triplets as JSON strings for the prompt
    const existingTripletsJson = JSON.stringify(existingTriplets);
    const newTripletsJson = JSON.stringify(newTriplets);

    const userPrompt = `Reconcile the following triplets:

Existing Triplets:
${existingTripletsJson}

New Triplets:
${newTripletsJson}

Return the final, reconciled list as a JSON array.`;

    const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
    ];

     try {
        const responseContent = await callOpenAI(messages);
        console.log("Raw response for reconciliation:", responseContent);

        let parsedJson = JSON.parse(responseContent);

        if (parsedJson["result"]) {
            parsedJson = parsedJson["result"];
        }

        if (!Array.isArray(parsedJson)) {
             console.error("LLM response for reconciliation was not a JSON array:", parsedJson);
             throw new Error("Reconciliation failed: LLM did not return a valid JSON array.");
        }
        // Optional: Further validation

        console.log("Parsed reconciled triplets:", parsedJson);
        return parsedJson; // Return the updated list of triplets

    } catch (error) {
        console.error("Error in llm_reconcileTriplets:", error);
        throw new Error(`Reconciliation failed: ${error.message}`);
    }
}

async function llm_answerQuestion(question, currentTriplets) {
    console.log("Calling OpenAI to answer question based on triplets:", question);

    const systemPrompt = `You are an assistant answering questions based *strictly* on the provided knowledge graph context (a list of triplets).
- Analyze the 'World Knowledge (Triplets)' provided below.
- Answer the user's 'Question' factually based *only* on the information contained within those triplets.
- If the information needed to answer the question is not present in the triplets, clearly state that you don't know or that the information is not available in the current world model.
- Do not make assumptions or use external knowledge.
- Keep your answer concise and directly relevant to the question based on the provided facts.`;

    // Format triplets as JSON string for context
    const tripletsJson = JSON.stringify(currentTriplets, null, 2); // Pretty print slightly for readability if needed

    const userPrompt = `World Knowledge (Triplets):
\`\`\`json
${tripletsJson}
\`\`\`

Question: ${question}

Answer based only on the provided knowledge triplets:`;

     // For Q&A, we typically want more creative/natural language, so don't force JSON output
     const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
    ];

    try {
         // Override response format for this call if it was set globally
        const requestBody = {
            model: OPENAI_MODEL,
            messages: messages,
            temperature: 0.7 // Slightly higher temp for more natural answers
            // max_tokens: 100
        };

        console.log("Calling OpenAI for Q&A...");
        const response = await fetch(OPENAI_API_URL, {
             method: "POST",
             headers: {
                 "Content-Type": "application/json",
                 "Authorization": `Bearer ${OPENAI_API_KEY}`
             },
             body: JSON.stringify(requestBody)
         });

         const responseData = await response.json();

         if (!response.ok) {
             console.error("OpenAI API Error Response (Q&A):", responseData);
             const errorMsg = responseData.error?.message || `HTTP error! status: ${response.status}`;
             throw new Error(`OpenAI API Error: ${errorMsg}`);
         }

        console.log("OpenAI API Success Response (Q&A):", responseData);

         if (!responseData.choices || responseData.choices.length === 0 || !responseData.choices[0].message?.content) {
              throw new Error("Invalid response structure from OpenAI API (Q&A).");
         }

        const answer = responseData.choices[0].message.content.trim();
        console.log("LLM answer:", answer);
        return answer; // Return the text answer string

    } catch (error) {
        console.error("Error in llm_answerQuestion:", error);
        throw new Error(`Answering question failed: ${error.message}`);
    }
}

    // --- Event Handlers ---

    // Load World Button
    loadButton.addEventListener('click', async () => {
        const text = gameTextInput.value.trim();
        if (!text) {
            addMessageToChat('Please paste some game text first.', 'error-message');
            return;
        }

        addMessageToChat('Generating world model from text...', 'system-message');
        loadButton.disabled = true;
        resetButton.disabled = true;

        try {
            const initialTriplets = await llm_extractTriplets(text);
            // On initial load, reconciliation is mainly about deduplication if the LLM repeats itself
            worldTriplets = await llm_reconcileTriplets(initialTriplets, []); // Start fresh
            updateGraphVisualization(worldTriplets);
            addMessageToChat(`World model generated with ${nodes.length} entities and ${edges.length} relationships.`, 'system-message');
            gameTextInput.value = ''; // Clear input after loading
        } catch (error) {
            console.error("Error loading world:", error);
            addMessageToChat('Error generating world model. Check console.', 'error-message');
        } finally {
             loadButton.disabled = false;
             resetButton.disabled = false;
        }
    });

    // Reset Button
    resetButton.addEventListener('click', resetWorld);

    // Send User Input (Question or Statement)
    async function handleUserInput() {
         const text = userInput.value.trim();
        if (!text) return;

        addMessageToChat(text, 'user-message');
        userInput.value = ''; // Clear input immediately
        sendButton.disabled = true; // Prevent rapid sending

        try {
            // Simple heuristic: Does it look like a question?
            const isQuestion = text.endsWith('?') || /^(what|where|who|is|are|does|do|can|list)\s/i.test(text);

            if (isQuestion) {
                addMessageToChat('Thinking...', 'system-message');
                const answer = await llm_answerQuestion(text, worldTriplets);
                addMessageToChat(answer, 'system-message');
            } else {
                // Assume it's a statement to update the world
                addMessageToChat('Updating world model...', 'system-message');
                const newTriplets = await llm_extractTriplets(text);
                if (newTriplets.length > 0) {
                    worldTriplets = await llm_reconcileTriplets(newTriplets, worldTriplets);
                    updateGraphVisualization(worldTriplets);
                    addMessageToChat(`World model updated.`, 'system-message');
                } else {
                     addMessageToChat(`Couldn't extract new information from that statement.`, 'system-message');
                }
            }
        } catch (error) {
             console.error("Error processing input:", error);
             addMessageToChat('Error processing input. Check console.', 'error-message');
        } finally {
             sendButton.disabled = false;
             userInput.focus(); // Put focus back on input
        }
    }

    sendButton.addEventListener('click', handleUserInput);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleUserInput();
        }
    });


    // --- Initial Setup ---
    initializeGraph();

}); // End DOMContentLoaded