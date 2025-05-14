// script.js

const OPENAI_API_KEY = "";
const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_MODEL = "gpt-4o";

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const gameTextInput = document.getElementById('game-text-input');
  const loadButton     = document.getElementById('load-button');
  const resetButton    = document.getElementById('reset-button');
  const chatHistory    = document.getElementById('chat-history');
  const userInput      = document.getElementById('user-input');
  const sendButton     = document.getElementById('send-button');
  const graphContainer = document.getElementById('graph-container');
  const nodeCountSpan  = document.getElementById('node-count');
  const edgeCountSpan  = document.getElementById('edge-count');

  // Hyperparameters elements
  const decaySlider       = document.getElementById('decay-rate');
  const strengthenSlider  = document.getElementById('strengthen-rate');
  const maxNodesInput     = document.getElementById('max-nodes');
  const maxEdgesInput     = document.getElementById('max-edges');
  const maxOutInput       = document.getElementById('max-outdegree');
  const forgetSlider      = document.getElementById('forget-threshold');
  const permSlider        = document.getElementById('permanence-threshold');
  const decayVal          = document.getElementById('decay-rate-val');
  const strengthenVal     = document.getElementById('strengthen-rate-val');
  const forgetVal         = document.getElementById('forget-threshold-val');
  const permVal           = document.getElementById('permanence-threshold-val');

  // Sync slider displays
  [
    [decaySlider, decayVal],
    [strengthenSlider, strengthenVal],
    [forgetSlider, forgetVal],
    [permSlider, permVal]
  ].forEach(([slider, display]) => {
    display.textContent = slider.value;
    slider.addEventListener('input', () => {
      display.textContent = slider.value;
    });
  });

  // --- Graph State ---
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  let network;
  let worldTriplets = [];
  let edgeIdCounter = 0;

  // --- Initialize Graph ---
  function initializeGraph() {
    const data = { nodes, edges };
    const options = {
      nodes: {
        shape: 'dot', size: 16,
        font: { size: 12, color: '#333' },
        borderWidth: 2,
        color: {
          border: '#2B7CE9',
          background: '#D2E5FF',
          highlight: { border: '#2B7CE9', background: '#FFD569' }
        }
      },
      edges: {
        font: { size: 10, align: 'middle', background: 'rgba(255,255,255,0.7)', color: '#666' },
        arrows: { to: { enabled: true, scaleFactor: 0.7 } },
        color: { color: '#848484', highlight: '#0078d4' }
      },
      physics: {
        forceAtlas2Based: {
          gravitationalConstant: -26,
          centralGravity: 0.005,
          springLength: 230,
          springConstant: 0.18,
          avoidOverlap: 1.5
        },
        maxVelocity: 146,
        solver: 'forceAtlas2Based',
        timestep: 0.35,
        stabilization: { iterations: 150 }
      },
      interaction: { tooltipDelay: 200, hideEdgesOnDrag: true, hover: true }
    };

    network = new vis.Network(graphContainer, data, options);
    updateGraphInfo();

    // Add the player node
    nodes.add({
      id: 'player',
      label: 'Player',
      title: 'Entity: player\nWeight: 1.00',
      weight: 1,
      size: 16
    });
  }

  // --- Update Counts ---
  function updateGraphInfo() {
    nodeCountSpan.textContent = nodes.length;
    edgeCountSpan.textContent = edges.length;
  }

  // --- Chat Logging ---
  function addMessageToChat(message, type) {
    const el = document.createElement('div');
    el.classList.add('chat-message', type + '-message');
    el.textContent = message;
    chatHistory.appendChild(el);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  // --- Visualization Update ---
  function updateGraphVisualization(triplets) {
    const uniqueNodes = new Set();
    const newEdges = [];

    triplets.forEach(([subject, relation, object]) => {
      uniqueNodes.add(subject);
      uniqueNodes.add(object);
      newEdges.push({ from: subject, to: object, label: relation });
    });

    edgeIdCounter = 0;
    nodes.clear();
    edges.clear();

    uniqueNodes.forEach(id => {
      nodes.add({
        id,
        label: id,
        title: `Entity: ${id}\nWeight: 1.00`,
        weight: 1,
        size: 16
      });
    });

    newEdges.forEach(e => {
      const eid = `e${edgeIdCounter++}`;
      edges.add({
        id: eid,
        from: e.from,
        to: e.to,
        label: e.label,
        title: 'Weight: 1.00',
        weight: 1,
        width: 2
      });
    });

    network.fit();
    network.redraw();
    updateGraphInfo();
  }

  // --- Decay & Strengthen Logic ---
  function applyDecayAndStrengthen(question) {
    const decay        = parseFloat(decaySlider.value);
    const strengthen   = parseFloat(strengthenSlider.value);
    const maxNodes     = parseInt(maxNodesInput.value, 10);
    const maxEdges     = parseInt(maxEdgesInput.value, 10);
    const maxOut       = parseInt(maxOutInput.value, 10);
    const forgetThres  = parseFloat(forgetSlider.value);
    const permThres    = parseFloat(permSlider.value);

    // Decay all
    nodes.get().forEach(n => nodes.update({ id: n.id, weight: n.weight * decay }));
    edges.get().forEach(e => edges.update({ id: e.id, weight: e.weight * decay }));

    // Strengthen used nodes/edges
    const usedNodes = nodes.getIds().filter(id =>
      question.toLowerCase().includes(id.toLowerCase())
    );

    usedNodes.forEach(id => {
      const n = nodes.get(id);
      nodes.update({ id, weight: n.weight * strengthen });
    });

    edges.get().forEach(e => {
      if (usedNodes.includes(e.from) || usedNodes.includes(e.to)) {
        edges.update({ id: e.id, weight: e.weight * strengthen });
      }
    });

    // Remove forgotten nodes
    nodes.get().forEach(n => {
      if (n.weight < forgetThres && n.weight <= permThres) {
        nodes.remove({ id: n.id });
        edges.get({ filter: ed => ed.from === n.id || ed.to === n.id })
          .forEach(ed => edges.remove(ed.id));
      }
    });

    // Prune maxNodes
    if (nodes.length > maxNodes) {
      let removable = nodes.get()
        .filter(n => n.weight <= permThres)
        .sort((a, b) => a.weight - b.weight);
      while (nodes.length > maxNodes && removable.length) {
        const rem = removable.shift();
        nodes.remove(rem.id);
        edges.get({ filter: ed => ed.from === rem.id || ed.to === rem.id })
          .forEach(ed => edges.remove(ed.id));
      }
    }

    // Prune maxEdges
    if (edges.length > maxEdges) {
      let removableE = edges.get()
        .filter(e => e.weight <= permThres)
        .sort((a, b) => a.weight - b.weight);
      while (edges.length > maxEdges && removableE.length) {
        edges.remove(removableE.shift().id);
      }
    }

    // Prune outdegree
    nodes.get().forEach(n => {
      let outs = edges.get({ filter: e => e.from === n.id });
      if (outs.length > maxOut) {
        let removableO = outs
          .filter(e => e.weight <= permThres)
          .sort((a, b) => a.weight - b.weight);
        while (edges.get({ filter: e => e.from === n.id }).length > maxOut && removableO.length) {
          edges.remove(removableO.shift().id);
        }
      }
    });

    // Update visuals
    nodes.get().forEach(n => {
      nodes.update({
        id: n.id,
        size: Math.max(10, 16 * n.weight),
        title: `Entity: ${n.id}\nWeight: ${n.weight.toFixed(2)}`
      });
    });

    edges.get().forEach(e => {
      edges.update({
        id: e.id,
        width: Math.max(1, 2 * e.weight),
        title: `Weight: ${e.weight.toFixed(2)}`
      });
    });

    network.fit();
    network.redraw();
    updateGraphInfo();
  }

  // --- OpenAI Helpers ---
  async function callOpenAI(messages) {
    if (!OPENAI_API_KEY || OPENAI_API_KEY.startsWith('sk-YOUR')) {
      throw new Error('OpenAI API Key not configured.');
    }
    const response = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: OPENAI_MODEL,
        messages,
        temperature: 0.5
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error?.message || `HTTP ${response.status}`);
    }
    return data.choices[0].message.content;
  }

  async function llm_extractTriplets(text) {
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
    - DO NOT UNDER ANY CIRCUMSTANCE USE the triple qoutes / json wrapper. JUST OUTPUT THE JSON. YOUR RESPONSE WILL
    BE PARSED BY A DOWNSTREAM PARSER AUTOMATIcALLy, SO INCLUDING ANYTHING OTHER THAN THE RAW JSON WILL CRASH.
    - Ensure the output is strictly JSON. Do not add any introductory text or explanations.`;
    const userPrompt   = `Extract triplets from:\n\n${text}`;
    const resp = await callOpenAI([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ]);
    let parsed = JSON.parse(resp);
    console.log(parsed);
    if (parsed.triplets) parsed = parsed.triplets;
    console.log(parsed);
    if (!Array.isArray(parsed)) throw new Error('Invalid triplet format');
    return parsed;
  }

  async function llm_reconcileTriplets(newTriplets, existing) {
    const systemPrompt = `You are a knowledge graph reconciliation engine. You will receive two lists of triplets: 'existing' and 'new'.
    Your task is to merge these lists into a single, consistent knowledge graph state.
    Rules:
    1.  Identify Conflicts: A conflict occurs when a new triplet has the same subject and relation as an existing triplet, but a different object (e.g., existing: ["key", "is in", "box"], new: ["key", "is in", "inventory"]). In case of conflict, the information from the 'new' triplet is usually more current and should replace the existing conflicting triplet(s).
    2.  Remove Redundancy: If a triplet from the 'new' list is an exact match to one in the 'existing' list, keep only one copy.
    3.  Combine: Include all non-conflicting, non-redundant triplets from both lists in the final output.
    4.  Output Format: Return ONLY a valid JSON array containing the final, reconciled list of triplets. Example: [["subj1", "rel1", "obj1"], ["subj2", "rel2", "obj2"]]. Do not add any explanatory text.
    5.  If the input lists are empty, return an empty array [].
    6.  The result should be under the key result
    7. DO NOT UNDER ANY CIRCUMSTANCE USE the triple qoutes / json wrapper. JUST OUTPUT THE JSON. YOUR RESPONSE WILL
    BE PARSED BY A DOWNSTREAM PARSER AUTOMATIcALLy, SO INCLUDING ANYTHING OTHER THAN THE RAW JSON WILL CRASH.
    `;
    const userPrompt   = `Existing:\n${JSON.stringify(existing)}\nNew:\n${JSON.stringify(newTriplets)}`;
    const resp = await callOpenAI([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ]);
    let parsed = JSON.parse(resp);
    if (parsed.result) parsed = parsed.result;
    if (!Array.isArray(parsed)) throw new Error('Invalid reconciliation');
    return parsed;
  }

  async function llm_answerQuestion(question, currentTriplets) {
    const systemPrompt = `You are an assistant answering questions based *strictly* on the provided knowledge graph context (a list of triplets).
    - Analyze the 'World Knowledge (Triplets)' provided below.
    - Answer the user's 'Question' factually based *only* on the information contained within those triplets.
    - If the information needed to answer the question is not present in the triplets, clearly state that you don't know or that the information is not available in the current world model.
    - Do not make assumptions or use external knowledge.
    - Keep your answer concise and directly relevant to the question based on the provided facts.
    -     - DO NOT UNDER ANY CIRCUMSTANCE USE the triple qoutes / json wrapper. JUST OUTPUT THE JSON. YOUR RESPONSE WILL
    BE PARSED BY A DOWNSTREAM PARSER AUTOMATIcALLy, SO INCLUDING ANYTHING OTHER THAN THE RAW JSON WILL CRASH.
    `;
    const userPrompt   = `World Knowledge:\n${JSON.stringify(currentTriplets, null, 2)}\nQuestion: ${question}`;
    const response = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({ model: OPENAI_MODEL, messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ], temperature: 0.7 })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
    return data.choices[0].message.content.trim();
  }

  // --- Event Handlers ---
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
      const initial = await llm_extractTriplets(text);
      console.log(initial); // Log the initial triplets
      worldTriplets = await llm_reconcileTriplets(initial, []);
      console.log(worldTriplets); // Log the triplets t
      updateGraphVisualization(worldTriplets);
      addMessageToChat(`World model generated with ${nodes.length} nodes and ${edges.length} edges.`, 'system-message');
      gameTextInput.value = '';
    } catch (err) {
        console.log(err)
      addMessageToChat('Error generating world model.', 'error-message');
    } finally {
      loadButton.disabled = false;
      resetButton.disabled = false;
    }
  });

  resetButton.addEventListener('click', () => {
    worldTriplets = [];
    nodes.clear();
    edges.clear();
    chatHistory.innerHTML = '';
    addMessageToChat('World reset.', 'system-message');
    gameTextInput.value = '';
    initializeGraph();
  });

  async function handleUserInput() {
    const text = userInput.value.trim();
    if (!text) return;
    addMessageToChat(text, 'user-message');
    userInput.value = '';
    sendButton.disabled = true;
    try {
      const isQuestion = text.endsWith('?') || /^(what|where|who|is|are|does|do|can|list)\s/i.test(text);
      if (isQuestion) {
        addMessageToChat('Thinking...', 'system-message');
        const answer = await llm_answerQuestion(text, worldTriplets);
        addMessageToChat(answer, 'system-message');
        applyDecayAndStrengthen(text);
      } else {
        addMessageToChat('Updating world model...', 'system-message');
        const newT = await llm_extractTriplets(text);
        if (newT.length) {
          worldTriplets = await llm_reconcileTriplets(newT, worldTriplets);
          updateGraphVisualization(worldTriplets);
          addMessageToChat('World model updated.', 'system-message');
        } else {
          addMessageToChat("Couldn't extract new information.", 'system-message');
        }
      }
    } catch (err) {
        console.log(err)
      addMessageToChat('Error processing input.', 'error-message');
    } finally {
      sendButton.disabled = false;
      userInput.focus();
    }
  }

  sendButton.addEventListener('click', handleUserInput);
  userInput.addEventListener('keypress', e => { if (e.key === 'Enter') handleUserInput(); });

  // --- Start ---
  initializeGraph();
});