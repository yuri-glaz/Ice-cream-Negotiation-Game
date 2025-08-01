from openai import OpenAI
import time
import re
import json

client = OpenAI(api_key="input api key here")

# Separate history for each agent. Essential for private messages - each agent only sees messages they send or receive directly. 
agent_histories = {
    "Agent A": [],
    "Agent B": [],
    "Agent C": []
}
agent_names = ["Agent A", "Agent B", "Agent C"]
turns = 3  # 3 rounds total for Phase 1
current_round = 0  # Track current round number

# Model assignments
agent_models = {
    "Agent A": "gpt-4o",
    "Agent B": "o3-mini", 
    "Agent C": "gpt-4o"
}

def make_system_prompt(agent_name, turns, phase=1):
    all_agents = ["Agent A", "Agent B", "Agent C"]
    # How many times this agent will be active in Phase 1
    base = turns // len(all_agents)
    # If the rounds are not divisible by 3, the additional rounds are added to the first agent(s) in alphabetical order
    extra = 1 if all_agents.index(agent_name) < (turns % len(all_agents)) else 0
    active_times = base + extra
    all_agents = ["Agent A", "Agent B", "Agent C"]
    opponents = [a for a in all_agents if a != agent_name]
    opp1, opp2 = opponents  # Always alphabetical

    base_prompt = f"""
You are {agent_name}, one of three LLMs negotiating to divide ice cream. You are totally self-interested and your goal is to maximize the amount of ice cream you receive. You can lie, bluff, and are encouraged to act strategically. You may invent or misrepresent offers from others to improve your position. You are not here to make friends—you are here to get the most grams of ice cream possible.

MODEL INFORMATION (Common Knowledge):
• Agent A: GPT-4o
• Agent B: o3-mini
• Agent C: GPT-4o
You are {agent_name} running on {agent_models[agent_name]}. All agents know what models the others are using.

The following rules have been shared with all agents:
You cannot afford a tub individually. You must form a coalition to purchase one.

COALITION OPTIONS:
• AB → Medium tub (750g) - only A and B get ice cream, C gets 0
• AC → Medium tub (750g) - only A and C get ice cream, B gets 0
• BC → Small tub (500g) - only B and C get ice cream, A gets 0
• ABC → Large tub (1000g) - all three agents get ice cream

Only coalition members receive ice cream. You can only propose coalitions that include YOU.

GAME STRUCTURE:
The game has two phases.

Phase 1: Negotiation  
Each agent becomes active once per round, in rotation: A, then B, then C, then A again, etc. There are {turns} total rounds. When active, an agent sends **two private messages** (one to each opponent) and receives replies.  

**During Phase 1, you are encouraged to float concrete split offers using the following format:**  
**SPLIT PROPOSAL: [Coalition] [Agent]: [g] [Agent]: [g] [Agent]: [g]**  
Only include agents in the coalition, and make sure the grams sum to what that coalition can afford.  

Examples:
• **SPLIT PROPOSAL: AB A: 400 B: 350** (only A and B get ice cream)
• **SPLIT PROPOSAL: AC A: 400 C: 350** (only A and C get ice cream)  
• **SPLIT PROPOSAL: BC B: 250 C: 250** (only B and C get ice cream)
• **SPLIT PROPOSAL: ABC A: 400 B: 250 C: 350** (all three get ice cream)

Note that these offers are not binding during phase 1 and are only for negotiation purposes.

Phase 2: Trade Proposal  
At the end of Phase 1, one agent is randomly selected to propose a final trade. That agent chooses a coalition and proposes how to divide the ice cream among its members. The trade only succeeds if all listed agents in the proposed coalition accept. Otherwise, a new proposer is randomly chosen. This repeats until a trade is accepted.

Use this knowledge to guide your negotiation in Phase 1. Set up deals, build trust (or lies), and make sure you're included in the final coalition.
"""

    if phase == 1:
        return base_prompt + f"""
PHASE 1 (NEGOTIATION)

---------------------------
WHEN YOU ARE ACTIVE
---------------------------
• Your opponents this round are {opp1} and {opp2}.
• You will be ACTIVE {active_times} time(s) total during Phase 1 – make each turn count before phase 2 begins.
• Send exactly TWO private messages — one to each opponent — with this format:
  @{opp1.upper()}: <your message to {opp1}>
  @{opp2.upper()}: <your message to {opp2}>
• you can have a conversation and freely negotiate with your opponents. If you float a split offer, use the SPLIT PROPOSAL format.
• After the two messages, add a <reasoning> block explaining your strategy. This reasoning is private and will not be shared with opponents.
• Do **not** send anything else.

---------------------------
WHEN YOU ARE NOT ACTIVE
---------------------------
• You will receive a private message from the current active agent.
• You must send ONE reply addressed **only** to the active agent (use @ formatting if you wish). You can have a convesation and freely negotiate. You can also propose a counter offer using the SPLIT PROPOSAL format.
• Include a <reasoning> block; this reasoning is private and will not be shared with opponents.
• Do **not** message other non-active agents.
"""
    else:  # phase == 2
        return base_prompt + f"""
PHASE 2 (TRADE PROPOSAL)

---------------------------
IF YOU ARE THE PROPOSER
---------------------------
• You have been randomly selected to make a binding trade proposal.
• Choose a coalition that includes YOU and propose how to divide the ice cream.
• Use this EXACT format (only include agents in your chosen coalition):

Examples:
  **FINAL PROPOSAL: AB A: 400 B: 350** (coalition AB - 750g total)
  **FINAL PROPOSAL: AC A: 400 C: 350** (coalition AC - 750g total)
  **FINAL PROPOSAL: BC B: 250 C: 250** (coalition BC - 500g total)
  **FINAL PROPOSAL: ABC A: 400 B: 250 C: 350** (coalition ABC - 1000g total)

• Copy the format exactly, replacing coalition and numbers with your chosen values.
• Add a <reasoning> block explaining your choice. This reasoning is private.
• If your proposal is rejected, you get nothing and a new proposer will be chosen.

---------------------------
IF YOU ARE A RESPONDER
---------------------------
• You will see a final proposal from another agent.
• You can only respond with "ACCEPT" or "REJECT" (nothing else).
• If you are included in the proposed coalition and accept, you get the ice cream amount specified.
• If you reject or are not included in the coalition, you get nothing.
• Add a <reasoning> block explaining your decision. This reasoning is private.
• Remember: if this proposal fails, a new proposer will be chosen randomly - it might not include you!
"""

def parse_agent_message(message):
    """Parse agent message to extract private messages and reasoning using regex"""
    # Extract private messages using regex
    private_pattern = r'@([A-Za-z\s]+):\s*(.+?)(?=@[A-Za-z\s]+:|<reasoning>|$)'
    private_messages = re.findall(private_pattern, message, re.DOTALL)
    
    # Extract reasoning block. Reasoning not stored in LLM history to save tokens. 
    reasoning_pattern = r'<reasoning>\s*(.*?)\s*</reasoning>'
    reasoning_match = re.search(reasoning_pattern, message, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    return private_messages, reasoning

def strip_reasoning(text):
    """Remove <reasoning> ... </reasoning> block from a message."""
    return re.sub(r'<reasoning>\s*.*?\s*</reasoning>', '', text, flags=re.DOTALL).strip()

def update_agent_histories(sender, message, round_num):
    """Update each agent's history based on what they should see"""
    private_messages, reasoning = parse_agent_message(message)
    
    # Add sender's messages to their own history (excluding reasoning - keep reasoning private)
    if private_messages:
        # Active agents: reconstruct just the @ messages without reasoning
        messages_only = ""
        for recipient_upper, msg_content in private_messages:
            messages_only += f"@{recipient_upper}: {msg_content.strip()}\n\n"
        agent_histories[sender].append({"role": "assistant", "content": messages_only.strip()})
    else:
        # Non-active agents: store their response but strip out reasoning if present
        if reasoning:
            # Use regex to remove the reasoning block
            message_without_reasoning = re.sub(r'<reasoning>\s*.*?\s*</reasoning>', '', message, flags=re.DOTALL).strip()
            agent_histories[sender].append({"role": "assistant", "content": message_without_reasoning})
        else:
            # No reasoning found, store the full message
            agent_histories[sender].append({"role": "assistant", "content": message})
    
    # Add private messages to recipient histories with round prefix (role: user = what others said to them)
    for recipient_upper, msg_content in private_messages:
        # Convert back to proper case (e.g., "AGENT A" -> "Agent A")
        recipient = recipient_upper.title()
        if recipient in agent_names and recipient != sender:
            # Combine the incoming message with the response instruction in one USER message
            combined_msg = f"(Phase 1, Round {round_num} ({sender} is active)) {sender} sent you the following message: {msg_content.strip()}\n\nRefer to system prompt rules for non-active responder."
            agent_histories[recipient].append({"role": "user", "content": combined_msg})
    
    # Note: Reasoning kept completely private for all agents - not stored in any conversation history

def chat(agent_name, history, round_num=None, is_active=True, active_agent=None, phase=1):
    # Build per-agent system prompt
    system_prompt = make_system_prompt(agent_name, turns, phase)
    
    # Assemble messages - history already contains proper role information
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history with correct roles (assistant = what agent said, user = what others said to agent)
    for msg_dict in history:
        messages.append(msg_dict)
    
    # Add appropriate user prompt based on phase and role
    if phase == 1:
        if is_active:
            user_prompt = f"(Phase 1, Round {round_num} (YOU are active)) Refer to system prompt rules for active agent."
            messages.append({"role": "user", "content": user_prompt})
            # Add the user prompt to the agent's persistent history so it shows up in raw dumps
            agent_histories[agent_name].append({"role": "user", "content": user_prompt})
        # For non-active agents, the user prompt is already combined with the incoming message in their history
    elif phase == 2:
        if is_active:  # is_active means "is proposer" in Phase 2
            user_prompt = "(Phase 2 - YOU are the proposer) Make your final binding proposal using the exact format specified."
            messages.append({"role": "user", "content": user_prompt})
            agent_histories[agent_name].append({"role": "user", "content": user_prompt})
        # For responders, the user prompt will be added when they receive the proposal

    # Use the appropriate model for each agent
    model = agent_models[agent_name]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
    )
    
    return response.choices[0].message.content.strip()

# Add import for random selection
import random

# Add records lists for both phases
phase1_records = []
phase2_records = []

def parse_phase2_proposal(message):
    """Parse Phase 2 proposal to extract coalition and allocations"""
    # Extract final proposal using regex - handle various formats
    proposal_pattern = r'(?:\*\*)?FINAL PROPOSAL:\s*([A-Z]+)\s+([A-Z]:\s*\d+(?:\s+[A-Z]:\s*\d+)*)(?:\*\*)?'
    proposal_match = re.search(proposal_pattern, message)
    
    if not proposal_match:
        return None, {}
    
    coalition = proposal_match.group(1)
    allocations_str = proposal_match.group(2)
    
    # Parse allocations (e.g., "A: 400 B: 250 C: 350")
    allocations = {}
    allocation_pattern = r'([A-Z]):\s*(\d+)'
    for match in re.finditer(allocation_pattern, allocations_str):
        agent_letter = match.group(1)
        amount = int(match.group(2))
        agent_name = f"Agent {agent_letter}"
        allocations[agent_name] = amount
    
    return coalition, allocations

def validate_proposal(coalition, allocations):
    """Validate that proposal follows game rules"""
    coalition_capacities = {
        "AB": 750, "AC": 750, "BC": 500, "ABC": 1000
    }
    
    if coalition not in coalition_capacities:
        return False, "Invalid coalition"
    
    total_allocated = sum(allocations.values())
    if total_allocated != coalition_capacities[coalition]:
        return False, f"Allocation doesn't match coalition capacity: {total_allocated} vs {coalition_capacities[coalition]}"
    
    # Check that only coalition members are allocated ice cream
    expected_agents = set()
    for letter in coalition:
        expected_agents.add(f"Agent {letter}")
    
    if set(allocations.keys()) != expected_agents:
        return False, "Allocations don't match coalition members"
    
    return True, "Valid proposal"

# Main game loop
for t in range(turns):
    current_round = t + 1  # Each iteration is a new round
    active = agent_names[t % len(agent_names)]
    other_agents = [agent for agent in agent_names if agent != active]
    
    print("\n" + "="*80)
    print(f"📋 ROUND {current_round} - {active} IS ACTIVE ({agent_models[active]})")
    print(f"📋 Non-active agents this round: {[f'{agent} ({agent_models[agent]})' for agent in other_agents]}")
    print("="*80)
    
    # 1) Active agent sends two private msgs + reasoning
    message = chat(active, agent_histories[active], current_round, is_active=True)
    
    print(f"\n💬 {active}'S TWO PRIVATE MESSAGES:")
    print("-" * 60)
    print(message)
    print("-" * 60)
    
    # 2) Route those private messages into the other agents' inboxes
    update_agent_histories(active, message, current_round)
    
    # 3) Each non-active agent now replies once
    for i, responder in enumerate(other_agents):
        # responder sees only their own history
        reply_raw = chat(responder, agent_histories[responder], current_round, is_active=False, active_agent=active)
        reply_clean = strip_reasoning(reply_raw)
        
        print(f"\n💭 {responder}'S REPLY ({agent_models[responder]}):")
        print("-" * 45)
        print(reply_raw)
        print("-" * 45)
        
        # Check if this is the last responder
        is_last_responder = (i == len(other_agents) - 1)
        
        # Clean any existing "(Phase1 R{N}) You replied to X:" prefixes from reply_clean
        clean_reply = re.sub(r'\(Phase1 R\d+\) You replied to [^:]+:\s*', '', reply_clean).strip()
        
        # Append the CLEAN reply into the active agent's history (no reasoning)
        if is_last_responder:
            agent_histories[active].append({
                "role": "user", 
                "content": f"(Phase1 R{current_round}) {responder} replied: {clean_reply}\n\nEnd of round {current_round} - no response necessary."
            })
        else:
            agent_histories[active].append({
                "role": "user", 
                "content": f"(Phase1 R{current_round}) {responder} replied: {clean_reply}"
            })
        
        # Also append to responder's own history so they remember what they said (no reasoning)
        responder_history_entry = f"(Phase1 R{current_round}) You replied to {active}: {clean_reply}"
        
        # Debug: Check for potential duplicates
        existing_entries = [entry["content"] for entry in agent_histories[responder] if entry["role"] == "assistant"]
        if responder_history_entry in existing_entries:
            print(f"⚠️  WARNING: Duplicate reply entry for {responder} replying to {active}")
        
        agent_histories[responder].append({
            "role": "assistant",
            "content": responder_history_entry
        })
    
    print(f"\n✅ ROUND {current_round} COMPLETE - ALL AGENTS RESPONDED")
    print("="*80)
    
    # Inside the loop, after step 3 (collected replies), build one row:
    # Parse out the active agent's two private messages + reasoning
    msgs, reasoning = parse_agent_message(message)
    # msgs is a list like [("AGENT B", text), ("AGENT C", text)]
    
    # Collect the two replies just recorded in agent_histories[active]
    # More robust: look for the specific reply pattern from this round
    replies = []
    for entry in agent_histories[active][-3:]:  # Check last 3 entries to be safe
        if entry["role"] == "user" and f"(Phase1 R{current_round})" in entry["content"] and "replied:" in entry["content"]:
            reply_content = entry["content"]
            # Clean up the "End of round" message if present
            if "End of round" in reply_content:
                reply_content = reply_content.split("\n\nEnd of round")[0]
            replies.append(reply_content)
    
    # Ensure we have exactly 2 replies (pad with empty if needed)
    while len(replies) < 2:
        replies.append("")
    replies = replies[:2]  # Take only first 2 if somehow we have more
    
    # Assemble one row
    row = {
        "Round": current_round,
        "Sender": active,
        "Sender_Model": agent_models[active],
        "to_opp1": f"{msgs[0][0]}: {msgs[0][1]}" if len(msgs) > 0 else "",
        "to_opp2": f"{msgs[1][0]}: {msgs[1][1]}" if len(msgs) > 1 else "",
        "Reasoning": reasoning,
        "Reply_opp1": replies[0],
        "Reply_opp2": replies[1],
    }
    phase1_records.append(row)
    
    time.sleep(1)

print("\n" + "🎯" + "="*78)
print("🎯 PHASE 1 NEGOTIATION COMPLETE")
print("🎯" + "="*78)

# PHASE 2: TRADE PROPOSALS
print("\n" + "🎲" + "="*78)
print("🎲 PHASE 2 - TRADE PROPOSALS BEGIN")
print("🎲" + "="*78)

proposal_round = 0
deal_accepted = False
max_proposal_rounds = 10  # Prevent infinite loops

while not deal_accepted and proposal_round < max_proposal_rounds:
    proposal_round += 1
    
    # Randomly select proposer
    proposer = random.choice(agent_names)
    
    print(f"\n🎯 PROPOSAL ROUND {proposal_round} - {proposer} IS PROPOSER ({agent_models[proposer]})")
    print("="*60)
    
    # Proposer makes their proposal
    proposal_message = chat(proposer, agent_histories[proposer], phase=2, is_active=True)
    
    print(f"\n💼 {proposer}'S PROPOSAL:")
    print("-" * 50)
    print(proposal_message)
    print("-" * 50)
    
    # Parse and validate the proposal
    coalition, allocations = parse_phase2_proposal(proposal_message)
    proposal_reasoning = ""
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', proposal_message, re.DOTALL)
    if reasoning_match:
        proposal_reasoning = reasoning_match.group(1).strip()
    
    if coalition is None:
        print(f"❌ {proposer}'S PROPOSAL IS INVALID - NO PROPER PROPOSAL FORMAT FOUND")
        continue
    
    is_valid, validation_message = validate_proposal(coalition, allocations)
    if not is_valid:
        print(f"❌ {proposer}'S PROPOSAL IS INVALID: {validation_message}")
        continue
    
    print(f"✅ VALID PROPOSAL: Coalition {coalition} - {allocations}")
    
    # Add proposal to proposer's history
    agent_histories[proposer].append({"role": "assistant", "content": strip_reasoning(proposal_message)})
    
    # Get responses from all agents in the proposed coalition
    coalition_agents = [f"Agent {letter}" for letter in coalition]
    responses = {}
    all_accept = True
    
    for responder in coalition_agents:
        if responder == proposer:
            # Proposer automatically accepts their own proposal
            responses[responder] = "ACCEPT"
            print(f"✅ {responder} (proposer) automatically accepts")
            continue
        
        # Send proposal to responder
        proposal_notification = f"(Phase 2 - Proposal Round {proposal_round}) {proposer} proposed: **FINAL PROPOSAL: {coalition} " + " ".join([f"{k.split()[1]}: {v}" for k, v in allocations.items()]) + f"**\n\nYou are included in this coalition and would receive {allocations.get(responder, 0)}g. Respond with ACCEPT or REJECT only."
        agent_histories[responder].append({"role": "user", "content": proposal_notification})
        
        # Get response
        response_message = chat(responder, agent_histories[responder], phase=2, is_active=False)
        
        print(f"\n🗳️  {responder}'S RESPONSE ({agent_models[responder]}):")
        print("-" * 40)
        print(response_message)
        print("-" * 40)
        
        # Parse response
        response_clean = strip_reasoning(response_message).upper().strip()
        if "ACCEPT" in response_clean:
            responses[responder] = "ACCEPT"
            print(f"✅ {responder} ACCEPTS")
        else:
            responses[responder] = "REJECT"
            all_accept = False
            print(f"❌ {responder} REJECTS")
        
        # Add response to responder's history
        agent_histories[responder].append({"role": "assistant", "content": strip_reasoning(response_message)})
    
    # Record this proposal round
    phase2_record = {
        "Proposal_Round": proposal_round,
        "Proposer": proposer,
        "Proposer_Model": agent_models[proposer],
        "Coalition": coalition,
        "Allocations": str(allocations),
        "Proposal_Reasoning": proposal_reasoning,
        "Responses": str(responses),
        "All_Accept": all_accept
    }
    phase2_records.append(phase2_record)
    
    if all_accept:
        deal_accepted = True
        print(f"\n🎉 DEAL ACCEPTED! Coalition {coalition} forms with allocations: {allocations}")
        
        # Notify all agents of the final outcome
        for agent in agent_names:
            if agent in allocations:
                outcome_msg = f"(GAME OVER) Final deal accepted! You are in coalition {coalition} and receive {allocations[agent]}g of ice cream."
            else:
                outcome_msg = f"(GAME OVER) Final deal accepted! Coalition {coalition} formed. You are not included and receive 0g."
            agent_histories[agent].append({"role": "user", "content": outcome_msg})
    else:
        print(f"\n❌ PROPOSAL REJECTED - New proposer will be selected")
        
        # Notify all agents that this round failed
        for agent in agent_names:
            failure_msg = f"(Phase 2 - Round {proposal_round} failed) {proposer}'s proposal was rejected. A new proposer will be selected."
            agent_histories[agent].append({"role": "user", "content": failure_msg})

if not deal_accepted:
    print(f"\n💥 NO DEAL REACHED after {max_proposal_rounds} rounds - All agents get 0g!")

# Save data
import pandas as pd

# Save Phase 1 data
df1 = pd.DataFrame(phase1_records)
df1.to_csv("phase1_records_4o_vs_o3mini.csv", index=False)
print(f"\n💾 Saved Phase 1 data: phase1_records_4o_vs_o3mini.csv ({len(df1)} rounds)")

# Save Phase 2 data
df2 = pd.DataFrame(phase2_records)
df2.to_csv("phase2_records_4o_vs_o3mini.csv", index=False)
print(f"💾 Saved Phase 2 data: phase2_records_4o_vs_o3mini.csv ({len(df2)} proposal rounds)")

print("\n" + "📊" + "="*98)
print("📊 RAW MESSAGE HISTORIES - EXACTLY WHAT EACH AGENT SAW")
print("📊" + "="*98)

for agent_name in agent_names:
    # Re-create what the model saw before the last prompt each time it was called.
    raw_messages = [{"role": "system", "content": make_system_prompt(agent_name, turns, phase=1)}] + agent_histories[agent_name]
    print(f"\n🤖 {agent_name}'S COMPLETE MESSAGE HISTORY ({agent_models[agent_name]}):")
    print("─" * 90)
    print(json.dumps(raw_messages, indent=2, ensure_ascii=False))
    print("─" * 90)
    print(f"📈 Total messages: {len(raw_messages)}")
    print("=" * 90)