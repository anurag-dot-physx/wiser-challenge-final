from pathlib import Path

path = Path("README.md")
text = path.read_text()

team_block = '''## Team: QPhysicists

| Team member | Email | Institution | Country |
|---|---|---|---|
| **Anurag Sarkar** | [anuragsarkar315@gmail.com](mailto:anuragsarkar315@gmail.com) | University of the Witwatersrand | South Africa |
| **Ankit Gill** | [ankit.1.gill@gmail.com](mailto:ankit.1.gill@gmail.com) | Indian Institute of Technology Kanpur | India |

'''

# Remove any previously inserted top-level team block.
if '## Team: QPhysicists' in text:
    a = text.index('## Team: QPhysicists')
    next_heading = text.find('\n## ', a + 3)
    if next_heading != -1:
        text = text[:a] + text[next_heading + 1:]

# Insert directly after the challenge subtitle.
anchor = '**WISER x Vanguard - Quantum for Finance Challenge 2026**\n'
if anchor not in text:
    raise SystemExit('README challenge subtitle anchor not found')
pos = text.index(anchor) + len(anchor)
text = text[:pos] + '\n' + team_block + text[pos:]

# Replace the later team section with a concise contribution note, avoiding duplicate contacts.
start = '## 10. Team members and contributions'
end = '## 11. AI and tools usage'
if start in text and end in text:
    a = text.index(start)
    b = text.index(end)
    replacement = '''## 10. Team contributions

Both members of **QPhysicists** made equal or comparable contributions across the major stages of the work, including theoretical development, portfolio formulation and constraints, higher-order extensions, classical and quantum optimization workflows, validation, interpretation of results, and overall implementation.

The project evolved through repeated discussion, testing and refinement, and both team members were thoroughly involved throughout that process.

'''
    text = text[:a] + replacement + text[b:]

path.write_text(text)
