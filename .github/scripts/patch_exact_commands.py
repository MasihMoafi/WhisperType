import re
from pathlib import Path

COMMAND_WORDS = (
    'copy', 'paste', 'tab', 'tap', 'dash', 'switch',
    'desktop', 'exit', 'enter', 'delete', 'escape'
)

patched = []
for path in Path('.').rglob('*.py'):
    if '.git' in path.parts or path.name == 'patch_exact_commands.py':
        continue
    text = path.read_text(encoding='utf-8')
    original = text

    # Generic command-table implementation.
    if 'candidate.startswith(prefix)' in text:
        text = text.replace('COMMAND_PREFIXES', 'COMMANDS')
        text = text.replace('for prefix, action in COMMANDS:', 'for command, action in COMMANDS:')
        text = text.replace('if candidate.startswith(prefix):', 'if candidate == command:')

    # Direct if/elif implementation used by the portable Windows script.
    direct_pattern = r'transcribed_text\.startswith\("(' + '|'.join(COMMAND_WORDS) + r')"\)'
    if re.search(direct_pattern, text):
        anchor = 'transcribed_text = original_transcribed_text.lower()'
        replacement = anchor + "\n            command_text = transcribed_text.strip().rstrip(\"!.,;:\")"
        if 'command_text = transcribed_text.strip()' not in text:
            text = text.replace(anchor, replacement)
        text = re.sub(direct_pattern, r'command_text == "\1"', text)

    if text != original:
        if '.startswith("copy")' in text or '.startswith("delete")' in text:
            raise RuntimeError(f'Unsafe copy/delete prefix matching remains in {path}')
        path.write_text(text, encoding='utf-8')
        patched.append(str(path))

if not patched:
    raise RuntimeError('No unsafe voice-command matcher was found')

# Regression check for the intended behavior.
def match(candidate, commands):
    candidate = candidate.strip().lower().rstrip('!.,;:')
    for command, action in commands:
        if candidate == command:
            return action
    return None

sample = [('copy', 'copy'), ('delete', 'delete')]
assert match('copy', sample) == 'copy'
assert match('delete', sample) == 'delete'
assert match('copy this paragraph', sample) is None
assert match('delete the old function', sample) is None

print('Patched:', ', '.join(patched))
