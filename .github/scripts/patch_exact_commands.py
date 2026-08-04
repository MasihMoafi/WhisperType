from pathlib import Path

patched = []
for path in Path('.').rglob('*.py'):
    if '.git' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    if 'candidate.startswith(prefix)' not in text:
        continue

    text = text.replace('COMMAND_PREFIXES', 'COMMANDS')
    text = text.replace('for prefix, action in COMMANDS:', 'for command, action in COMMANDS:')
    text = text.replace('if candidate.startswith(prefix):', 'if candidate == command:')

    if 'candidate.startswith(' in text:
        raise RuntimeError(f'Unsafe prefix command matching remains in {path}')

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
