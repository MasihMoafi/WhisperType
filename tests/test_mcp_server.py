import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_server(**env):
    spec = importlib.util.spec_from_file_location("voice_mcp_server", ROOT / "voice_mcp_server.py")
    module = importlib.util.module_from_spec(spec)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("WHISPERTYPE_")}
    sys.modules.pop("cloud_transcribe", None)  # it reads WHISPERTYPE_CLOUD_MODEL at import
    with patch.dict(os.environ, {**clean, **env}, clear=True):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("mcp"), "needs the mcp package")
class McpServerTests(unittest.TestCase):
    def test_transcription_defaults_to_openrouter(self):
        server = load_server()
        with patch.object(server.cloud, "transcribe", return_value="hello") as cloud, \
                patch("os.path.isfile", return_value=True):
            result = json.loads(server.transcribe_audio(audio_path="clip.wav"))
        cloud.assert_called_once_with("clip.wav", model="openai/gpt-4o-mini-transcribe")
        self.assertEqual((result["success"], result["text"], result["backend"]), (True, "hello", "openrouter"))

    def test_status_reports_backend_without_exposing_the_key(self):
        server = load_server(WHISPERTYPE_CLOUD_MODEL="nvidia/parakeet-tdt-0.6b-v3")
        with patch.dict(os.environ, OPENROUTER_API_KEY_COMPANY="sk-secret"), \
                patch.object(server, "whispertype", return_value='{"running": false}'):
            status = server.voice_status()
        self.assertNotIn("sk-secret", status)
        status = json.loads(status)
        self.assertEqual(status["backend"], {"default": "openrouter", "model": "nvidia/parakeet-tdt-0.6b-v3", "company_key_configured": True})
        self.assertEqual(status["dictation"], {"running": False})

    def test_dictation_controls_use_the_whispertype_cli(self):
        server = load_server()
        with patch.object(server, "whispertype", return_value='{"running": true}') as cli:
            server.start_dictation(engine="cloud", model="nvidia/parakeet-tdt-0.6b-v3")
            server.stop_dictation()
        self.assertEqual(cli.call_args_list[0].args, ("cloud", "--background"))
        self.assertEqual(cli.call_args_list[0].kwargs["env"], {"WHISPERTYPE_CLOUD_MODEL": "nvidia/parakeet-tdt-0.6b-v3"})
        self.assertEqual(cli.call_args_list[1].args, ("stop",))

    def test_unknown_engine_is_refused(self):
        server = load_server()
        result = json.loads(server.start_dictation(engine="rm -rf"))
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
