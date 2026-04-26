from unittest.mock import MagicMock, patch

import requests

from llms.tools.transcripts import BiliStoreTranscriptClient


def test_transcript_client_returns_structured_error_on_http_failure():
    response = MagicMock()
    response.status_code = 404
    response.raise_for_status.side_effect = requests.HTTPError(
        "404 Client Error",
        response=response,
    )

    with patch("llms.tools.transcripts.requests.post", return_value=response):
        client = BiliStoreTranscriptClient(base_url="http://bili-store.test")

        result = client.get_video_transcript("BV1missing")

    assert result == {
        "error": "Transcript request failed with HTTP 404",
        "status_code": 404,
        "path": "/transcripts/BV1missing",
    }
