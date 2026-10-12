import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.auth import create_access_token

client = TestClient(app)

def test_chat_history_and_voice():
    token_1 = create_access_token(user_id=101, username="chat_user_1", role="user")
    headers_1 = {"Authorization": f"Bearer {token_1}"}

    token_2 = create_access_token(user_id=102, username="chat_user_2", role="user")
    headers_2 = {"Authorization": f"Bearer {token_2}"}

    # 1. Clear any prior history
    client.delete("/chat/history", headers=headers_1)
    client.delete("/chat/history", headers=headers_2)

    # 2. Get history - initially empty
    res = client.get("/chat/history", headers=headers_1)
    assert res.status_code == 200
    assert res.json() == []

    # 3. Ask question as user 1
    ask_res = client.post(
        "/ask",
        json={"query": "Hello world from User 1?"},
        headers=headers_1
    )
    assert ask_res.status_code == 200

    # 4. Verify user 1 has history
    h1_res = client.get("/chat/history", headers=headers_1)
    assert h1_res.status_code == 200
    h1_data = h1_res.json()
    assert len(h1_data) == 1
    assert h1_data[0]["user_query"] == "Hello world from User 1?"

    # 5. Verify user 2 has NO history (user isolation!)
    h2_res = client.get("/chat/history", headers=headers_2)
    assert h2_res.status_code == 200
    assert h2_res.json() == []

    # 6. Test delete history for user 1
    del_res = client.delete("/chat/history", headers=headers_1)
    assert del_res.status_code == 200

    h1_after_del = client.get("/chat/history", headers=headers_1)
    assert h1_after_del.status_code == 200
    assert h1_after_del.json() == []

    # 7. Test TTS endpoint
    tts_res = client.post("/voice/tts", json={"text": "Hello world, testing voice over."})
    assert tts_res.status_code == 200
    assert tts_res.headers["content-type"] == "audio/mpeg"
    assert len(tts_res.content) > 0
