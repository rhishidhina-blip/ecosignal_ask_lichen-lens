# Recent conversation and semantic interpretation

Send the current question plus prior messages, oldest first, excluding the current
question. At most six user/assistant messages of 1–1200 nonblank characters each
are accepted. All six are available as untrusted conversational context. There
is no topic keyword allowlist or short-follow-up phrase matcher.

```json
{
  "question": "Could you unpack that with a few concrete cases?",
  "user_id": "cac-answer-quality-test",
  "conversation_id": "current-chat-id",
  "conversation_history": [
    {"role": "user", "content": "What do lichens primarily bioindicate?"},
    {"role": "assistant", "content": "They can reflect atmospheric pollution exposure."}
  ]
}
```

Gemini reads the full request and history, identifies the task semantically and
resolves referents for retrieval. Clearly unrelated questions remain outside the
app's scope; genuinely ambiguous questions request clarification. History never
becomes an approved citation or a real observation. Current sample_id / mission_id
must select records for specific scan or mission interpretation.

conversation_id is only an audit label, not authentication or automatic stored
history retrieval. The website must send conversation_history. The notebook chat
widget maintains and sends recent turns automatically and includes a New chat
button. Context resets when the selected sample or mission changes.

Frontend request construction:
```javascript
const conversation_history = priorMessages
  .filter(m => m.role === "user" || m.role === "assistant")
  .slice(-6)
  .map(m => ({role: m.role, content: m.content.trim().slice(0, 1200)}))
  .filter(m => m.content.length);
await fetch(`${BACKEND_URL}/assistant/ask`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({question: currentQuestion, user_id: userId,
    conversation_id: conversationId, conversation_history})
});
```

The ZIP updates the backend. It does not modify the deployed website request
builder. Existing clients without history still work for self-contained questions.
History is not additionally saved in the existing audit log, and contextual answers
bypass the shared educational answer cache. This is bounded recent context, not
unlimited memory. Diagnostics should remain disabled for normal use.
