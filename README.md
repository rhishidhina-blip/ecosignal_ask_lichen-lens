# EcoSignal Ask LichenLens — Semantic Context Update

The keyword topic router is removed. Gemini interprets the full current question
and up to six recent messages, resolves follow-up referents and supplies a bounded
standalone retrieval question. Retrieval and grounded answer generation follow.
No particular words or phrase templates are required for topic acceptance.

Scientific claim checks, citation validation, stored-record requirements,
concurrency, bounded retries, answer caching and audit logging remain. Targeted
prompt/secret protection still runs separately. History is untrusted context,
not scientific evidence. If AI interpretation fails, the assistant reports that
uncertainty rather than falsely calling the question out of scope.

Upload ZIP contents to the existing GitHub repository root and redeploy Render.
Keep GEMINI_API_KEY private and the normal start command. Default primary model
is gemini-3.1-flash-lite. See CONVERSATION_CONTEXT.md for website history support.

Each uncached accepted request normally makes two generation calls: interpretation
and answer. Each stage has the existing bounded application retries and timeout.
Cached educational answers still require interpretation, but skip answer generation.
SDK automatic retries are disabled so application limits are meaningful.

RENDER_DIAGNOSTICS.md describes the optional disabled one-shot diagnostic. The
full diagnostic now tests both semantic interpretation and answer generation.
