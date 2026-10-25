# Apply the data-only chatbot overlay

Open the latest EcoSignal Site source for appgprj_6a5266428ac48191be22792a6f91e469. Read 00_START_HERE.md. Compare the current files with baseline commit 9ac604128d4f018dcaf9ea20660ac373642fe547 before merging. Merge only Sites_Patch files; preserve newer unrelated changes, all existing features, runtime secrets and audience. The user requested a ZIP in the originating conversation, so creation of this archive did not deploy changes.

Remove photographs from Ask LichenLens, retain structured observation data and chat history. Keep the LichenLens image-analysis feature and stored image assets intact. Do not silently discard selected fields. Preserve the existing context receipt/privacy checks. Build and run chat and existing regression checks before publishing when instructed to apply this patch.

For full repository tests, Sites_Patch/tests/multimodal-chat.test.mjs REPLACES the former multimodal test at the same path. Image sanitization helpers remain in the source for existing module compatibility, but the active chat paths reject image selection before provider calls. Do not re-enable image input just to satisfy obsolete multimodal tests.

Update the matching embedded integrations/ask-lichenlens-render/ source from Render_Backend/. The live Render service requires its own repository commit and deployment; do not imply a Site publication deploys Python. Existing Site ECOSIGNAL_CHATBOT_URL stays unchanged. Do not put keys into source or browser code.

Verify both direct Gemini and Render proxy with selected data, unselected-coordinate exclusion, conversation history and no inline image parts. Verify the menu offers no Photograph item. Test actual live text generation after Render deployment and report Gemini 503 separately from transport success.
