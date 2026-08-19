# AI Heuristic Validation Report (Issue #21 validation)

## Summary
- Total samples: 20 (10 AI-labeled, 10 human-labeled)
- Average score - AI-labeled reviews: 0.4872
- Average score - Human-labeled reviews: 0.4259
- Separation (AI avg - Human avg): 0.0613

## Score Ranges
- AI-labeled: 0.3540 - 0.6260
- Human-labeled: 0.2870 - 0.5570

## Conclusion
The heuristic shows the AI-labeled samples score higher on average than human-labeled ones,
in the expected direction. However, the separation (0.0613) is small and the score ranges
overlap significantly, meaning the heuristic alone is not strongly reliable at distinguishing
AI-generated text from human-written text on this small validation set.