# Data

Place the parallel corpus at `data/train_data1.json`. It is not committed to the repository.

Format: one key per language pair, then a split (`Train` / `Validation`), then entries with a
`source` (English) and `target` (Hindi) sentence:

```json
{
  "English-Hindi": {
    "Train": {
      "1": {"source": "Most of the children are born with innate physical defects .",
            "target": "अधिकांश बच्चे जन्मजात शारीरिक विकारों के साथ पैदा होते हैं ।"}
    },
    "Validation": {
      "1": {"source": "...", "target": "..."}
    }
  }
}
```

The English-Hindi training split contains 80,797 sentence pairs.
