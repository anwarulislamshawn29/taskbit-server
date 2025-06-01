import { InferenceClient } from '@huggingface/inference';

const client = new InferenceClient('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');

import express from 'express';
const app = express();

app.use(express.json());

app.post('/prioritize', async (req, res) => {
  try {
    const { description, deadline, time } = req.body;

    // Use InferenceClient for text classification
    const response = await client.textClassification({
      model: 'distilbert-base-uncased-finetuned-sst-2-english',
      inputs: description,
    });

    // Extract the sentiment label and calculate priority
    let priority = response[0].label === 'POSITIVE' ? 0.7 : 0.3;

    // Adjust priority based on deadline and time
    if (deadline) {
      const dueDate = new Date(deadline);
      const now = new Date();
      const daysUntilDue = (dueDate - now) / (1000 * 60 * 60 * 24);
      if (daysUntilDue < 1) priority += 0.3;
      else if (daysUntilDue < 3) priority += 0.2;
      else if (daysUntilDue < 7) priority += 0.1;
      if (dueDate < now) priority += 0.2;
    }
    if (time > 0) {
      if (time > 10) priority -= 0.1;
      else if (time < 2) priority += 0.1;
    }

    priority = Math.min(Math.max(priority, 0), 1);
    res.json({ priority });
  } catch (error) {
    console.error('Error prioritizing task:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

export default app;