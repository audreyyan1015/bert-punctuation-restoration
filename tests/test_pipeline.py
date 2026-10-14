import unittest
import numpy as np
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizer
from pathlib import Path
import tempfile
from punct.data import to_example, restore
from punct.engine import Decoder, windows, encode_chars
from punct.metrics import confusion, summarize


class PipelineTests(unittest.TestCase):
    def test_annotations_and_punctuation(self):
        text, y = to_example('他{ta1} 说 ： “ 好 ， 我 来 ！ ”')
        self.assertEqual(text, '他说好我来')
        self.assertEqual(restore(text, y), '他说：好，我来！')

    def test_window_ownership(self):
        for n in [0, 1, 299, 300, 301, 400, 401, 511, 512, 513, 600, 601, 5003]:
            ownership = np.zeros(n, dtype=int)
            for w in windows(n):
                self.assertLessEqual(w.right - w.left, 400)
                self.assertTrue(w.left <= w.core_left < w.core_right <= w.right)
                ownership[w.core_left:w.core_right] += 1
            self.assertTrue(np.all(ownership == 1), n)

    def test_metrics_exclude_none(self):
        m = summarize(confusion([0, 1, 2, 3], [0, 1, 1, 0]))
        self.assertAlmostEqual(m['punctuation_micro_precision'], .5)
        self.assertAlmostEqual(m['punctuation_micro_recall'], 1 / 3)
        self.assertAlmostEqual(m['punctuation_micro_f1'], .4)

    def test_real_bert_forward_and_batch_alignment(self):
        # A randomly initialized tiny BERT is only a plumbing test, never an experiment.
        with tempfile.TemporaryDirectory() as d:
            v = Path(d) / 'vocab.txt'
            v.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '天', '地', '人', '1', 'a']), encoding='utf-8')
            tokenizer = BertTokenizer.from_pretrained(d)
            model = BertForTokenClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=24,
                       num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=8,
                       hidden_dropout_prob=0, attention_probs_dropout_prob=0))
            decoder = Decoder(model, tokenizer, 'cpu', amp=False)
            texts = ['', '天地人' * 201, 'Ａ１天地人' * 101, '天']
            self.assertEqual(len(encode_chars(tokenizer, texts[2])), len(texts[2]))
            one, _ = decoder.fixed(texts, 1)
            many, _ = decoder.fixed(texts, 4)
            for t, x, y in zip(texts, one, many):
                self.assertEqual(len(t), len(x))
                np.testing.assert_array_equal(x, y)
            seq, _ = decoder.sequential(texts)
            self.assertEqual([len(x) for x in seq], [len(x) for x in texts])


if __name__ == '__main__':
    unittest.main()
